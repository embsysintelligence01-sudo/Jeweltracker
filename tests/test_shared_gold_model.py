import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from concurrent.futures import ThreadPoolExecutor
import time
import unittest
from unittest.mock import Mock,patch
from pathlib import Path

import cv2
import numpy as np
import jewel_tracker_5 as tracker


class SharedGoldTests(unittest.TestCase):
    def test_one_backend_serializes_tracking_and_rubbing_calls(self):
        with patch.object(tracker,'YOLO') as yolo:
            model=tracker.SharedGoldModel()
            self.addCleanup(model.close)
            active=0
            peak=0
            def predict(source,**kwargs):
                nonlocal active,peak
                active+=1;peak=max(peak,active)
                time.sleep(.01)
                self.assertEqual(320,kwargs['imgsz'])
                self.assertFalse(kwargs['stream'])
                active-=1
                return kwargs['conf']
            yolo.return_value.predict.side_effect=predict
            with ThreadPoolExecutor(max_workers=2) as pool:
                first=pool.submit(model.predict,np.zeros((320,320,3),np.uint8),imgsz=640,conf=.15)
                second=pool.submit(model.predict,np.zeros((320,320,3),np.uint8),imgsz=320,conf=.70)
                self.assertEqual(.15,first.result())
                self.assertEqual(.70,second.result())
            yolo.assert_called_once_with(tracker.MODEL_PATH,task='segment')
            self.assertEqual(1,peak)

    def test_worker_passes_the_same_model_to_rubbing(self):
        with patch.object(tracker,'JewelIdModel'):
            worker=tracker.Worker(0,[(0,0),(100,0),(100,100)])
        with patch.object(tracker,'SharedGoldModel') as shared, \
             patch.object(tracker,'StoneOnnxModel') as stone_model, \
             patch.object(tracker,'RTSPGrabber') as grabber, \
             patch.object(tracker,'StoneWatcher') as stone, \
             patch.object(tracker,'RubbingObserver') as rubbing:
            grabber.return_value.start.side_effect=lambda:setattr(worker,'running',False)
            for watcher in (grabber,stone,rubbing):watcher.return_value.is_alive.return_value=False
            worker.run()
            shared.assert_called_once_with(tracker.MODEL_PATH)
            stone_model.assert_called_once_with(tracker.STONE_MODEL_PATH)
            self.assertIs(worker.model,shared.return_value)
            self.assertEqual(worker.stone_box,rubbing.call_args.args[0])
            self.assertEqual(tracker.MODEL_PATH,tracker.RUB_GOLD_MODEL_PATH)
            self.assertEqual('rojeweltrack.onnx',Path(tracker.MODEL_PATH).name)

    def test_real_full_frame_model_output_is_reused_for_stone_masking(self):
        model=tracker.SharedGoldModel()
        self.addCleanup(model.close)
        path=next((Path(tracker.SCRIPT_DIR)/'captures').glob('*_capture.jpg'))
        frame=cv2.imread(str(path))
        result=model.predict(frame,imgsz=640,conf=tracker.CONF,device='cpu',verbose=False)[0]
        self.assertIsNotNone(result.boxes)
        self.assertEqual({0:'gold'},result.names)
        self.assertEqual([320,320],model.backend.predictor.imgsz)
        stone_model=tracker.StoneOnnxModel(tracker.STONE_MODEL_PATH)
        stone=tracker.StoneWatcher(None,stone_model,lambda:None).model_box(frame)
        self.assertIsNotNone(stone)
        self.assertEqual({0:'St'},stone_model.names)
        evidence=tracker.rubbing_evidence(result.masks.xy if result.masks is not None else [],result.boxes.conf.cpu().numpy(),frame.shape,stone)
        if evidence is not None:
            self.assertEqual(2,len(evidence['c']))
            self.assertGreater(evidence['mask'].sum(),0)
        self.assertEqual(['CPUExecutionProvider'],model.backend.predictor.model.session.get_providers())


if __name__=='__main__':unittest.main()
