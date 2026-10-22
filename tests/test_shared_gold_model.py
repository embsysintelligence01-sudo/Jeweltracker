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
             patch.object(tracker,'YOLO') as yolo, \
             patch.object(tracker,'RTSPGrabber') as grabber, \
             patch.object(tracker,'StoneWatcher') as stone, \
             patch.object(tracker,'RubbingWatcher') as rubbing:
            grabber.return_value.start.side_effect=lambda:setattr(worker,'running',False)
            for watcher in (grabber,stone,rubbing):watcher.return_value.is_alive.return_value=False
            worker.run()
            shared.assert_called_once_with(tracker.MODEL_PATH)
            yolo.assert_called_once_with(tracker.STONE_MODEL_PATH,task='segment')
            self.assertIs(worker.model,shared.return_value)
            self.assertIs(rubbing.call_args.args[1],worker.model)
            self.assertEqual(tracker.MODEL_PATH,tracker.RUB_GOLD_MODEL_PATH)
            self.assertEqual('rojeweltrack.onnx',Path(tracker.MODEL_PATH).name)

    def test_real_model_supports_bench_and_stone_crop_paths(self):
        model=tracker.SharedGoldModel()
        self.addCleanup(model.close)
        path=next((Path(tracker.SCRIPT_DIR)/'captures').glob('*_capture.jpg'))
        frame=cv2.imread(str(path))
        result=model.predict(frame,imgsz=640,conf=tracker.CONF,device='cpu',verbose=False)[0]
        self.assertIsNotNone(result.boxes)
        self.assertEqual({0:'gold'},result.names)
        self.assertEqual([320,320],model.backend.predictor.imgsz)
        stone=tracker.find_stone_cv(frame,tracker.to_poly([(0,0),(frame.shape[1]-1,0),(frame.shape[1]-1,frame.shape[0]-1),(0,frame.shape[0]-1)]))
        if stone is None:
            h,w=frame.shape[:2];stone=(w//3,h//3,2*w//3,2*h//3)
        evidence=tracker.rubbing_evidence(model,frame,stone)
        if evidence is not None:
            self.assertEqual(2,len(evidence['c']))
            self.assertGreater(evidence['mask'].sum(),0)
        self.assertEqual(['CPUExecutionProvider'],model.backend.predictor.model.session.get_providers())


if __name__=='__main__':unittest.main()
