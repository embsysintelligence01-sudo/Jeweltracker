import threading
import unittest
from unittest.mock import Mock,patch

import cv2
import numpy as np
import jewel_tracker_5 as app
from jewel_identity import IdentityMatcher


SLIC=hasattr(cv2,'ximgproc') and hasattr(cv2.ximgproc,'createSuperpixelSLIC')


@unittest.skipUnless(SLIC,'Run with .venv-slic for version 6 identity matching')
class TrackingV6Tests(unittest.TestCase):
    def worker(self):
        return app.Worker(0,[(0,0),(399,0),(399,299),(0,299)],
                          stone_pts=[(300,200),(390,200),(390,290),(300,290)],continuous=True)

    def detection(self,x,t=100):
        box=np.array([x-10,90.,x+10,110.])
        return app.Det(box,np.array([float(x),100.]),400,.95)

    def test_slic_matcher_is_enabled_without_loading_weights_on_ui_thread(self):
        with patch.object(app,'JewelIdModel') as extractor:
            w=self.worker()
            self.assertIsInstance(w.id_model,IdentityMatcher)
            self.assertFalse(w.id_model.ready)
            extractor.assert_not_called()

    def test_contested_motion_uses_signature_to_preserve_jewel_ids(self):
        w=self.worker();w.mode='LOCKED'
        w.tracks=[app.Track(1,self.detection(100),100),app.Track(2,self.detection(130),100)]
        w.id_model.ready=True
        w.id_model.signature_scores=Mock(side_effect=[{1:.05,2:.98},{1:.98,2:.05}])
        w.id_model.predict=Mock(return_value=(None,0.))
        detections=[self.detection(110),self.detection(120)]
        for name in ('update_rubbing','update_acid','evaluate','check_done'):
            setattr(w,name,Mock())
        w.update_tracks(detections,100.1,np.full((300,400,3),220,np.uint8))
        np.testing.assert_array_equal(detections[1].box,w.tracks[0].box)
        np.testing.assert_array_equal(detections[0].box,w.tracks[1].box)
        self.assertEqual([1,2],[tr.id for tr in w.tracks])
        self.assertEqual(2,w.id_model.signature_scores.call_count)

    def test_uncontested_tracking_does_not_run_signature_inference(self):
        w=self.worker();w.mode='LOCKED'
        w.tracks=[app.Track(1,self.detection(60),100),app.Track(2,self.detection(340),100)]
        w.id_model.ready=True
        w.id_model.signature_scores=Mock()
        w.id_model.predict=Mock(return_value=(None,0.))
        w.score_contested([self.detection(60),self.detection(340)],[0,1],
                          np.full((300,400,3),220,np.uint8),100.01)
        w.id_model.signature_scores.assert_not_called()

    def test_old_gallery_cannot_publish_after_reset_or_block_camera_lock(self):
        w=self.worker();w.mode='LOCKED'
        w.tracks=[app.Track(1,self.detection(100),100)]
        w.negative_crops=lambda frame:[]
        entered=threading.Event();release=threading.Event()
        def train(candidate,*args):
            entered.set();release.wait(2)
            candidate.ready=True;candidate.check_answered=1.
            return 1.
        with patch.object(app,'JewelIdModel',return_value=object()),patch.object(IdentityMatcher,'train',train):
            w.learn_ids(np.full((300,400,3),220,np.uint8))
            self.assertTrue(entered.wait(1))
            acquired=w.lock.acquire(timeout=.1)
            self.assertTrue(acquired)
            if acquired:w.lock.release()
            previous=w.id_model
            w.arm();release.set();w.id_training_thread.join(2)
            self.assertIs(previous,w.id_model)
            self.assertFalse(w.id_model.ready)


if __name__=='__main__':unittest.main()
