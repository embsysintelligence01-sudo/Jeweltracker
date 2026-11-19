"""Saved-video regression fixtures; no camera or network required."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from backup_analysis import AnalysisEngine, RecordedSource
from backup_recovery import BackupWorker, configuration
from acid_application import AcidApplicationVerifier
from audit_report import audit_outcome


class EmptyModels:
    def marks(self, frame):return np.zeros(frame.shape[:2],np.uint8)


class SeedRefiner:
    stats={}
    models={}
    def __init__(self):self.calls=0
    def refine(self,image,seed,*args):
        self.calls+=1
        return dict(mask=seed.copy(),score=.99,point_coords=[],point_labels=[],prompt_box=[])


class BackupVideoTests(unittest.TestCase):
    def fixture(self,root,acid=True,transient=False,different=False):
        root=Path(root)
        quad=np.float32([[40,30],[239,30],[239,229],[40,229]])
        before=np.full((260,300,3),210,np.uint8)
        texture=np.random.default_rng(9).integers(60,71,(200,200,3),dtype=np.uint8)
        before[30:230,40:240]=texture
        cv2.imwrite(str(root/'before.png'),before)
        video=root/'operation.avi'
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),5,(300,260))
        self.assertTrue(writer.isOpened())
        entries=[]
        for i in range(66):
            frame=before.copy()
            if i>=10:frame[110:140,125:141]=145
            if acid and i>=30 and (not transient or i==30):
                if different:frame[160:185,185:203]=22
                else:frame[108:142,123:143]=22
            writer.write(frame)
            entries.append(dict(timestamp=100+i*.2,frame_index=i,video=str(video),file='video:'+str(i)))
        writer.release()
        contact=np.zeros((512,512),np.uint8);contact[190:300,200:290]=255
        cv2.imwrite(str(root/'contact.png'),contact)
        return dict(directory=str(root/'result'),cycle_id='fixture',entries=entries,
            before=dict(file=str(root/'before.png'),timestamp=99,quad=quad.tolist()),
            poses=[dict(timestamp=e['timestamp'],quad=quad.tolist()) for e in entries],
            started=100,finished=113,returned=104,acid_onset=None,application_candidate_at=None,
            acid_baseline=None,contact_region=str(root/'contact.png'),rubbing_verified=True,
            contact_samples=[],rub=None,placement=None,need_acid_area=True)

    def engine(self):
        config=configuration(dict(max_frames=24,min_sharpness=.05))
        refiner=SeedRefiner()
        engine=AnalysisEngine(config,refiner=refiner,models=EmptyModels())
        engine.acid_detector=Mock(predict=Mock(return_value=([],0)))
        return engine,refiner

    def test_saved_video_recovers_missed_overlap_without_live_trigger(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=self.fixture(tmp)
            engine,refiner=self.engine()
            result=engine.analyze(job)
            self.assertEqual('VERIFIED',result['status'],result)
            self.assertEqual('EDGESAM_BACKUP',result['source'])
            self.assertTrue(Path(result['rub']['mask']).is_file())
            self.assertTrue(Path(result['acid']['mask']).is_file())
            self.assertAlmostEqual(106,result['acid']['timestamp'],places=2)
            self.assertLess(result['timings']['decoded_frames'],30)
            self.assertLessEqual(len(result['diagnostics']['initial_frame_indices']),4)
            self.assertGreater(refiner.calls,0)
            self.assertIn('video_decoding_seconds',result['timings'])
            self.assertIn('acid_verification_seconds',result['timings'])
            from acid_area import compare_area
            rub=cv2.resize(cv2.imread(result['rub']['mask'],0),(256,256),interpolation=cv2.INTER_NEAREST)
            acid=cv2.imread(result['acid']['mask'],0)
            self.assertEqual('ACID_OVERLAPS_NEW_RUB',compare_area(acid,rub)['acid_target_status'])

    def test_different_location_is_not_silently_marked_overlapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=self.fixture(tmp,different=True)
            engine,_=self.engine();result=engine.analyze(job)
            self.assertIn('acid',result,result)
            from acid_area import compare_area
            rub=cv2.resize(cv2.imread(result['rub']['mask'],0),(256,256),interpolation=cv2.INTER_NEAREST)
            acid=cv2.imread(result['acid']['mask'],0)
            self.assertEqual('ACID_AT_DIFFERENT_LOCATION',compare_area(acid,rub)['acid_target_status'])

    def test_transient_tool_cannot_become_acid(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine,_=self.engine();result=engine.analyze(self.fixture(tmp,transient=True))
            self.assertEqual('REVIEW_REQUIRED',result['status'])
            self.assertNotIn('acid',result)

    def test_missing_models_falls_back_without_fabricating_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine,_=self.engine();engine.refiner=None;engine.refiner_error='Missing model'
            result=engine.analyze(self.fixture(tmp))
            self.assertEqual('REVIEW_REQUIRED',result['status'])
            self.assertNotIn('rub',result)
            self.assertNotIn('acid',result)

    def test_no_contact_cannot_verify_a_new_mark(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=self.fixture(tmp);job['contact_region']=None;job['rubbing_verified']=False
            engine,refiner=self.engine();engine.contact_observer=lambda f,s:None
            result=engine.analyze(job)
            self.assertEqual('REVIEW_REQUIRED',result['status'])
            self.assertNotIn('rub',result)

    def test_video_reader_uses_sidecar_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=self.fixture(tmp);job['entries'][3]['timestamp']=100.631
            source=RecordedSource(job['entries'])
            try:
                selected=source.between(100.62,100.64)
                self.assertEqual([100.631],[e['timestamp'] for e in selected])
                self.assertEqual((260,300,3),source.read(selected[0]).shape)
            finally:source.close()

    def test_worker_timeout_kills_process_and_returns_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker=BackupWorker(configuration(dict(timeout_seconds=.1)))
            process=Mock();process.stdout=Mock();process.poll.return_value=None
            worker.process=process
            with patch('backup_recovery.select.select',return_value=([],[],[])):
                result=worker.analyze({},tmp)
            self.assertEqual('REVIEW_REQUIRED',result['status'])
            self.assertTrue(result['timed_out'])
            process.terminate.assert_called_once()
            self.assertIsNone(worker.process)

    def test_persistent_worker_is_isolated_and_low_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker=BackupWorker(configuration(dict(timeout_seconds=10)))
            try:
                request=dict(entries=[],directory=tmp,started=1,finished=2,poses=[])
                first=worker.analyze(request,tmp)
                second=worker.analyze(request,tmp)
                self.assertEqual('REVIEW_REQUIRED',first['status'])
                self.assertEqual(first['worker']['pid'],second['worker']['pid'])
                self.assertNotEqual(os.getpid(),first['worker']['pid'])
                self.assertGreaterEqual(first['worker']['nice'],15)
                self.assertEqual(1,len(first['worker']['cpus']))
            finally:worker.close()

    def test_pending_backup_does_not_block_next_jewel(self):
        import threading
        from test_sop_revision import monitor_case,snap
        entered=threading.Event();release=threading.Event()
        with monitor_case() as (monitor,events):
            monitor.backup_config['enabled']=True
            monitor.backup_worker=Mock()
            def analyze(*args):
                entered.set()
                release.wait(5)
                return dict(status='REVIEW_REQUIRED',reason='Fixture lacks contact',timings={})
            monitor.backup_worker.analyze.side_effect=analyze
            monitor.step(snap(100,held=1))
            first=monitor.active
            monitor._finish(101)
            try:
                self.assertTrue(entered.wait(3))
                self.assertEqual([],monitor.tests)
                self.assertTrue(any(k=='backup_pending' for k,_ in events))
                monitor.step(snap(102,held=2))
                self.assertEqual(2,monitor.active['jewel_id'])
                self.assertNotEqual(first['id'],monitor.active['id'])
            finally:release.set()
            monitor.drain_finalization()
            self.assertEqual('REVIEW_REQUIRED',monitor.tests[0]['sop_review_status'])

    def test_existing_classification_keeps_padding_and_complete_frames(self):
        from test_sop_revision import monitor_case
        from test_confirmed_acid_crop import ConfirmedAcidCropTests,PREDICTION
        from test_acid_application import DETECTION
        from acid_test.flow import pad_bbox
        with monitor_case() as (monitor,events):
            monitor.backup_config['enabled']=True
            monitor.backup_worker=Mock()
            monitor.backup_worker.analyze.return_value=dict(status='REVIEW_REQUIRED',timings={})
            attempt,_=ConfirmedAcidCropTests().prepare(monitor)
            monitor._finish(106.1)
            monitor.drain_finalization()
            record=monitor.tests[0]
            self.assertEqual(PREDICTION,record['prediction'])
            self.assertEqual(list(pad_bbox(DETECTION['bbox'],.45,400,300)),record['reaction_bbox'])
            self.assertEqual('COMPLETE',record['reaction_coverage_status'])
            monitor.classifier.predict.assert_called_once()
            self.assertEqual('REVIEW_REQUIRED',record['sop_review_status'])

    def test_unresolved_report_cannot_be_a_pass(self):
        self.assertEqual('REVIEW REQUIRED',audit_outcome(dict(sop_review_status='REVIEW_REQUIRED',prediction=None)))
        self.assertEqual('PENDING',audit_outcome(dict(sop_review_status='PENDING')))


if __name__=='__main__':unittest.main()
