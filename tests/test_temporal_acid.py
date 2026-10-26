import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2
import jewel_tracker_5 as tracker
from acid_test.sequence import AcidSequence
from acid_test.onnx_detector import Detection, OnnxAcidDetector
from acid_test.integration import AcidTestThread, AcidModelCache
from acid_test.runtime import TemporalMobileNet, latest_temporal_model_path
from acidphintegration.auto_sop import AutoSOP, analyze_paper, FRESH_FRAMES


FRAME = np.full((100, 160, 3), 80, np.uint8)
STONE = (10, 10, 150, 90)
STREAK = (40, 30, 70, 60)
BOX = (42, 32, 68, 58)


def track(jid):
    mask=np.zeros(FRAME.shape[:2],np.uint8)
    mask[STREAK[1]:STREAK[3],STREAK[0]:STREAK[2]]=255
    target=SimpleNamespace(roi_mask=mask,streak=STREAK,stone=STONE,shape=FRAME.shape,
                           locate=lambda frame:dict(roi_mask=mask,stone=STONE,streak=STREAK))
    return SimpleNamespace(id=jid, rubbed=False, acid_done=False, acid_result=None,
                           acid_box=None, done=False, thumb=None, rub_img=None,
                           rub_boxes=[STREAK], rub_masks=[], rub_last=0, rub_time=0,
                           det_seen=0, state='OK', pos=np.array([50., 40.]), move=3.,
                           vel=np.zeros(2), settle_since=None,new_rub_target=target,
                           acid_roi_mask=mask,acid_roi_stone=STONE)


def record(label='non_gold'):
    return dict(detector_bbox=list(BOX), record_path='test/manifest.json',
                prediction=dict(prediction=label, confidence=.94),
                ph_verification=dict(result='ACIDIC_REACTION_VERIFIED'))


def paper_detection(reacted=False):
    hsv = np.full((20, 60, 3), (25, 200, 220), np.uint8)
    if reacted:
        hsv[:, :18] = (175, 200, 220)
    return dict(bbox=(20, 20, 80, 40), score=.95, mode='obb',
                crop=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),
                mask=np.full((20, 60), 255, np.uint8))


class PaperModel:
    def __init__(self, precolored=False):
        self.calls = 0
        self.precolored = precolored

    def detect(self, *args):
        self.calls += 1
        # The cache warms up with a synthetic frame before any live frames.
        return [paper_detection(self.precolored or self.calls > FRESH_FRAMES + 1)]


class PhSOPTests(unittest.TestCase):
    def metrics(self, reacted=False):
        d = paper_detection(reacted)
        return d, analyze_paper(d['crop'], d['mask'])

    def test_requires_fresh_paper_then_sustained_reaction(self):
        sop = AutoSOP()
        used, pink = self.metrics(True)
        for i in range(12):
            self.assertFalse(sop.update(used, pink, i))
        self.assertEqual('WAIT_FRESH', sop.state)
        fresh, yellow = self.metrics()
        for i in range(FRESH_FRAMES):
            self.assertFalse(sop.update(fresh, yellow, i))
        self.assertEqual('WATCH_REACTION', sop.state)
        self.assertFalse(sop.update(used, pink, 20))
        self.assertTrue(sop.update(used, pink, 21))
        self.assertEqual('VERIFIED', sop.state)
        self.assertFalse(sop.update(used, pink, 22))
        self.assertEqual('WAIT_FRESH', AutoSOP().state)  # a new jewel starts over

    def test_occlusion_and_paper_loss_cannot_pass(self):
        sop = AutoSOP()
        fresh, yellow = self.metrics()
        for i in range(FRESH_FRAMES):
            sop.update(fresh, yellow, i)
        used, pink = self.metrics(True)
        covered = dict(pink, dark=.5)
        for i in range(10):
            self.assertFalse(sop.update(used, covered, 20+i))
        for i in range(9):
            self.assertFalse(sop.update(None, None, 40+i))
        self.assertEqual('WAIT_FRESH', sop.state)
        self.assertFalse(sop.update(used, pink, 50))



class CaptureTests(unittest.TestCase):
    def prepare(self, observation_seconds=1):
        self.now=0.0
        seq=AcidSequence(STONE,STREAK,observation_seconds=observation_seconds,clock=lambda:self.now)
        for i in range(22):
            self.now=i/10
            seq.push(FRAME,self.now)
        self.assertEqual('WAIT_FOR_ACID',seq.state)
        return seq

    def test_requires_three_matching_detections_on_the_selected_streak(self):
        seq=self.prepare()
        for t in (2.2,2.4,2.6):
            seq.push(FRAME,t)
            seq.accept_detections([Detection((110,30,130,50),.99)],t)
        self.assertEqual('WAIT_FOR_ACID',seq.state)  # old acid elsewhere
        for t in (2.8,3.0):
            seq.push(FRAME,t);seq.accept_detections([Detection(BOX,.9)],t)
        self.assertEqual('WAIT_FOR_ACID',seq.state)
        self.now=3.2
        seq.push(FRAME,3.2);seq.accept_detections([Detection(BOX,.9)],3.2)
        self.assertEqual('OBSERVING',seq.state)
        self.assertGreater(len(seq.baseline_frames),1)
        self.assertGreater(len(seq.transition_frames),1)
        for t in (3.4,3.6,3.8,4.0,4.21):seq.push(FRAME,t)
        self.assertEqual('PROCESSING',seq.state)
        self.assertGreater(len(seq.observation_frames),1)
        self.assertEqual(seq.baseline_frames[0].shape,seq.observation_frames[0].shape)

    def test_miss_resets_confirmation(self):
        seq=self.prepare()
        seq.accept_detections([Detection(BOX,.9)],2.2)
        seq.accept_detections([],2.4)
        self.assertEqual(0,seq.hits)

    def test_slow_buffer_decoding_does_not_consume_observation_time(self):
        for decoding_seconds in (0.0, 2.5, 6.0):
            with self.subTest(decoding_seconds=decoding_seconds):
                seq=self.prepare(observation_seconds=5)
                for t in (2.2,2.4):
                    self.now=t
                    seq.push(FRAME,t)
                    seq.accept_detections([Detection(BOX,.9)],t)
                self.now=2.6
                seq.push(FRAME,self.now)
                count=len(seq.transition_buffer)
                decode=tracker.cv2.imdecode
                def slow_decode(*args,**kwargs):
                    image=decode(*args,**kwargs)
                    self.now+=decoding_seconds/count
                    return image
                with patch('acid_test.sequence.cv2.imdecode',side_effect=slow_decode):
                    seq.accept_detections([Detection(BOX,.9)],2.6)
                started=seq.detected
                self.assertAlmostEqual(2.6+decoding_seconds,started)
                self.assertEqual('OBSERVING',seq.state)
                self.assertEqual([],seq.observation_frames)
                self.assertIn('5.0s',seq.instruction(started))
                for i in range(25):
                    seq.push(FRAME,started+.01+i*.2)
                self.assertEqual('OBSERVING',seq.state)
                self.assertEqual(25,len(seq.observation_frames))
                self.assertAlmostEqual(.01,seq.observation_offsets[0])
                seq.push(FRAME,started+5.01)
                self.assertEqual('PROCESSING',seq.state)
                self.assertGreaterEqual(seq.observation_offsets[-1],5.0)

    def test_resolution_change_and_wait_timeout_are_errors(self):
        seq=self.prepare()
        with self.assertRaisesRegex(RuntimeError,'resolution'):
            seq.push(np.zeros((120,160,3),np.uint8),3)
        with self.assertRaisesRegex(RuntimeError,'120 seconds'):seq.push(FRAME,123)

    def test_prediction_capture_is_saved_without_a_ground_truth_label(self):
        seq=self.prepare()
        for t in (2.2,2.4,2.6):
            self.now=t
            seq.push(FRAME,t);seq.accept_detections([Detection(BOX,.9)],t)
        seq.push(FRAME,2.8);seq.push(FRAME,3.61)
        context=dict(token='token',jewel_id=2,batch_id='batch')
        thread=AcidTestThread(context,1,'22k-batch')
        thread.ph_verification=record()['ph_verification']
        thread.ph_images={'baseline':paper_detection()['crop'], 'reaction':paper_detection(True)['crop']}
        with tempfile.TemporaryDirectory() as tmp, patch('acid_test.integration.LAB_ROOT',Path(tmp)):
            result=thread.save_capture(seq,record()['prediction'])
            manifest=json.loads(Path(result['record_path']).read_text())
            self.assertEqual(2,manifest['jewel_id'])
            self.assertIsNone(manifest['actual_label'])
            self.assertEqual('ACIDIC_REACTION_VERIFIED',manifest['ph_verification']['result'])
            for filename in manifest['ph_verification']['images'].values():
                self.assertTrue((Path(result['record_path']).parent/filename).is_file())
            self.assertTrue((Path(result['record_path']).parent/manifest['acid_preview']).is_file())
            self.assertEqual('jewel_acid_prediction_v1',manifest['dataset_version'])
            for phase in ('baseline','transition','observation'):
                for entry in manifest[phase+'_frames']:
                    self.assertTrue((Path(result['record_path']).parent/entry['path']).is_file())


class TrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=tracker.QtWidgets.QApplication.instance() or tracker.QtWidgets.QApplication([])

    def setUp(self):
        with patch.object(tracker,'JewelIdModel',return_value=Mock()):
            self.worker=tracker.Worker(0,[(0,0),(159,0),(159,99),(0,99)])
        self.worker.tracks=[track(1),track(2),track(3)]
        self.worker.rub_reference_ready=True
        self.worker.stone_box=Mock(return_value=STONE)
        self.worker.grabber=SimpleNamespace(latest=lambda:(FRAME,time.time()))
        self.worker.rubber=SimpleNamespace(active=True,reset=Mock(),snapshot=lambda:({'box':STREAK,'mask':np.ones((5,5),np.uint8),'origin':(0,0)},True,10.0))
        self.worker.rub_spot_image=Mock(return_value=None)

    def rub_second(self):
        tr=self.worker.tracks[1]
        tr.rub_last=9.9;tr.rub_time=tracker.RUB_CONFIRM_S
        self.worker.rub_track=tr
        self.worker.update_rubbing(10)
        self.assertTrue(tr.rubbed)
        self.assertEqual(2,self.worker.acid_pending_id)
        return tr

    def test_j2_then_j1_and_result_belongs_only_to_selected_jewel(self):
        tr=self.rub_second()
        self.worker.update_rubbing(10.1)
        self.assertFalse(self.worker.tracks[0].rubbed)
        self.assertFalse(self.worker.rubber.active)
        context=self.worker.begin_acid_test()
        with self.assertRaisesRegex(RuntimeError,'already running'):self.worker.begin_acid_test()
        self.assertFalse(self.worker.finish_acid_test(dict(token=context['token'],jewel_id=1,result=record())))
        self.assertFalse(tr.acid_done)
        self.assertTrue(self.worker.finish_acid_test(dict(token=context['token'],jewel_id=2,result=record())))
        self.assertTrue(tr.acid_done)
        self.assertIsNone(self.worker.acid_pending_id)
        self.assertFalse(self.worker.tracks[0].acid_done)
        self.assertEqual(2,tr.acid_result.get('jewel_id',2))
        # The next jewel can be J1 even though J3 is still untested.
        first=self.worker.tracks[0]
        first.rub_last=9.9;first.rub_time=tracker.RUB_CONFIRM_S
        self.worker.rub_track=first
        self.worker.rub_reference_ready=True
        self.worker.update_rubbing(10)
        self.assertEqual(1,self.worker.acid_pending_id)

    def test_result_without_ph_verification_cannot_complete_a_jewel(self):
        tr=self.rub_second()
        context=self.worker.begin_acid_test()
        unverified=record()
        unverified.pop('ph_verification')
        self.assertFalse(self.worker.finish_acid_test(dict(token=context['token'],jewel_id=2,result=unverified)))
        self.assertFalse(tr.acid_done)
        self.assertEqual(2,self.worker.acid_pending_id)

    def test_ph_bypass_is_accepted_only_for_the_attempt_started_with_it_off(self):
        tr=self.rub_second()
        context=self.worker.begin_acid_test()
        skipped=record()
        skipped['ph_verification']=dict(result='SKIPPED_BY_OPERATOR',enabled=False)
        self.assertFalse(self.worker.finish_acid_test(dict(token=context['token'],jewel_id=2,result=skipped)))
        self.assertFalse(tr.acid_done)
        self.worker.fail_acid_test(context['token'],'retry')
        context=self.worker.begin_acid_test(ph_verification_enabled=False)
        self.assertFalse(context['ph_verification_enabled'])
        self.assertTrue(self.worker.finish_acid_test(dict(token=context['token'],jewel_id=2,result=skipped)))
        self.assertTrue(tr.acid_done)

    def test_ph_checkbox_defaults_on_and_is_snapshotted_when_test_starts(self):
        with patch.object(tracker.QtCore.QTimer,'singleShot'),patch.object(tracker,'write_log'):
            window=tracker.MainWindow()
            window.worker=self.worker
            self.rub_second()
            self.assertTrue(window.chk_ph_verification.isChecked())
            window.chk_ph_verification.setChecked(False)
            self.assertIn('OFF',window.chk_ph_verification.text())
            self.assertNotIn('verify pH paper',window.acid_instruction.text())
            with patch.object(tracker,'AcidTestThread') as thread:
                window.start_acid_test()
                self.assertFalse(thread.call_args.args[0]['ph_verification_enabled'])
                self.assertFalse(window.chk_ph_verification.isEnabled())
                thread.return_value.start.assert_called_once()
            window.acid_thread=window.acid_owner=None
            window.worker=None
            window.close()

    def test_cancel_and_reset_reject_old_results(self):
        tr=self.rub_second();context=self.worker.begin_acid_test()
        self.assertTrue(self.worker.fail_acid_test(context['token'],'cancelled'))
        self.assertEqual(2,self.worker.acid_pending_id)
        self.assertFalse(tr.acid_done)
        retry=self.worker.begin_acid_test()
        self.assertNotEqual(context['token'],retry['token'])
        self.assertFalse(self.worker.finish_acid_test(dict(token=context['token'],jewel_id=2,result=record())))
        self.worker.arm()
        self.assertFalse(self.worker.finish_acid_test(dict(token=retry['token'],jewel_id=2,result=record())))

    def test_cannot_start_without_rubbing_or_a_fresh_camera_frame(self):
        with self.assertRaisesRegex(RuntimeError,'Rub a jewel'):self.worker.begin_acid_test()
        self.rub_second()
        self.worker.grabber.latest=lambda:(FRAME,time.time()-10)
        with self.assertRaisesRegex(RuntimeError,'stale'):self.worker.begin_acid_test()
        self.assertIsNone(self.worker.acid_token)

    def test_done_requires_classification_even_without_legacy_acid_watcher(self):
        tr=self.rub_second()
        tr.settle_since=1
        self.worker.on_stone=Mock(return_value=False)
        self.worker.check_done(tr,True,10)
        self.assertFalse(tr.done)
        tr.acid_done=True
        self.worker.check_done(tr,True,10)
        self.assertTrue(tr.done)

    def test_table_button_and_result_for_j2(self):
        with patch.object(tracker.QtCore.QTimer,'singleShot'),patch.object(tracker,'write_log'):
            window=tracker.MainWindow()
        window.worker=self.worker
        self.worker.jewel_table.connect(window.show_table)
        self.rub_second()
        self.assertTrue(window.btn_start_acid.isEnabled())
        self.assertIn('J2',window.btn_start_acid.text())
        self.assertIsNone(window.acid_models._models)  # rubbing alone does not run acid models
        context=self.worker.begin_acid_test()
        window.acid_owner=self.worker
        window.acid_ui_token=context['token']
        crop=np.full((26,26,3),125,np.uint8)
        window.show_acid_crop(dict(token='wrong-token',jewel_id=2,crop=crop))
        self.assertIsNone(getattr(self.worker.tracks[1],'acid_crop',None))
        window.show_acid_crop(dict(token=context['token'],jewel_id=2,crop=crop))
        self.assertFalse(window.table.cellWidget(1,6).pixmap().isNull())
        self.assertIsNone(window.table.cellWidget(0,6).pixmap())
        self.worker.finish_acid_test(dict(token=context['token'],jewel_id=2,result=record('gold_22k')))
        self.assertFalse(window.table.cellWidget(1,6).pixmap().isNull())
        self.assertIn('22K GOLD',window.table.item(1,5).text())
        self.assertEqual('',window.table.item(0,5).text())
        self.assertFalse(window.btn_start_acid.isEnabled())
        window.worker=None
        window.close()

    def test_model_error_remains_visible_after_controls_refresh(self):
        with patch.object(tracker.QtCore.QTimer,'singleShot'),patch.object(tracker,'write_log'):
            window=tracker.MainWindow()
            window.worker=self.worker
            self.worker.jewel_table.connect(window.show_table)
            self.rub_second()
            context=self.worker.begin_acid_test()
            window.acid_owner=self.worker
            window.acid_ui_token=context['token']
            window.acid_failed(dict(token=context['token'],jewel_id=2,
                                    message='OpenVINO model load failed',traceback='load traceback'))
            window.refresh_acid_controls()
            self.assertIn('OpenVINO model load failed',window.acid_instruction.text())
            self.assertTrue(window.btn_start_acid.isEnabled())
            self.assertFalse(self.worker.tracks[1].acid_done)
            window.worker=None
            window.close()

    def test_model_ready_phase_is_logged_once(self):
        with patch.object(tracker.QtCore.QTimer,'singleShot'),patch.object(tracker,'write_log'):
            window=tracker.MainWindow()
            window.worker=self.worker
            self.rub_second()
            context=self.worker.begin_acid_test()
            window.acid_owner=self.worker
            window.acid_ui_token=context['token']
            window.add_log=Mock()
            payload=dict(token=context['token'],jewel_id=2,phase='PH_READY',
                         message='pH model ready',model_path='/test/acidph.xml')
            window.acid_progress(payload)
            window.acid_progress(payload)
            window.add_log.assert_called_once()
            self.assertIn('/test/acidph.xml',window.add_log.call_args.args[0])
            window.worker=None
            window.close()

    def test_ready_button_requires_this_attempts_waiting_phase_and_one_click(self):
        with patch.object(tracker.QtCore.QTimer,'singleShot'),patch.object(tracker,'write_log'):
            window=tracker.MainWindow()
            window.worker=self.worker
            self.rub_second()
            context=self.worker.begin_acid_test()
            thread=Mock()
            thread.confirm_stone_ready.return_value=True
            window.acid_thread=thread
            window.acid_owner=self.worker
            window.acid_ui_token=context['token']
            window.acid_progress(dict(token=context['token'],jewel_id=2,
                                      phase='PH_FRESH',message='Show fresh paper'))
            self.assertFalse(window.btn_stone_ready.isEnabled())
            window.acid_progress(dict(token='old-token',jewel_id=2,
                                      phase='WAIT_FOR_STONE_READY',message='old test'))
            self.assertFalse(window.btn_stone_ready.isEnabled())
            window.acid_progress(dict(token=context['token'],jewel_id=2,
                                      phase='WAIT_FOR_STONE_READY',message='Click Ready for stone test'))
            self.assertTrue(window.btn_stone_ready.isEnabled())
            self.assertFalse(window.btn_start_acid.isEnabled())
            window.btn_stone_ready.click()
            self.assertFalse(window.btn_stone_ready.isEnabled())
            window.refresh_acid_controls()
            self.assertFalse(window.btn_stone_ready.isEnabled())
            window.btn_stone_ready.click()
            thread.confirm_stone_ready.assert_called_once()
            window.acid_thread=None
            window.worker=None
            window.close()

    def test_shutdown_retains_worker_until_native_threads_finish(self):
        with patch.object(tracker.QtCore.QTimer,'singleShot'),patch.object(tracker,'write_log'):
            window=tracker.MainWindow()
        worker=Mock()
        worker.wait.return_value=False
        window.worker=worker
        self.assertFalse(window.stop_worker())
        self.assertIs(worker,window.worker)
        worker.wait.return_value=True
        self.assertTrue(window.stop_worker())
        self.assertIsNone(window.worker)
        window.close()


class LocalModelTests(unittest.TestCase):
    def test_models_run_without_original_lab_folder(self):
        model=TemporalMobileNet(latest_temporal_model_path())
        result=model.predict([FRAME],[FRAME],[FRAME])
        self.assertIn(result['prediction'],('gold_22k','non_gold','invalid'))
        self.assertEqual({'gold_22k','non_gold','invalid'},set(result['probabilities']))
        detector=OnnxAcidDetector(Path(tracker.SCRIPT_DIR)/'acid_test/models/acid_detector.onnx')
        detections,elapsed=detector.predict(FRAME,.6)
        self.assertIsInstance(detections,list)
        self.assertGreater(elapsed,0)



class ThreadFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=tracker.QtWidgets.QApplication.instance() or tracker.QtWidgets.QApplication([])

    def run_thread(self, stone_getter=lambda:STONE, cancel_at_baseline=False, precolored=False, cancel_during_ph=False, confirm_ready=True, cancel_waiting=False, ph_enabled=True, cancel_when_hidden=False, lose_at_baseline=False):
        grabber=SimpleNamespace(latest=lambda:(FRAME.copy(),time.time()))
        stone_lost=[False]
        context=dict(jewel_id=2,token='thread-token',batch_id='thread-batch',
                     stone=STONE,streak=STREAK,grabber=grabber,
                     stone_getter=lambda:None if stone_lost[0] else stone_getter(),
                     ph_verification_enabled=ph_enabled)
        detector=Mock();detector.predict.return_value=([Detection(BOX,.95)],1.0)
        classifier=Mock();classifier.predict.return_value=record()['prediction']
        events,completed,failed=[],[],[]
        thread=AcidTestThread(context,.12)
        def progress(p):
            events.append(p)
            if p['phase']=='WAIT_FOR_STONE_READY':
                classifier.predict.assert_not_called()
                if cancel_waiting or (cancel_when_hidden and 'not visible' in p['message']):
                    thread.cancel()
                elif confirm_ready:
                    self.assertTrue(thread.confirm_stone_ready())
                    self.assertFalse(thread.confirm_stone_ready())
            if cancel_at_baseline and p['phase']=='BASELINE':thread.cancel()
            if lose_at_baseline and p['phase']=='BASELINE':stone_lost[0]=True
            if p['phase'] in ('PH_FRESH','PH_REACTION'):
                classifier.predict.assert_not_called()
                if cancel_during_ph and p['phase']=='PH_REACTION':thread.cancel()
        thread.progress.connect(progress)
        thread.completed.connect(completed.append)
        thread.failed.connect(failed.append)
        with tempfile.TemporaryDirectory() as tmp, \
             patch('acid_test.integration.LAB_ROOT',Path(tmp)), \
             patch('acid_test.integration.OpenVINOOBB',return_value=PaperModel(precolored)) as ph_model, \
             patch('acid_test.integration.OnnxAcidDetector',return_value=detector), \
             patch('acid_test.integration.TemporalMobileNet',return_value=classifier), \
             patch('acid_test.integration.AcidSequence',side_effect=lambda stone,streak,duration:AcidSequence(stone,streak,.12,.1)):
            thread.start()
            deadline=time.monotonic()+5
            while thread.isRunning() and time.monotonic()<deadline:
                self.app.processEvents();time.sleep(.01)
            thread.cancel()
            self.assertTrue(thread.wait(2000))
            self.app.processEvents()
            if completed:
                self.assertTrue(Path(completed[0]['result']['record_path']).exists())
            if not ph_enabled:
                ph_model.assert_not_called()
            return events,completed,failed

    def test_thread_captures_detects_classifies_and_preserves_jewel_id(self):
        events,completed,failed=self.run_thread()
        self.assertFalse(failed)
        self.assertEqual(1,len(completed))
        self.assertEqual(2,completed[0]['jewel_id'])
        self.assertEqual('thread-token',completed[0]['token'])
        phases={p['phase'] for p in events}
        self.assertTrue({'PH_FRESH','PH_REACTION','PH_VERIFIED','WAIT_FOR_STONE_READY','BASELINE','WAIT_FOR_ACID','OBSERVING','PROCESSING'}<=phases)
        ordered=[p['phase'] for p in events]
        self.assertLess(ordered.index('PH_VERIFIED'),ordered.index('WAIT_FOR_STONE_READY'))
        self.assertLess(ordered.index('WAIT_FOR_STONE_READY'),ordered.index('BASELINE'))
        self.assertEqual('ACIDIC_REACTION_VERIFIED',completed[0]['result']['ph_verification']['result'])

    def test_ph_pass_does_not_start_baseline_without_operator(self):
        events,completed,failed=self.run_thread(confirm_ready=False)
        self.assertFalse(completed)
        self.assertFalse(failed)
        phases=[p['phase'] for p in events]
        self.assertIn('WAIT_FOR_STONE_READY',phases)
        self.assertNotIn('BASELINE',phases)
        self.assertNotIn('LOADING',phases)

    def test_ph_off_completes_without_loading_or_verifying_paper(self):
        events,completed,failed=self.run_thread(ph_enabled=False)
        self.assertFalse(failed)
        self.assertEqual(1,len(completed))
        phases=[p['phase'] for p in events]
        self.assertIn('PH_SKIPPED',phases)
        self.assertNotIn('PH_LOADING',phases)
        self.assertNotIn('PH_VERIFIED',phases)
        self.assertLess(phases.index('WAIT_FOR_STONE_READY'),phases.index('BASELINE'))
        result=completed[0]['result']
        self.assertFalse(result['ph_verification_enabled'])
        self.assertEqual('SKIPPED_BY_OPERATOR',result['ph_verification']['result'])
        self.assertEqual({},result['ph_verification']['images'])

    def test_ph_off_still_requires_stone_ready_confirmation(self):
        events,completed,failed=self.run_thread(ph_enabled=False,cancel_waiting=True)
        self.assertFalse(completed)
        self.assertFalse(failed)
        self.assertIn('WAIT_FOR_STONE_READY',[p['phase'] for p in events])
        self.assertNotIn('BASELINE',[p['phase'] for p in events])

    def test_ph_off_keeps_stone_guarding(self):
        events,completed,failed=self.run_thread(ph_enabled=False,stone_getter=lambda:None,cancel_when_hidden=True)
        self.assertFalse(completed)
        self.assertFalse(failed)
        self.assertTrue(any('not visible' in p['message'] for p in events))
        self.assertNotIn('BASELINE',[p['phase'] for p in events])

    def test_hidden_stone_does_not_restart_ph_verification(self):
        events,completed,failed=self.run_thread(stone_getter=lambda:None,cancel_when_hidden=True)
        self.assertFalse(completed)
        self.assertFalse(failed)
        phases=[p['phase'] for p in events]
        self.assertEqual(1,phases.count('PH_LOADING'))
        self.assertEqual(1,phases.count('PH_VERIFIED'))
        self.assertTrue(any('retained' in p['message'] for p in events))

    def test_cancel_while_waiting_never_starts_baseline(self):
        events,completed,failed=self.run_thread(cancel_waiting=True)
        self.assertFalse(completed)
        self.assertFalse(failed)
        self.assertIn('WAIT_FOR_STONE_READY',[p['phase'] for p in events])
        self.assertNotIn('BASELINE',[p['phase'] for p in events])

    def test_ready_confirmation_cannot_be_given_before_ph_or_reused(self):
        first=AcidTestThread(dict(token='one',jewel_id=2))
        self.assertFalse(first.confirm_stone_ready())
        first.cancel()
        self.assertFalse(first.confirm_stone_ready())
        retry=AcidTestThread(dict(token='retry',jewel_id=2))
        self.assertFalse(retry.confirm_stone_ready())

    def test_cancelled_thread_never_emits_a_result(self):
        _,completed,failed=self.run_thread(cancel_at_baseline=True)
        self.assertFalse(completed)
        self.assertFalse(failed)

    def test_precolored_paper_never_starts_stone_capture(self):
        events,completed,failed=self.run_thread(precolored=True)
        self.assertFalse(completed)
        self.assertFalse(failed)
        self.assertNotIn('BASELINE',[p['phase'] for p in events])

    def test_cancel_during_ph_never_starts_stone_capture(self):
        events,completed,failed=self.run_thread(cancel_during_ph=True)
        self.assertFalse(completed)
        self.assertFalse(failed)
        self.assertNotIn('BASELINE',[p['phase'] for p in events])

    def test_stale_camera_cannot_verify_ph(self):
        context=dict(jewel_id=2,token='stale',grabber=SimpleNamespace(
            latest=lambda:(FRAME,time.time()-10)),stone_getter=lambda:STONE,stone=STONE)
        cache=Mock()
        thread=AcidTestThread(context,models=cache)
        with patch('acid_test.integration.time.monotonic',side_effect=[0,4]):
            with self.assertRaisesRegex(RuntimeError,'Camera stream lost'):
                thread.verify_ph_paper()
        cache.get.assert_not_called()
        self.assertIsNone(thread.ph_verification)

    def test_ph_timeout_cannot_start_temporal_models(self):
        context=dict(jewel_id=2,token='timeout',grabber=SimpleNamespace(
            latest=lambda:(FRAME,time.time())),stone_getter=lambda:STONE,stone=STONE)
        cache=Mock()
        thread=AcidTestThread(context,models=cache)
        with patch('acid_test.integration.time.monotonic',side_effect=[0,121]):
            with self.assertRaisesRegex(RuntimeError,'120 seconds'):
                thread.verify_ph_paper()
        cache.get.assert_not_called()
        self.assertIsNone(thread.ph_verification)

    def test_stone_loss_fails_without_classifying(self):
        _,completed,failed=self.run_thread(lose_at_baseline=True)
        self.assertFalse(completed)
        self.assertEqual(1,len(failed))
        self.assertIn('Stone moved',failed[0]['message'])


class ModelCacheTests(unittest.TestCase):
    def test_initialization_is_lazy_and_warmup_happens_only_once(self):
        with patch('acid_test.integration.OnnxAcidDetector') as detector, \
             patch('acid_test.integration.TemporalMobileNet') as classifier:
            cache=AcidModelCache()
            detector.assert_not_called()
            classifier.assert_not_called()
            first=cache.get()
            second=cache.get()
            self.assertIs(first,second)
            detector.assert_called_once()
            classifier.assert_called_once()
            detector.return_value.predict.assert_called_once()
            classifier.return_value.predict.assert_called_once()
            one=AcidTestThread(dict(token='one',jewel_id=2),models=cache)
            two=AcidTestThread(dict(token='two',jewel_id=1),models=cache)
            self.assertIs(one.models,two.models)
            self.assertIs(one.models.get(),first)
            detector.return_value.predict.assert_called_once()  # no repeated warmup

    def test_ph_model_cached_but_each_attempt_starts_unverified(self):
        with patch('acid_test.integration.OpenVINOOBB') as factory:
            cache=AcidModelCache()
            self.assertIsNone(cache._ph_model)
            self.assertIs(cache.get_ph(),cache.get_ph())
            factory.assert_called_once()
            factory.return_value.detect.assert_called_once()
            one=AcidTestThread(dict(token='one',jewel_id=2),models=cache)
            one.ph_verification=record()['ph_verification']
            two=AcidTestThread(dict(token='two',jewel_id=1),models=cache)
            self.assertIsNone(two.ph_verification)
            self.assertIs(one.models,two.models)

    def test_missing_openvino_has_actionable_error_and_can_retry(self):
        missing=ModuleNotFoundError("No module named 'openvino'",name='openvino')
        with patch('acid_test.integration.OpenVINOOBB',side_effect=[missing,Mock()]):
            cache=AcidModelCache()
            with self.assertRaisesRegex(RuntimeError,'OpenVINO is missing'):
                cache.get_ph()
            self.assertIsNone(cache._ph_model)
            self.assertIsNotNone(cache.get_ph())

    def test_failed_initialization_can_be_retried(self):
        with patch('acid_test.integration.OnnxAcidDetector') as detector, \
             patch('acid_test.integration.TemporalMobileNet',side_effect=[RuntimeError('load failed'),Mock()]):
            cache=AcidModelCache()
            with self.assertRaisesRegex(RuntimeError,'load failed'):cache.get()
            self.assertIsNone(cache._models)
            cache.get()
            self.assertIsNotNone(cache._models)
            detector.return_value.predict.assert_called_once()


if __name__=='__main__':unittest.main()
