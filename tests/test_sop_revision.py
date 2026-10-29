import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from contextlib import contextmanager
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np

from continuous_testing import ContinuousMonitor, RubReference, reaction_covered
from rubbing_recovery import recover_contact
from sop_evidence import contact_region, validate_mark
from audit_report import audit_outcome
from session_evidence import FrameJournal, write_report
from test_continuous_testing import scene,track,Photos,STONE,QUAD


@contextmanager
def monitor_case(**kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        events=[]
        monitor=ContinuousMonitor(tmp,lambda:None,Photos(),Mock(),lambda kind,**data:events.append((kind,data)),**kwargs)
        try:yield monitor,events
        finally:
            monitor.pool.shutdown(wait=True,cancel_futures=True)
            monitor.detector_pool.shutdown(wait=True,cancel_futures=True)
            monitor.drain_finalization()
            monitor.finish_pool.shutdown(wait=True)
            monitor.report_pool.shutdown(wait=True)
            monitor.journal.close()


def snap(ts,held=None,revision=0,frame=None,contact=None):
    return dict(timestamp=ts,frame=scene() if frame is None else frame,mode='LOCKED',revision=revision,
        stone=STONE,placement=np.float32([[50,30],[250,30],[250,230],[50,230]]),
        bench_moving=held is not None,rubbing=False,contact=contact,
        tracks=[track(jid,seen=ts,in_hand=held==jid) for jid in (1,2)])


def contact(ts,frame=None,confirmed=False):
    return dict(timestamp=ts,frame=scene() if frame is None else frame,box=(125,105,145,140),
                mask=np.ones((35,20),np.uint8),origin=(125,105),stone=STONE,
                c=np.array([135.,122.]),confirmed=confirmed,area=700)


class MaskEvidenceTests(unittest.TestCase):
    def candidate(self):
        return RubReference(Photos(),scene(),1,QUAD).after(Photos(),scene(mark=True),2,QUAD)

    def test_same_frame_cannot_confirm_persistence_and_no_contact_is_unverified(self):
        c=self.candidate()
        region=contact_region(contact(2),QUAD)
        for _ in range(10):
            state,_=validate_mark(c,scene(mark=True),QUAD,2,region)
            self.assertEqual('CHECKING',state)
        self.assertEqual('CHECKING',validate_mark(c,scene(mark=True),QUAD,2.08,region)[0])
        self.assertEqual('VERIFIED',validate_mark(c,scene(mark=True),QUAD,2.16,region)[0])
        c=self.candidate()
        for ts in (2,2.08,2.16):state,_=validate_mark(c,scene(mark=True),QUAD,ts,None)
        self.assertEqual('UNVERIFIED_CONTACT_LOCATION',state)

    def test_wrong_contact_location_and_fixed_roi_reject_candidates(self):
        c=self.candidate();different=contact(2);different['origin']=(180,150);different['box']=(180,150,200,185)
        region=contact_region(different,QUAD)
        for ts in (2,2.08,2.16):state,_=validate_mark(c,scene(mark=True),QUAD,ts,region)
        self.assertEqual('REJECTED_WRONG_LOCATION',state)
        roi=np.float32([[80,60],[110,60],[110,199],[80,199]])
        self.assertEqual('REJECTED_OUTSIDE_STONE',validate_mark(self.candidate(),scene(mark=True),QUAD,2,None,roi)[0])

    def test_hand_disappearance_is_not_a_persistent_stone_mark(self):
        c=self.candidate()
        self.assertEqual('REJECTED_TRANSIENT',validate_mark(c,scene(),QUAD,2.2,contact_region(contact(2),QUAD))[0])


class CycleLifecycleTests(unittest.TestCase):
    def test_first_acid_application_is_latched_even_if_a_second_result_arrives(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1))
            a=m.active
            first=dict(bbox=(128,107,143,139),confidence=.9)
            second=dict(bbox=(178,107,193,139),confidence=.95)
            m._set_acid(a,100.1,first)
            m._set_acid(a,100.2,second)
            self.assertEqual(100.1,a['acid_onset'])
            self.assertEqual(first['bbox'],a['detector_bbox'])
            self.assertEqual(1,sum(k=='acid_detection' and e.get('detected') for k,e in events))

    def test_hand_and_cloth_over_reaction_are_not_usable_reaction_frames(self):
        box=(128,107,143,139)
        self.assertFalse(reaction_covered(scene(acid=True),box))
        hand=scene()
        colour=cv2.cvtColor(np.uint8([[[10,80,180]]]),cv2.COLOR_HSV2BGR)[0,0]
        hand[80:165,105:170]=colour
        self.assertTrue(reaction_covered(hand,box))
        cloth=scene();cloth[80:165,105:170]=220
        self.assertTrue(reaction_covered(cloth,box))

    def test_sustained_hand_coverage_keeps_the_last_clear_reaction_frame(self):
        with monitor_case() as (m,events):
            m.classifier=Mock()
            for i in range(21):
                ts=100+i*.1;image=scene()
                m.journal.offer(image,ts);m.step(snap(ts,held=1 if i==0 else None,frame=image))
            a=m.active
            m._set_acid(a,102,dict(bbox=(128,107,143,139),confidence=.9,source_frame=scene(acid=True)))
            colour=cv2.cvtColor(np.uint8([[[10,80,180]]]),cv2.COLOR_HSV2BGR)[0,0]
            for i in range(21,73):
                ts=100+i*.1;image=scene(acid=True)
                if ts>=104:image[80:165,105:170]=colour
                m.journal.offer(image,ts);m.step(snap(ts,frame=image))
            m.drain_finalization()
            r=m.tests[0]
            self.assertEqual('PARTIAL',r['reaction_coverage_status'])
            self.assertLess(r['reaction_last_timestamp'],104)
            self.assertIsNone(r['prediction'])
            m.classifier.predict.assert_not_called()

    def test_reaction_with_a_camera_gap_does_not_claim_complete_evidence_or_run_classifier(self):
        with monitor_case() as (m,events):
            m.classifier=Mock()
            m.step(snap(100,held=1))
            a=m.active;a['returned']=100.1;a['return_frame']=scene()
            a['poses']=[(100+i*.1,QUAD.copy()) for i in range(65)]
            for i in list(range(15))+list(range(55,65)):
                m.journal.offer(scene(acid=i>=10),100+i*.1)
            m.journal.queue.join()
            m._set_acid(a,101,dict(bbox=(128,107,143,139),confidence=.9,source_frame=scene(acid=True)))
            m._finish(106.1);m.drain_finalization()
            r=m.tests[0]
            self.assertEqual('PARTIAL',r['reaction_coverage_status'])
            self.assertLess(r['usable_reaction_seconds'],5)
            self.assertIsNone(r['prediction'])
            m.classifier.predict.assert_not_called()

    def test_acid_image_preserves_its_original_source_instead_of_nearest_archive_sample(self):
        with monitor_case() as (m,events):
            m.classifier=Mock()
            m.classifier.predict.return_value=dict(prediction='non_gold',confidence=.95)
            m.step(snap(100,held=1))
            a=m.active
            a['returned']=100.1;a['return_frame']=scene()
            a['poses']=[(100+i*.1,QUAD.copy()) for i in range(65)]
            for i in range(65):m.journal.offer(scene(acid=i>=10),100+i*.1)
            m.journal.queue.join()
            source=scene(acid=True);source[80:90,100:110]=(11,99,221)
            captured={};original=m._picture
            def picture(directory,name,frame):
                if name=='acid_application_source':captured['source']=frame.copy()
                return original(directory,name,frame)
            m._picture=picture
            m._set_acid(a,101.025,dict(bbox=(128,107,143,139),confidence=.9,source_frame=source))
            m._finish(106.1)
            m.drain_finalization()
            self.assertEqual(101.025,m.tests[0]['acid_placement_timestamp'])
            np.testing.assert_array_equal(source,captured['source'])
            self.assertEqual('COMPARISON_UNAVAILABLE',m.tests[0]['acid_target_status'])
            self.assertEqual('non_gold',m.tests[0]['prediction']['prediction'])

    def test_live_contact_uses_its_original_frame_and_timestamp(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1))
            source=scene();source[90:130,100:130]=(10,50,200)
            ev=contact(100.05,source,confirmed=True)
            # Snapshot belongs to a later, different camera frame.
            m.step(snap(100.2,held=1,frame=np.zeros_like(source),contact=ev))
            e=[e for k,e in events if k=='contact'][0]
            self.assertEqual(100.05,e['timestamp'])
            self.assertTrue(e['confirmed'])
            self.assertTrue(np.any(e['photo'][:,:,2]>150))
            self.assertEqual('CONFIRMED_LIVE',m.active['rubbing_status'])
            m.step(snap(100.3,held=1,contact=contact(99,source,True)))
            self.assertEqual(1,len([1 for k,_ in events if k=='contact']))

    def test_later_larger_unconfirmed_contact_cannot_replace_confirmed_rubbing_source(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1))
            m.step(snap(100.1,held=1,contact=contact(100.1,confirmed=True)))
            later=contact(100.2);later['area']=100000
            later['origin']=(180,150);later['box']=(180,150,200,185)
            m.step(snap(100.2,held=1,contact=later))
            self.assertEqual(100.1,m.active['contact']['timestamp'])
            self.assertTrue(m.active['contact']['confirmed'])
            expected=contact_region(contact(100.1),QUAD)
            self.assertGreater(np.count_nonzero(m.active['contact_region'] & expected),0)
            self.assertEqual(1,len([1 for k,_ in events if k=='contact']))

    def test_next_pickup_is_debounced_and_slow_finalization_does_not_block_j2(self):
        with monitor_case(inactivity_seconds=60,pickup_debounce=.3) as (m,events):
            entered=threading.Event();release=threading.Event()
            def slow(a):entered.set();release.wait(3)
            m._recover_and_save=slow
            m.step(snap(100,held=1))
            old=m.active['id']
            m.step(snap(100.2))
            m.step(snap(101,held=2))
            m.step(snap(101.1)) # One noisy pickup cannot end J1.
            self.assertEqual(old,m.active['id'])
            m.step(snap(101.2,held=2))
            m.step(snap(101.4,held=2))
            start=time.monotonic()
            m.step(snap(101.6,held=2))
            self.assertLess(time.monotonic()-start,.2)
            self.assertEqual(2,m.active['jewel_id'])
            self.assertNotEqual(old,m.active['id'])
            self.assertEqual(101.2,m.active['started'])
            self.assertTrue(entered.wait(1))
            release.set()

    def test_application_after_pending_next_pickup_cannot_latch_to_j1(self):
        with monitor_case(inactivity_seconds=60) as (m,events):
            m._recover_and_save=Mock()
            m.step(snap(100,held=1))
            old=m.active
            m.step(snap(100.2,held=2))
            m._candidate_acid(old,100.25,dict(bbox=(128,107,143,139),confidence=.9))
            self.assertIsNone(old['acid_onset'])
            m.step(snap(100.4,held=2))
            m.step(snap(100.6,held=2))
            self.assertEqual(2,m.active['jewel_id'])
            self.assertIsNone(old['acid_onset'])
            self.assertEqual('NEXT_JEWEL_PICKUP',old['close_reason'])

    def test_transient_pickup_releases_a_valid_held_acid_observation(self):
        with monitor_case(inactivity_seconds=60) as (m,events):
            m.step(snap(100,held=1))
            m.step(snap(100.2,held=2))
            m._candidate_acid(m.active,100.25,dict(bbox=(128,107,143,139),confidence=.9))
            self.assertIsNone(m.active['acid_onset'])
            m.step(snap(100.3))
            self.assertEqual(100.25,m.active['acid_onset'])
            self.assertEqual(1,m.active['jewel_id'])

    def test_last_jewel_closes_after_configured_inactivity_without_model_calls(self):
        with monitor_case(inactivity_seconds=1) as (m,events):
            for i in range(15):m.step(snap(100+i*.1,held=1 if i==0 else None))
            self.assertIsNone(m.active)
            m.drain_finalization()
            self.assertEqual('INACTIVITY_TIMEOUT',m.tests[0]['close_reason'])
            self.assertIsNone(m.tests[0]['prediction'])
            self.assertFalse(m.tests[0]['capture_complete'])

    def test_broad_hand_movement_is_activity_but_does_not_verify_rubbing(self):
        with monitor_case(inactivity_seconds=1) as (m,events):
            for i in range(20):
                image=scene()
                if i%2:image[65:160,85:205]=180
                m.step(snap(100+i*.15,held=1 if i==0 else None,frame=image))
            self.assertIsNotNone(m.active)
            self.assertFalse(m.active['live_rub'])

    def test_handoff_debounce_does_not_bridge_a_stream_gap(self):
        with monitor_case(inactivity_seconds=60) as (m,events):
            m.step(snap(100,held=1))
            old=m.active['id']
            m.step(snap(100.1,held=2))
            m.step(snap(110,held=2))
            self.assertEqual(old,m.active['id'])

    def test_stream_gap_is_not_observed_inactivity(self):
        with monitor_case(inactivity_seconds=1) as (m,events):
            m.step(snap(100,held=1))
            m.step(snap(110))
            self.assertIsNotNone(m.active)
            for i in range(1,5):m.step(snap(110+i*.1))
            self.assertIsNotNone(m.active)

    def test_late_cycle_events_keep_their_original_reset_generation(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1))
            old=m.active
            m.revision=1
            m._phase('REACTION','Old cycle event',old)
            payload=[e for k,e in events if k=='phase'][-1]
            self.assertEqual(0,payload['revision'])
            self.assertEqual(old['id'],payload['token'])

    def test_roi_revision_finalizes_old_cycle_instead_of_reusing_contact(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1,contact=contact(100,confirmed=True)))
            old=m.active['id']
            m.step(snap(100.2,held=2,revision=1))
            self.assertEqual(2,m.active['jewel_id'])
            self.assertNotEqual(old,m.active['id'])
            self.assertIsNone(m.active['contact_region'])

    def test_finalization_queue_spools_overload_and_drains_all_records(self):
        with monitor_case(inactivity_seconds=60) as (m,events):
            release=threading.Event();done=[]
            def finish(a):release.wait(2);done.append(a['id'])
            m._recover_and_save=finish
            for i in range(5):
                m._new_cycle(None,snap(100+i),100+i)
                m._finish(100+i+.1)
            self.assertLessEqual(m.final_pending,2)
            self.assertGreater(len(list(m.deferred_dir.glob('*.pkl'))),0)
            release.set();m.drain_finalization()
            self.assertEqual(5,len(done))
            self.assertEqual([],list(m.deferred_dir.glob('*.pkl')))


class RecoveryAndReportTests(unittest.TestCase):
    def test_new_mark_report_photo_has_no_jewellery_contact_outline(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1))
            a=m.active
            a['returned']=100.1
            a['reference']=RubReference(Photos(),scene(),99,QUAD)
            a['mask']=a['reference'].after(Photos(),scene(mark=True),100.2,QUAD)
            a['mask_evidence_status']='VERIFIED'
            a['contact_region']=np.ones_like(a['mask']['new_mask'],np.uint8)*255
            expected=m._stone_picture(a['mask']['photo'],QUAD)
            pictures={}
            save=m._picture
            def capture(directory,name,frame):
                pictures[name]=frame.copy()
                return save(directory,name,frame)
            with patch.object(m,'_picture',side_effect=capture), \
                 patch.object(m,'_contact_overlay',side_effect=AssertionError('Contact outline leaked into report')):
                m._finish(101)
                m.drain_finalization()
            self.assertIn('new_rub',pictures)
            np.testing.assert_array_equal(expected,pictures['new_rub'])
            self.assertTrue(Path(m.tests[0]['mask_image']).is_file())

    def test_bbox_then_motion_fallback_are_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=FrameJournal(tmp)
            observations=[]
            for i in range(10):
                ts=100+i*.15;frame=scene()
                frame[100:120,100+i*4:120+i*4]=225
                journal.offer(frame,ts);observations.append((ts,STONE))
            journal.close()
            cycle=dict(started=100,returned=101.4,finished=101.4,acid_onset=None,
                       directory=Path(tmp),stone_observations=observations)
            def save(d,n,img):p=d/(n+'.jpg');cv2.imwrite(str(p),img);return str(p)
            def bbox(frame,stone):
                return dict(box=(100,100,140,120),origin=(100,100),mask=np.ones((20,40),np.uint8),
                            c=np.array([120.,110.]),kind='BOUNDING_BOX')
            r=recover_contact(journal,cycle,bbox,save)
            self.assertEqual('UNVERIFIED_BOUNDING_BOX',r['rubbing_contact_status'])
            self.assertFalse(r['rubbing_verified'])
            self.assertNotIn('rubbing_contact_mask_image',r)
            r=recover_contact(journal,cycle,lambda *a:None,save)
            self.assertEqual('UNVERIFIED_LOCALIZED_MOTION',r['rubbing_contact_status'])
            self.assertFalse(r['rubbing_verified'])
            self.assertTrue(100<=r['rub_timestamp']<=101.4)
            def unavailable(*args):raise RuntimeError('Model temporarily unavailable')
            r=recover_contact(journal,cycle,unavailable,save)
            self.assertEqual('UNVERIFIED_LOCALIZED_MOTION',r['rubbing_contact_status'])
            self.assertTrue(r['rubbing_recovery_errors'])

    def test_live_contact_persists_the_exact_source_when_journal_sampling_missed_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=FrameJournal(tmp)
            frame=scene();frame[90:100,100:110]=200
            journal.offer(scene(),100);journal.close()
            live=contact(100.02,frame,confirmed=True)
            cycle=dict(started=100,returned=100.2,finished=106,acid_onset=101,
                       directory=Path(tmp),contact=live,contact_region=contact_region(live,QUAD))
            def save(d,n,img):p=d/(n+'.png');cv2.imwrite(str(p),img);return str(p)
            result=recover_contact(journal,cycle,None,save)
            self.assertEqual(100.02,result['rub_source_timestamp'])
            np.testing.assert_array_equal(frame,cv2.imread(result['rub_source_frame']))
            self.assertTrue(result['rubbing_verified'])

    def test_premature_return_keeps_later_pre_acid_live_contact_and_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=FrameJournal(tmp)
            journal.offer(scene(),100);journal.close()
            live=contact(103.9,scene(),confirmed=True)
            cycle=dict(started=100,returned=100.4,finished=112,acid_onset=107,
                       directory=Path(tmp),contact=live,contact_region=contact_region(live,QUAD))
            def save(d,n,img):p=d/(n+'.png');cv2.imwrite(str(p),img);return str(p)
            result=recover_contact(journal,cycle,None,save)
            self.assertEqual('CONTACT_AND_MOTION_LIVE',result['rubbing_contact_status'])
            self.assertTrue(result['rubbing_verified'])
            self.assertTrue(result['rubbing_return_precedes_contact'])
            self.assertEqual(103.9,result['rub_timestamp'])
            self.assertEqual([100,103.9],result['rubbing_evidence_interval'])
            np.testing.assert_array_equal(live['frame'],cv2.imread(result['rub_source_frame']))
            photo=cv2.imread(result['rub_image'])
            self.assertGreater(np.count_nonzero(np.all(photo==(0,215,255),axis=2)),0)
            self.assertTrue(Path(result['rubbing_contact_mask_image']).is_file())

    def test_out_of_cycle_live_contact_cannot_verify_a_fallback_photo(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=FrameJournal(tmp)
            journal.offer(scene(),100);journal.close()
            def save(d,n,img):p=d/(n+'.png');cv2.imwrite(str(p),img);return str(p)
            for ts,frame in ((99,scene()),(105,scene()),(106,scene()),(112,scene()),(100.1,None)):
                with self.subTest(timestamp=ts,missing_frame=frame is None):
                    live=contact(ts,confirmed=True);live['frame']=frame
                    cycle=dict(started=100,returned=100.4,finished=110,acid_onset=105,
                               directory=Path(tmp),contact=live,stone_observations=[(100,STONE)])
                    result=recover_contact(journal,cycle,None,save)
                    self.assertEqual('CONTACT_NOT_VISIBLE',result['rubbing_contact_status'])
                    self.assertFalse(result['rubbing_verified'])
                    self.assertNotIn('rubbing_contact_box',result)
                    self.assertEqual('contact_not_visible.png',Path(result['rub_image']).name)

    def test_audit_outcome_never_promotes_weak_contact_or_missing_mask(self):
        r=dict(capture_complete=True,prediction=dict(prediction='non_gold',confidence=.9),
               rubbing_verified=False,mask_evidence_status='VERIFIED',acid_target_status='ACID_OVERLAPS_NEW_RUB')
        self.assertEqual('REVIEW REQUIRED',audit_outcome(r))
        r['rubbing_verified']=True
        self.assertEqual('EVIDENCE VERIFIED',audit_outcome(r))
        r['mask_evidence_status']='UNVERIFIED_CONTACT_LOCATION'
        self.assertEqual('REVIEW REQUIRED',audit_outcome(r))
        r['capture_complete']=False
        self.assertEqual('INCOMPLETE',audit_outcome(r))
        with tempfile.TemporaryDirectory() as tmp:
            r.update(jewel_id=1,issues=['Hand covered the stone'])
            path=write_report(tmp,[dict(id=1)],[r])
            self.assertTrue(Path(path).read_bytes().startswith(b'%PDF'))
