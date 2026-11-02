import tempfile,time,unittest
from pathlib import Path
from unittest.mock import Mock
import cv2
import numpy as np
import jewel_tracker_5 as app
from pickup_evidence import HomeAnchor
from test_sop_revision import monitor_case,snap,contact
from test_continuous_testing import scene,STONE,QUAD,Photos
from rubbing_recovery import recover_contact
from session_evidence import FrameJournal

HOME=np.array([280.,80.,320.,130.]);OTHER=np.array([345.,140.,375.,180.])
NEW=np.array([260.,210.,300.,260.])

def jewel(frame,box):
    x1,y1,x2,y2=map(int,box)
    cv2.ellipse(frame,((x1+x2)//2,(y1+y2)//2),(max(2,(x2-x1)//2-3),max(2,(y2-y1)//2-3)),0,0,360,(20,160,215),3)
    cv2.line(frame,(x1+8,y1+10),(x2-8,y2-10),(30,125,180),2)

def bed(home=True,hand=False,new=False,second=True):
    frame=scene()
    if home:jewel(frame,HOME)
    if second:jewel(frame,OTHER)
    if new:jewel(frame,NEW)
    if hand:frame[65:145,265:335]=(75,125,185)
    return frame

def detection(box):
    b=np.asarray(box,float)
    return app.Det(b,(b[:2]+b[2:])/2,float(np.prod(b[2:]-b[:2])),.95)

def pickup(ts,state='OCCLUDED_HOME',event=1,departed=None,returned=None,started=100.):
    return dict(enabled=True,state=state,event=event,started=started,departed_at=departed,
        returned_at=returned,observed_at=ts,home_presence='COVERED' if state=='OCCLUDED_HOME' else 'EMPTY',
        original_home_box=HOME.tolist(),home_box=HOME.tolist(),last_reliable_box=HOME.tolist(),last_reliable_at=99.8)

def live_snap(ts,first=None,second=None,frame=None):
    s=snap(ts,frame=frame);s.update(home_tracking=True,overlap_trigger_required=False)
    for tr,p in zip(s['tracks'],(first,second)):
        tr.update(pickup=p,in_bed=False,in_hand=False)
        if p:
            tr['state']='OCCLUDED_HOME' if p['state']=='OCCLUDED_HOME' else 'OCCLUDED'
            if p.get('returned_at') is not None:tr.update(state='OK',in_bed=True,box=NEW.copy(),pos=(NEW[:2]+NEW[2:])/2)
    return s

class HomeAnchorTests(unittest.TestCase):
    def test_hand_freezes_home_until_clear_and_rejects_partial_boxes(self):
        a=HomeAnchor(HOME,99.8,bed());self.assertTrue(a.enabled)
        a.probe(bed(hand=True),100)
        self.assertEqual('OCCLUDED_HOME',a.state);self.assertIsNone(a.departed_at)
        np.testing.assert_array_equal(HOME,a.home_box);np.testing.assert_array_equal(HOME,a.last_box)
        self.assertFalse(a.complete_box([280,80,288,110]))
        a.probe(bed(),100.2);self.assertEqual('CANCELLED',a.state)
        self.assertIsNone(a.departed_at)

    def test_empty_home_requires_successive_distinct_observations(self):
        a=HomeAnchor(HOME,99.8,bed());a.probe(bed(hand=True),100)
        a.probe(bed(home=False),100.2);a.probe(bed(home=False),100.2)
        self.assertIsNone(a.departed_at)
        a.probe(bed(home=False),100.4)
        self.assertEqual('DEPARTED',a.state);self.assertEqual(100,a.started)

    def test_new_bed_position_becomes_home_only_after_reliable_settle(self):
        a=HomeAnchor(HOME,99.8,bed());a.probe(bed(hand=True),100)
        a.probe(bed(home=False),100.2);a.probe(bed(home=False),100.4)
        frame=bed(home=False,new=True)
        a.reliable(NEW,100.5,frame,True,True);self.assertIsNone(a.returned_at)
        a.reliable(NEW,100.7,frame,True,True)
        self.assertEqual('RETURNED',a.state);np.testing.assert_array_equal(NEW,a.home_box)
        np.testing.assert_array_equal(HOME,a.original_box)

    def test_exposure_shift_or_missing_detection_with_same_image_does_not_trigger(self):
        a=HomeAnchor(HOME,99.8,bed())
        a.probe(np.clip(bed().astype(int)-8,0,255).astype(np.uint8),100)
        self.assertEqual('AT_HOME',a.state);self.assertIsNone(a.started)
        a.probe(bed(),100.2);self.assertIsNone(a.started)

    def test_low_saturation_jewel_uses_appearance_without_a_gold_colour_requirement(self):
        initial=cv2.cvtColor(cv2.cvtColor(bed(),cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        returned=cv2.cvtColor(cv2.cvtColor(bed(home=False,new=True),cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        a=HomeAnchor(HOME,99.8,initial)
        self.assertTrue(a.enabled);self.assertTrue(a.looks_like(returned,NEW))

    def test_uniform_gold_coloured_hand_is_not_a_reliable_return(self):
        a=HomeAnchor(HOME,99.8,bed());covered=bed(home=False)
        x1,y1,x2,y2=map(int,NEW);covered[y1:y2,x1:x2]=(20,160,215)
        self.assertFalse(a.looks_like(covered,NEW))

class WorkerOcclusionTests(unittest.TestCase):
    def worker(self):
        w=app.Worker(0,[(0,0),(399,0),(399,299),(0,299)],continuous=True)
        w.mode='LOCKED';w.stone_box=lambda:STONE
        w.tracks=[app.Track(1,detection(HOME),99.8),app.Track(2,detection(OTHER),99.8)]
        for tr in w.tracks:tr.home_anchor=HomeAnchor(tr.box,99.8,bed())
        for name in ('update_rubbing','update_acid','check_done'):setattr(w,name,Mock())
        return w

    def test_partial_under_hand_detection_does_not_move_j1_or_steal_j2(self):
        w=self.worker();w.update_tracks([detection([284,95,298,118]),detection(OTHER)],100,bed(hand=True))
        self.assertEqual('OCCLUDED_HOME',w.tracks[0].state)
        np.testing.assert_array_equal(HOME,w.tracks[0].det_box)
        self.assertEqual(99.8,w.tracks[0].det_seen)
        np.testing.assert_array_equal(OTHER,w.tracks[1].det_box)
        self.assertEqual([1,2],[t.id for t in w.tracks])

    def test_missing_segmentation_and_different_return_position_keep_identity(self):
        w=self.worker()
        w.update_tracks([detection(OTHER)],100,bed(hand=True))
        for ts in (100.2,100.4):w.update_tracks([detection(OTHER)],ts,bed(home=False))
        for ts in (100.6,100.8,101):w.update_tracks([detection(NEW),detection(OTHER)],ts,bed(home=False,new=True))
        a=w.tracks[0].home_anchor
        self.assertEqual('RETURNED',a.state);self.assertIsNotNone(a.returned_at)
        np.testing.assert_array_equal(NEW,w.tracks[0].det_box)
        self.assertIsNone(w.tracks[0].det_mask)
        np.testing.assert_array_equal(OTHER,w.tracks[1].det_box)

    def test_bbox_jitter_does_not_change_home_or_create_a_pickup_event(self):
        w=self.worker()
        for i,dx in enumerate((1,-1,2,-2,0)):
            w.update_tracks([detection(HOME+[dx,0,dx,0]),detection(OTHER)],100+i*.2,bed())
        self.assertIsNone(w.tracks[0].home_anchor.started)
        np.testing.assert_array_equal(HOME,w.tracks[0].home_anchor.home_box)
        np.testing.assert_array_equal(HOME,w.tracks[0].original_home_box)

    def test_original_or_slightly_shifted_home_return_is_not_mistaken_for_hand_coverage(self):
        for offset in (0,8):
            with self.subTest(offset=offset):
                w=self.worker();returned=HOME+[offset,0,offset,0]
                w.update_tracks([detection(OTHER)],100,bed(hand=True))
                for ts in (100.2,100.4):w.update_tracks([detection(OTHER)],ts,bed(home=False))
                frame=bed(home=False);jewel(frame,returned)
                for ts in (100.6,100.8,101):w.update_tracks([detection(returned),detection(OTHER)],ts,frame)
                self.assertEqual('RETURNED',w.tracks[0].home_anchor.state)
                np.testing.assert_array_equal(returned,w.tracks[0].home_anchor.home_box)

    def test_hand_clears_after_nudging_jewel_near_home_cancels_false_pickup(self):
        w=self.worker();w.update_tracks([detection(OTHER)],100,bed(hand=True))
        shifted=HOME+[8,0,8,0];frame=bed(home=False);jewel(frame,shifted)
        for ts in (100.2,100.4):w.update_tracks([detection(shifted),detection(OTHER)],ts,frame)
        self.assertEqual('CANCELLED',w.tracks[0].home_anchor.state)
        self.assertIsNone(w.tracks[0].home_anchor.departed_at)
        np.testing.assert_array_equal(HOME,w.tracks[0].home_anchor.original_box)

    def test_gold_pixels_with_skin_like_hue_are_not_themselves_a_hand(self):
        from types import SimpleNamespace
        frame=np.full((100,100,3),220,np.uint8)
        gold=cv2.cvtColor(np.uint8([[[10,180,180]]]),cv2.COLOR_HSV2BGR)[0,0]
        frame[30:60,30:60]=gold
        tr=SimpleNamespace(det_mask=np.ones((30,30),np.uint8),det_origin=(30,30))
        self.assertFalse(app.Worker.hand_contact(tr,frame))
        frame[30:60,60:70]=gold
        self.assertTrue(app.Worker.hand_contact(tr,frame))

    def test_two_covered_jewels_do_not_claim_a_new_box_without_identity(self):
        w=self.worker();covered=bed(hand=True);covered[130:190,335:385]=(75,125,185)
        w.update_tracks([],100,covered)
        frame=bed(home=False,new=True,second=False)
        for ts in (100.2,100.4,100.6):w.update_tracks([detection(NEW)],ts,frame)
        self.assertFalse(any(t.home_anchor.returned_at is not None for t in w.tracks))

class PickupWindowTests(unittest.TestCase):
    def test_home_occlusion_pins_preroll_without_test_or_acid_then_cancels(self):
        with monitor_case() as (m,events):
            m.step(live_snap(100,pickup(100)))
            self.assertIsNone(m.active);self.assertEqual(1,len(m.pickup_windows))
            w=next(iter(m.pickup_windows.values()))
            self.assertEqual(98.5,m.journal.pins[w['token']][0])
            m.step(live_snap(100.2,pickup(100.2,state='CANCELLED')))
            self.assertFalse(m.pickup_windows);self.assertNotIn(w['token'],m.journal.pins)
            self.assertFalse(any(k in ('begin','baseline','acid_detection') for k,_ in events))
            self.assertFalse(list(m.directory.glob('test_*')))

    def test_confirmed_departure_starts_cycle_without_stone_overlap(self):
        with monitor_case() as (m,events):
            m.step(live_snap(100,pickup(100)))
            m.step(live_snap(100.4,pickup(100.4,'DEPARTED',departed=100.3)))
            a=m.active;self.assertEqual(1,a['jewel_id']);self.assertTrue(a['pickup_confirmed'])
            self.assertEqual(100,a['started']);self.assertEqual(98.5,a['evidence_start'])
            self.assertIsNone(a['baseline_at']);self.assertIsNone(a['returned'])
            self.assertFalse(a['overlap_trigger_required'])

    def test_same_jewel_return_anywhere_checks_mask_then_baseline_without_contact(self):
        with monitor_case() as (m,events):
            m.step(live_snap(100,pickup(100)))
            m.step(live_snap(100.4,pickup(100.4,'DEPARTED',departed=100.3)))
            a=m.active
            for ts in (100.6,100.8,101,101.3,101.8,102):
                m.step(live_snap(ts,pickup(ts,'RETURNED',departed=100.3,returned=100.6)))
                if m.mask_job:m.mask_job[0].result(timeout=3)
            self.assertIsNotNone(a['returned']);self.assertIsNotNone(a['baseline_at'])
            self.assertIsNone(a['contact']);self.assertIsNone(a['acid_onset'])
            self.assertTrue(any(k=='mask_decision' for k,_ in events))

    def test_fast_return_preserves_early_window_and_reference_without_overlap(self):
        from continuous_testing import RubReference
        with monitor_case() as (m,events):
            m.step(live_snap(99.8))
            before=RubReference(Photos(),scene(),99,QUAD);m.reference=before
            m.step(live_snap(100,pickup(100)))
            # A result captured after rubbing must not replace the before image.
            m.reference=RubReference(Photos(),scene(mark=True),100.1,QUAD)
            m.step(live_snap(100.3,pickup(100.3,'RETURNED',departed=100.15,returned=100.28)))
            self.assertIs(before,m.active['reference'])
            self.assertEqual(100,m.active['started']);self.assertEqual(98.5,m.active['evidence_start'])
            self.assertIsNotNone(m.active['returned']);self.assertIsNone(m.active['acid_onset'])

    def test_mask_timeout_does_not_start_baseline_on_a_covered_stone(self):
        with monitor_case() as (m,events):
            m.step(live_snap(100,pickup(100)))
            m.step(live_snap(100.4,pickup(100.4,'DEPARTED',departed=100.3)))
            m.step(live_snap(100.6,pickup(100.6,'RETURNED',departed=100.3,returned=100.6)))
            a=m.active;self.assertIsNotNone(a['returned'])
            covered=scene();covered[60:200,80:220]=180
            for ts in (100.8,101.2,102):
                m.step(live_snap(ts,pickup(ts,'RETURNED',departed=100.3,returned=100.6),frame=covered))
            self.assertIsNone(a['baseline_at']);self.assertIsNone(a['acid_onset'])

    def test_wrong_owner_return_event_cannot_advance_the_current_cycle(self):
        with monitor_case() as (m,events):
            m.step(live_snap(100,pickup(100)))
            m.step(live_snap(100.4,pickup(100.4,'DEPARTED',departed=100.3)))
            a=m.active
            m.step(live_snap(100.8,pickup(100.8,'RETURNED',departed=100.3,returned=100.7,started=100.5)))
            self.assertIsNone(a['returned']);self.assertIsNone(a['baseline_at'])

    def test_clear_stone_or_missing_owner_alone_cannot_advance_to_baseline(self):
        with monitor_case() as (m,events):
            m.step(live_snap(100,pickup(100)))
            m.step(live_snap(100.4,pickup(100.4,'DEPARTED',departed=100.3)));a=m.active
            for ts in (101,102,103):
                s=live_snap(ts);s['tracks']=s['tracks'][1:];m.step(s)
            self.assertIsNone(a['returned']);self.assertIsNone(a['baseline_at'])

    def test_ambiguous_two_home_occlusions_do_not_guess_owner(self):
        with monitor_case() as (m,events):
            m.step(live_snap(100,pickup(100),pickup(100)))
            m.step(live_snap(100.4,pickup(100.4,'DEPARTED',departed=100.3),pickup(100.4)))
            self.assertIsNone(m.active);self.assertEqual(2,len(m.pickup_windows))
            m.step(live_snap(100.6,pickup(100.6,'DEPARTED',departed=100.3),pickup(100.6,'CANCELLED')))
            self.assertEqual(1,m.active['jewel_id'])

    def test_duplicate_or_stale_observations_cannot_confirm_return(self):
        with monitor_case() as (m,events):
            m.step(live_snap(100,pickup(100)))
            m.step(live_snap(100.4,pickup(100.4,'DEPARTED',departed=100.3)));a=m.active
            s=live_snap(101,pickup(100.4,'RETURNED',departed=100.3,returned=100.4))
            for tr in s['tracks']:tr['det_seen']=100.4
            m.step(s)
            self.assertIsNone(a['returned']);self.assertIsNone(a['baseline_at'])

    def test_next_jewel_occlusion_does_not_close_previous_acid_search(self):
        with monitor_case() as (m,events):
            m.step(live_snap(100,pickup(100)))
            m.step(live_snap(100.4,pickup(100.4,'DEPARTED',departed=100.3)));a=m.active
            for ts in (100.6,100.8,101):
                m.step(live_snap(ts,pickup(ts,'DEPARTED',departed=100.3),pickup(ts,started=100.6)))
            self.assertIs(a,m.active);self.assertFalse(m.tests)
            for ts in (101.2,101.4,101.6):
                m.step(live_snap(ts,pickup(ts,'DEPARTED',departed=100.3),pickup(ts,'DEPARTED',departed=101.1,started=100.6)))
            self.assertEqual(2,m.active['jewel_id']);self.assertTrue(m.acid_cancellations[a['id']].is_set())

    def test_stone_activity_preserves_unverified_cancelled_window_without_starting_acid(self):
        with monitor_case() as (m,events):
            for ts in (98.5,99,100,100.2,100.4):m.journal.offer(scene(),ts);m.journal.queue.join()
            m.step(live_snap(100,pickup(100)))
            w=next(iter(m.pickup_windows.values()));w['stone_activity']=True
            m.step(live_snap(100.4,pickup(100.4,'CANCELLED')))
            w['future'].result(timeout=3)
            self.assertIsNone(m.active);self.assertFalse(m.tests)
            self.assertNotIn(w['token'],m.journal.pins)
            self.assertTrue((m.directory/'provisional_events'/w['token']/'manifest.json').is_file())

    def test_recorded_recovery_runs_after_missed_overlap_and_keeps_motion_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            j=FrameJournal(tmp)
            for i in range(6):j.offer(scene(),100+i*.1);j.queue.join()
            j.close();observer=Mock(return_value=None)
            cycle=dict(started=100,returned=100.5,finished=101,acid_onset=None,
                overlap_trigger_required=True,rubbing_started_at=None,directory=Path(tmp),
                stone_observations=[(100+i*.1,STONE) for i in range(6)])
            result=recover_contact(j,cycle,observer,Mock())
            self.assertGreater(observer.call_count,0);self.assertFalse(result['rubbing_verified'])

class ProvisionalArchiveTests(unittest.TestCase):
    def test_unverified_window_is_archived_linked_and_never_becomes_a_test(self):
        import json,shutil
        from dataclasses import replace
        from unittest.mock import patch
        import test_evidence_retention as fixtures
        from evidence_archive import ArchiveManager,RetentionPolicy,validate_video
        from session_evidence import write_report,atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,record=fixtures.EvidenceRetentionTests().fixture(root,prediction=True)
            folder=root/'provisional_events'/'pickup_2_test';folder.mkdir(parents=True)
            path=folder/'pickup_unverified.avi';shutil.copyfile(record['acid_clip']['path'],path)
            clip=dict(record['acid_clip'],path=str(path))
            problem=dict(reason='Unverified pickup J2: jewel remained at home',start=108,end=115,count=1,
                status='UNVERIFIED_PICKUP',jewel_id=2,event_id='pickup_2_test',provisional_clip=clip,
                provisional_manifest=str(folder/'manifest.json'))
            atomic_json(folder/'manifest.json',problem)
            write_report(root,[dict(id=1,image=str(root/'J1.jpg'))],[record],[problem])
            with patch('evidence_archive.encoder_available',return_value=None):
                result=ArchiveManager(root,RetentionPolicy(require_server_ack=False)).finalize(True)
            self.assertEqual('cleaned',result['status']);self.assertFalse(path.exists())
            data=json.loads((root/'session.json').read_text());self.assertEqual(1,len(data['tests']))
            self.assertEqual(record['prediction'],data['tests'][0]['prediction'])
            kept=data['recording_problems'][0]
            self.assertEqual('UNVERIFIED_PICKUP',kept['status'])
            validate_video(kept['provisional_clip']['path'],kept['provisional_clip'])
            self.assertTrue(Path(kept['provisional_manifest']).is_file())
            self.assertEqual(kept,json.loads((folder/'manifest.json').read_text()))
            self.assertEqual(data['tests'][0]['acid_clip']['path'],kept['provisional_clip']['path'])
            self.assertTrue((root/'report.pdf').read_bytes().startswith(b'%PDF'))

if __name__=='__main__':unittest.main()
