import unittest
from test_sop_revision import monitor_case,snap

class RemovedDepartureTriggerTests(unittest.TestCase):
    def test_departure_event_without_pickup_cannot_open_a_cycle(self):
        with monitor_case() as (m,events):
            for ts in (100,100.2,100.4):
                s=snap(ts)
                s.update(overlap_trigger_required=True,tracker_evidence=dict(departures=[dict(
                    jewel_id=1,timestamp=100,last_seen=ts,initial_box=[280,90,320,110])],contacts=[]))
                m.step(s)
            self.assertIsNone(m.active)
            self.assertFalse(any(k=='begin' for k,_ in events))

    def test_old_provisional_flag_cannot_bypass_missing_stone_overlap(self):
        with monitor_case() as (m,events):
            s=snap(100,held=1);s['overlap_trigger_required']=True;m.step(s)
            a=m.active
            # Reproduce the flag that previously bypassed contact_armed.
            a['pickup_trigger_kind']='TRACK_BOX_DEPARTURE'
            for ts in (100.2,100.4,100.8,101.5,102):
                s=snap(ts);s['overlap_trigger_required']=True;m.step(s)
            self.assertIsNone(a['rubbing_started_at'])
            self.assertIsNone(a['returned'])
            self.assertIsNone(a['baseline_at'])
            self.assertIsNone(a['acid_onset'])
            self.assertFalse(any(k=='baseline' for k,_ in events))
            self.assertEqual('WAIT_FOR_CONTACT',m.phase[1])

    def test_real_two_frame_overlap_still_allows_return_and_baseline(self):
        with monitor_case() as (m,events):
            s=snap(100,held=1);s['overlap_trigger_required']=True;m.step(s);a=m.active
            s=snap(100.2,held=1)
            s.update(overlap_trigger_required=True,rubbing_trigger=dict(timestamp=100.1,
                confirmed_at=100.2,frames=2,cycle_token=a['id']))
            m.step(s)
            for ts in (100.4,100.6,101,101.8):
                s=snap(ts);s['overlap_trigger_required']=True;m.step(s)
            self.assertIsNotNone(a['returned'])
            self.assertIsNotNone(a['baseline_at'])

if __name__=='__main__':unittest.main()
