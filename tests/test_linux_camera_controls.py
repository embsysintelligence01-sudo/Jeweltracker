import unittest
from unittest.mock import Mock, patch

import linux_camera_controls as controls
import jewel_tracker_5 as tracker


class NativeCameraTests(unittest.TestCase):
    def setUp(self):
        self.values = dict(power_line_frequency=2, auto_exposure=3,
                           exposure_time_absolute=156, exposure_dynamic_framerate=1,
                           brightness=128, gain=0)
        self.original = dict(self.values)
        self.calls = []
        self.cap = Mock()
        self.state = {}

    def ctl(self, device, option):
        self.calls.append((device, option))
        if option.startswith('--set-ctrl='):
            key, value = option[len('--set-ctrl='):].split('=')
            self.values[key] = int(value)
            return ''
        return '\n'.join(f'{k}: {v} (menu)' for k, v in self.values.items())

    def test_native_units_readback_and_off_restore_without_changing_focus_or_wb(self):
        with patch.object(controls, '_ctl', side_effect=self.ctl), patch.object(controls.time, 'sleep'):
            message = controls.apply_linux_antiflicker(self.cap, controls.LINUX_AF_DEFAULT, self.state, 2)
            self.assertIn('verified (V4L2): 50 Hz, 20.0 ms', message)
            self.assertEqual(self.values, dict(power_line_frequency=1, auto_exposure=1,
                             exposure_time_absolute=200, exposure_dynamic_framerate=0,
                             brightness=96, gain=0))
            self.assertTrue(all(device == '/dev/video2' for device, _ in self.calls))
            controls.apply_linux_antiflicker(self.cap, {'enabled': False}, self.state, 2)
        self.assertEqual(self.original, self.values)

    def test_rejected_control_is_reported_as_failure(self):
        def reject(device, option):
            if option == '--set-ctrl=power_line_frequency=1':
                return ''
            return self.ctl(device, option)
        with patch.object(controls, '_ctl', side_effect=reject), patch.object(controls.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'NOT verified: power_line_frequency'):
                controls.apply_linux_antiflicker(self.cap, controls.LINUX_AF_DEFAULT, self.state)

    def test_off_without_prior_application_does_not_write_controls(self):
        with patch.object(controls, '_ctl') as ctl:
            controls.apply_linux_antiflicker(self.cap, {'enabled': False}, self.state)
        ctl.assert_not_called()

    def test_linux_never_uses_directshow_exposure_codes(self):
        with patch.object(tracker.sys, 'platform', 'linux'), \
             patch.object(tracker, 'apply_linux_antiflicker', return_value='verified') as apply:
            self.assertEqual('verified', tracker.apply_usb_antiflicker(self.cap, {}, self.state, 2))
        apply.assert_called_once_with(self.cap, {}, self.state, 2)
        self.cap.set.assert_not_called()

    def test_reconnect_reapplies_profile_before_publishing_frames(self):
        grabber = tracker.RTSPGrabber(2, {'antiflicker': dict(controls.LINUX_AF_DEFAULT)})
        caps = [Mock(), Mock()]
        caps[0].read.return_value = (False, None)
        def second_read():
            grabber.stop()
            return False, None
        caps[1].read.side_effect = second_read
        def apply(cap, af, state, index):
            self.assertFalse(cap.read.called)
            self.assertEqual(2, index)
            return 'verified'
        with patch.object(tracker, 'open_usb_camera', side_effect=caps), \
             patch.object(tracker, 'apply_usb_antiflicker', side_effect=apply) as af, \
             patch.object(tracker.time, 'sleep'):
            grabber.run()
        self.assertEqual(2, af.call_count)
        for cap in caps:
            cap.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
