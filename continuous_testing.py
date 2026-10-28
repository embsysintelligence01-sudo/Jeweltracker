"""Non-blocking pickup/rub/acid workflow with recorded-frame recovery.

Tracking evidence owns identity. The acid runtime owns the material prediction.
Neither a missed jewel nor a failed rub mask overwrites that prediction.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import threading
import time
import uuid

import cv2
import numpy as np

from new_rub_detection.flow import RubCheck
from new_rub_detection.geometry import DEST, project_mask
from rub_preparation import clear_surface, draw_acid_roi
from session_evidence import FrameJournal, atomic_json, write_report
from acid_freshness import AcidHistory, FreshAcidGate, footprint, normalized
from rubbing_recovery import recover_contact

HAND = {'IN_HAND', 'UNDER_HAND'}
# Trial policy: covered jewels remain tracked, but cannot start a test cycle.
PICKUP_STATES = {'IN_HAND'}
PHASE_TITLES = {
    'BEFORE_CAPTURE': 'CAPTURING BEFORE STONE', 'BEFORE_READY': 'BEFORE STONE CAPTURED',
    'WAIT_FOR_STONE': 'WAITING FOR CLEAR STONE', 'RUBBING': 'RUBBING / JEWEL IN HAND',
    'WAIT_FOR_PLACEMENT': 'WAITING FOR STONE INSIDE TEST ROI',
    'MASK_CHECK': 'CHECKING NEW RUBBED MASK', 'BASELINE': 'BASELINE CAPTURE',
    'WAIT_FOR_ACID': 'WAITING FOR ACID', 'REACTION': '5 SECOND REACTION',
    'PROCESSING': 'PROCESSING ACID RESULT', 'DONE': 'DONE',
}


def workflow_lines(state, completed=()):
    prefix = f"J{state['jewel_id']} | " if state.get('jewel_id') is not None else ''
    title = PHASE_TITLES.get(state.get('phase'), state.get('phase', 'STARTING'))
    lines = [prefix+'PROCESS: '+title]
    mask = state.get('mask_state')
    acid = state.get('acid_state')
    evidence = []
    if mask:
        evidence.append('NEW RUB: '+mask)
    if acid:
        evidence.append('ACID: '+acid)
    if evidence:
        lines.append(' | '.join(evidence))
    if state.get('remaining') is not None:
        lines.append(f"REACTION LEFT: {state['remaining']:.1f}/5.0 s")
    if state.get('mask_uncertain'):
        lines.append('MASK VISIBILITY UNCERTAIN - last position retained')
    if completed:
        lines.append('DONE: '+', '.join(f'J{jid}' for jid in completed))
    return lines


def draw_workflow(vis, state, completed=(), acid=None):
    """Development overlay only; original camera evidence stays unannotated."""
    lines = workflow_lines(state, completed)
    height = 28*len(lines)+12
    width = min(vis.shape[1]-20, 1100)
    cv2.rectangle(vis, (10, 48), (max(20, width), min(vis.shape[0]-1, 48+height)), (24, 24, 24), -1)
    colour = (80, 230, 130) if state.get('phase') in ('REACTION', 'DONE', 'BEFORE_READY') else (30, 210, 255)
    for index, line in enumerate(lines):
        while cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .6, 1)[0][0] > width-25 and len(line) > 5:
            line = line[:-4]+'...'
        cv2.putText(vis, line, (18, 73+28*index), cv2.FONT_HERSHEY_SIMPLEX, .6, colour, 1, cv2.LINE_AA)
    if acid and acid.get('detected') and acid.get('bbox') and state.get('phase') == 'REACTION':
        box = acid['bbox']
        cv2.rectangle(vis, tuple(box[:2]), tuple(box[2:]), (255, 255, 0), 2)
        cv2.putText(vis, f"ACID DETECTED {acid.get('confidence', 0):.2f}",
                    (box[0], max(20, box[1]-8)), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 0), 2)
    return vis


def evidence_reason(message):
    message = str(message).replace('RETAKE - ', '').replace('Please rub on a clear area and retry.', '')
    replacements = {
        'remove hands and jewellery from the stone, then capture again': 'stone surface was covered',
        'stone corners unclear; expose the complete surface': 'stone corners were not fully visible',
        'expose one complete stone': 'complete stone surface was not detected',
    }
    for instruction, reason in replacements.items():
        message = message.replace(instruction, reason)
    return message.strip()


def visibility(track, timestamp, bench_moving):
    """Use the same evidence as the tracker's hand labels, never absence alone."""
    if track['state'] in ('MISSING', 'LEFT_ROI'):
        return 'MISSED'
    if track.get('in_hand'):
        return 'IN_HAND'
    if track['state'] == 'OK' and 0 <= timestamp-track['det_seen'] <= .75:
        return 'VISIBLE'
    if track.get('spot') == 'hand':
        return 'UNDER_HAND'
    if track.get('spot') == 'removed' and bench_moving:
        return 'IN_HAND'
    return 'NOT_SEEN'


def choose_hand_jewel(tracks, timestamp, bench_moving, recent_hand):
    candidates = [tr for tr in tracks if not tr.get('acid_done')
                  and (visibility(tr, timestamp, bench_moving) in PICKUP_STATES
                       or 0 <= timestamp-recent_hand.get(tr['id'], -1e9) <= 2)]
    return candidates[0]['id'] if len(candidates) == 1 else None


def box_of(quad):
    return tuple(np.r_[np.floor(quad.min(0)), np.ceil(quad.max(0))].astype(int))


def inside_roi(quad, roi):
    return roi is not None and all(cv2.pointPolygonTest(np.asarray(roi, np.float32),
                                   tuple(map(float, p)), False) >= 0 for p in quad)


def surface_covered(frame, stone):
    """Broad bright/cloth coverage, separately from an uncertain pose/tool tip."""
    if stone is None:
        return False
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = np.clip(stone, [0, 0, 0, 0], [w, h, w, h]).astype(int)
    crop = frame[y1:y2, x1:x2]
    if not crop.size:
        return True
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    skin = (hsv[:, :, 0] <= 14) & (hsv[:, :, 1] >= 110) & (hsv[:, :, 2] > 60)
    light = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) > 110
    return float(light.mean()) > .65 and float(skin.mean()) < .10


class StonePose:
    """Cheap corner continuity between sparse model observations.

    Missing/covered geometry preserves the previous pose with uncertain=True.
    Corner transforms are explicitly recorded as a limited-motion assumption;
    a symmetric stone's rotation cannot be proven after complete occlusion.
    """
    def __init__(self):
        self.quad = None
        self.uncertain = True
        self.last_clear = None
        self.stable_since = None

    def update(self, frame, timestamp, stone):
        if stone is None:
            self.uncertain, self.stable_since = True, None
            return False
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(float, stone)
        pad = .18*max(x2-x1, y2-y1)
        x1, y1 = max(0, int(x1-pad)), max(0, int(y1-pad))
        x2, y2 = min(w, int(x2+pad)), min(h, int(y2+pad))
        region = frame[y1:y2, x1:x2]
        if not region.size:
            self.uncertain, self.stable_since = True, None
            return False
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        dark = (gray < 110).astype(np.uint8)*255
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        expected = max(1., (stone[2]-stone[0])*(stone[3]-stone[1]))
        for contour in contours:
            area = cv2.contourArea(contour)
            if not .65*expected <= area <= 1.45*expected:
                continue
            corners = cv2.approxPolyDP(contour, .025*cv2.arcLength(contour, True), True)
            if len(corners) != 4 or not cv2.isContourConvex(corners):
                continue
            quad = corners.reshape(4, 2).astype(np.float32)+(x1, y1)
            center = quad.mean(0)
            quad = quad[np.argsort(np.arctan2(quad[:, 1]-center[1], quad[:, 0]-center[0]))]
            quad = np.roll(quad, -np.argmin(quad.sum(1)), axis=0)
            if self.quad is not None:
                quad = min((np.roll(quad, k, axis=0) for k in range(4)),
                           key=lambda q: float(np.linalg.norm(q-self.quad, axis=1).sum()))
                if np.max(np.linalg.norm(quad-self.quad, axis=1)) > .4*np.sqrt(expected):
                    continue
            candidates.append(quad.astype(np.float32))
        if not candidates:
            self.uncertain, self.stable_since = True, None
            return False
        quad = max(candidates, key=cv2.contourArea)
        mask = cv2.fillConvexPoly(np.zeros((h, w), np.uint8), quad.astype(np.int32), 255)
        try:
            clear_surface(frame, mask)
        except RuntimeError:
            self.uncertain, self.stable_since = True, None
            return False
        moved = self.quad is None or np.max(np.linalg.norm(quad-self.quad, axis=1)) > 6
        if moved or self.uncertain or self.stable_since is None:
            self.stable_since = timestamp
        self.quad, self.uncertain, self.last_clear = quad, False, timestamp
        return True


class RubReference:
    def __init__(self, models, frame, timestamp, tracked_quad, revision=None):
        if tracked_quad is None:
            stone, marks, quad = models.photo(frame)
        else:
            quad = np.asarray(tracked_quad, np.float32).copy()
            stone = cv2.fillConvexPoly(np.zeros(frame.shape[:2], np.uint8), quad.astype(np.int32), 255)
            marks = models.marks(frame)
        self.check = RubCheck()
        self.before = self.check.before(frame, stone, marks, quad)
        self.quad, self.timestamp = quad, timestamp
        self.frame = frame
        self.revision = revision

    def after(self, models, frame, timestamp, tracked_quad):
        if timestamp <= self.timestamp:
            return dict(status='UNCONFIRMED', message='No pre-rubbing reference precedes this frame.')
        if tracked_quad is None:
            return dict(status='UNCONFIRMED', message='Stone corner position was uncertain in this frame.')
        quad = np.asarray(tracked_quad, np.float32).copy()
        stone = cv2.fillConvexPoly(np.zeros(frame.shape[:2], np.uint8), quad.astype(np.int32), 255)
        clear_surface(frame, stone)
        marks = models.marks(frame)
        result = self.check.after(frame, stone, marks, quad,
                    frame_to_reference=cv2.getPerspectiveTransform(quad.astype(np.float32), DEST),
                    alignment_detail='Tracked corner continuity; orientation assumed under limited motion')
        result['quad'], result['timestamp'] = quad, timestamp
        result['frame'] = frame
        if result['status'] == 'NEW_RUB':
            result['roi_mask'] = project_mask(result['new_mask'], frame.shape, result['frame_to_reference'])
        return result


class ContinuousMonitor(threading.Thread):
    def __init__(self, directory, snapshot, models, acid_models, emit, rub_observer=None):
        super().__init__(name='continuous-tests', daemon=True)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal = FrameJournal(self.directory)
        self.snapshot, self.models, self.acid_models, self.emit = snapshot, models, acid_models, emit
        self.rub_observer = rub_observer
        self.acid_history = AcidHistory()
        self.acid_context = None
        self.stop_event = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='live-rub-photo')
        self.detector_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='live-acid-roi')
        self.finish_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='clip-recovery')
        self.report_lock = threading.Lock()
        self.tests, self.jewels = [], {}
        self.intervals, self.recent_hand = {}, {}
        self.pose, self.reference = StonePose(), None
        self.reference_job = self.mask_job = self.acid_job = None
        self.recovery_job = None
        self.active = None
        self.last_ts = self.last_photo = self.last_detection = self.last_reference = -1e9
        self.last_acid_ts, self.last_acid_present = -1e9, True
        self.placement = None
        self.placement_auto = False
        self.revision = None
        self.detector = self.classifier = None
        self.model_error = None
        self.last_notice = -1e9
        self.phase = None
        self.last_phase_emit = -1e9
        self.last_capture_end = -1e9
        self.captured_ids = set()

    def run(self):
        try:
            # Recording is already running while models are prepared.
            self.emit('status', message='Preparing automatic test models; recording continues.')
            try:
                self.models.marks(np.zeros((640, 640, 3), np.uint8))
                self.detector, self.classifier = self.acid_models.get()
            except Exception as exc:
                self.model_error = str(exc)
                self.emit('status', message=f'Model preparation: {exc}. Recording continues.')
            while not self.stop_event.wait(.03):
                snap = self.snapshot()
                if snap is None:
                    continue
                try:
                    self.step(snap)
                except Exception as exc:
                    self.emit('status', message=f'Evidence processing: {exc}. Recording continues.')
            if self.active is not None:
                self._finish(self.last_ts, 'Session ended during the test.')
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.detector_pool.shutdown(wait=True, cancel_futures=True)
            self.journal.close()
            self.finish_pool.shutdown(wait=True)
            self._close_intervals(self.last_ts)
            try:
                self.report()
            except Exception as exc:
                self.emit('status', message=f'PDF could not be written: {exc}. Recorded frames are saved.')

    def stop(self):
        self.stop_event.set()

    def report(self):
        with self.report_lock:
            with self.journal.lock:
                problems = copy.deepcopy(self.journal.problems)
            path = write_report(self.directory, list(self.jewels.values()), self.tests, problems)
        self.emit('report', path=path)

    def _picture(self, directory, name, frame):
        if frame is None:
            return None
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/f'{name}.jpg'
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f'Could not save {name}')
        return str(path)

    def _visibility(self, snap, ts):
        for tr in snap['tracks']:
            jid = tr['id']
            if jid not in self.jewels:
                image = self._picture(self.directory, f'J{jid}', tr.get('thumb'))
                self.jewels[jid] = dict(id=jid, image=image, visibility=[])
            status = visibility(tr, ts, snap['bench_moving'])
            if status in PICKUP_STATES:
                self.recent_hand[jid] = ts
            current = self.intervals.get(jid)
            if current and current['status'] != status:
                self.jewels[jid]['visibility'].append(dict(current, end=ts))
                current = None
            if current is None:
                current = dict(jewel_id=jid, status=status, start=ts, end=ts)
                if status == 'MISSED':
                    current['image'] = self._picture(self.directory, f'J{jid}_missed_{ts:.3f}', snap['frame'])
                self.intervals[jid] = current
            current['end'] = ts

    def _close_intervals(self, ts):
        for jid, current in self.intervals.items():
            self.jewels[jid]['visibility'].append(dict(current, end=max(current['start'], ts)))
        self.intervals.clear()

    def _new_cycle(self, jid, snap, ts):
        directory = self.directory/f"test_{len(self.tests)+1}_{uuid.uuid4().hex[:8]}"
        directory.mkdir()
        self.active = dict(id=directory.name, directory=directory, jewel_id=jid,
            started=ts, returned=None, acid_onset=None, finished=None,
            reference=self.reference, mask=None, mask_candidate=None, mask_status='UNCONFIRMED',
            mask_reason='No clear new rubbing mask confirmed yet.', live_rub=False,
            rubbing_status='UNCONFIRMED', identity_status='IN_HAND' if jid else 'MISSED_OR_AMBIGUOUS',
            issues=[], poses=[], coverage_since=None, reaction_end=None,
            detector_bbox=None, pickup_image=self._picture(directory, 'pickup', snap['frame']),
            return_frame=None, version=snap['revision'], ph_status='Not observed in continuous capture',
            baseline_at=None, mask_decision_at=None, phase_events=[], acid_checks=[],
            stop_search=threading.Event(), recovery_cursor=None, last_recovery=-1e9)
        self.active.update(previous_acid_mask=self.acid_history.mask.copy(), fresh_gate=None,
                           fresh_acid_status='WAITING_FOR_BASELINE', stone_observations=[])
        if jid is None:
            self.active['issues'].append('Jewel identity was missed or multiple jewels were under hands; no identity was guessed.')
        if self.model_error:
            self.active['issues'].append('Model preparation: '+self.model_error)
        self.emit('begin', jewel_id=jid, token=directory.name)
        self._phase('RUBBING', 'Before stone saved; follow the rubbing and return.' if self.reference else
                    'Following rubbing; before-stone reference is not ready.', self.active)

    def _phase(self, phase, message, a=None, **details):
        token = a['id'] if a else None
        key = (token, phase)
        changed = key != self.phase
        if not changed and self.last_ts-self.last_phase_emit < .2:
            return
        self.phase, self.last_phase_emit = key, self.last_ts
        if a is not None:
            if changed:
                a.setdefault('phase_events', []).append(dict(phase=phase, timestamp=self.last_ts))
            a['phase'] = phase
        self.emit('phase', phase=phase, message=message, token=token,
                  jewel_id=a.get('jewel_id') if a else None,
                  mask_state=('DETECTED / PREVIOUS ACID OVERLAP' if a.get('rub_contamination_possible') else
                              'DETECTED' if a.get('mask') is not None else
                              'NOT DETECTED' if a.get('mask_decision_at') is not None else 'CHECKING') if a else None,
                  acid_state=('DETECTED - detector stopped' if a.get('acid_onset') is not None else
                              'PREVIOUS ACID IGNORED; WAITING FOR FRESH' if 'IGNORED' in a.get('fresh_acid_status', '') else
                              'FRESH APPLICATION NOT CONFIRMED') if a and a.get('baseline_at') is not None else None,
                  **details)

    def _baseline(self, a, frame, ts):
        if a.get('baseline_at') is not None:
            return
        a['baseline_at'] = a['mask_decision_at'] = ts
        a['recovery_cursor'] = a['returned']
        a['fresh_gate'] = FreshAcidGate(self.acid_history.mask,
            a['return_frame'] if a.get('return_frame') is not None else frame, self.pose.quad)
        a['previous_acid_mask'] = self.acid_history.mask.copy()
        a['fresh_acid_status'] = 'FRESH_APPLICATION_NOT_CONFIRMED'
        a['acid_baseline_checked'] = False
        a['baseline_start_image'] = self._picture(a['directory'], 'baseline_start', frame)
        self.emit('baseline', jewel_id=a['jewel_id'], token=a['id'], timestamp=ts,
                  photo=self._stone_picture(frame, self.pose.quad), path=a['baseline_start_image'])
        self.emit('mask_decision', jewel_id=a['jewel_id'], token=a['id'], detected=a['mask'] is not None,
                  reason=a['mask_reason'], timestamp=ts,
                  contamination_possible=a.get('rub_contamination_possible', False))
        self._phase('BASELINE', 'Mask check finished. Baseline saved; waiting for acid without ending the session.', a)

    def _set_acid(self, a, observed, best, recovered=False):
        a['acid_onset'], a['detector_bbox'] = observed, best['bbox']
        a['detector_confidence'], a['acid_recovered'] = best['confidence'], recovered
        a['fresh_acid_status'] = 'FRESH_APPLICATION_DETECTED'
        a['fresh_acid_evidence'] = {key: best.get(key) for key in ('freshness', 'fresh_changed_fraction')}
        quad = best.get('quad')
        if quad is None:
            quad = self._quad_at(a, observed)
        self.acid_history.remember([best], quad)
        a.setdefault('stop_search', threading.Event()).set()
        if a.get('mask') is not None and a['mask']['timestamp'] >= observed:
            a['mask'], a['mask_status'] = None, 'UNCONFIRMED'
            a['mask_reason'] = 'Mask frame overlaps acid application; searching earlier recorded frames.'
            self.emit('clear_mask', jewel_id=a['jewel_id'], token=a['id'])
        self.emit('acid_detection', jewel_id=a.get('jewel_id'), token=a['id'], detected=True,
                  bbox=best['bbox'], confidence=best['confidence'], timestamp=observed, recovered=recovered)
        self._phase('REACTION', 'Acid detected. Acid search stopped; recording the five-second reaction.', a,
                    remaining=max(0., observed+5-self.last_ts))

    def _search_recording(self, a, start, end):
        checked = start
        for entry in self.journal.between(start, end, fps=5):
            if a['stop_search'].is_set():
                break
            quad = self._quad_at(a, entry['timestamp'])
            if quad is None:
                continue
            frame = self.journal.read(entry)
            observed, _, found = self._detect(frame, entry['timestamp'], box_of(quad), a['id'])
            checked = observed
            gate = a.get('fresh_gate')
            if gate is not None:
                best, _ = gate.check(frame, quad, found, observed,
                                     geometry_known=self._pose_known(a, observed))
                if best is not None:
                    return observed, dict(best, quad=quad), checked, gate.rejections
        return None, None, checked, a['fresh_gate'].rejections if a.get('fresh_gate') else []

    @staticmethod
    def _pose_known(a, timestamp):
        return bool(a.get('poses')) and min(abs(p[0]-timestamp) for p in a['poses']) <= .65

    def _acid_observation(self, observed, cycle_id, found, context):
        a = self.active
        if context is None:
            return  # A detection without its own source geometry cannot arm capture.
        frame, quad, clear, geometry_known = context[:4]
        baseline_sample = len(context) > 4 and context[4]
        if baseline_sample and a is not None and a['id'] == cycle_id and a.get('fresh_gate') is not None:
            self.acid_history.remember(found, quad)
            a['fresh_gate'].old_mask |= self.acid_history.mask
            a['previous_acid_mask'] |= self.acid_history.mask
            a['baseline_acid_detections'] = found
            a['acid_baseline_checked'] = True
            return
        idle_sample = a is None or observed < (a.get('returned') or float('inf'))
        if idle_sample:
            if a is None or observed < a['started']:
                self.acid_history.observe_idle(frame, quad, found, observed, clear)
                if a and a.get('fresh_gate') is None:
                    a['previous_acid_mask'] = self.acid_history.mask.copy()
            return
        if a['id'] != cycle_id or a.get('baseline_at') is None or a['acid_onset'] is not None:
            return
        gate = a.get('fresh_gate')
        if gate is None:
            return
        best, reason = gate.check(frame, quad, found, observed, geometry_known)
        a['fresh_acid_status'] = reason
        if best is not None:
            self._set_acid(a, observed, dict(best, quad=quad))
            return
        # Preserve residue found in the baseline for future jewels as well.
        self.acid_history.mask |= gate.old_mask
        a['previous_acid_mask'] |= gate.old_mask
        previous_events = len(self.acid_history.events)
        self.acid_history.observe_idle(frame, quad, found, observed, clear)
        if len(self.acid_history.events) > previous_events:
            gate.old_mask[:] = 0
            gate.baseline = normalized(frame, quad)
            a['issues'].append('Previous patch visually cleared while waiting; chemical cleanliness is not measured.')
        a.setdefault('acid_checks', []).append(dict(timestamp=observed, detected=False,
                                                   acid_present=bool(found), reason=reason))
        a['acid_checks'] = a['acid_checks'][-200:]
        self.emit('acid_detection', jewel_id=a['jewel_id'], token=a['id'], detected=False,
                  acid_present=bool(found), freshness=reason, bbox=None, confidence=0., timestamp=observed)

    def _collect_jobs(self):
        if self.reference_job and self.reference_job.done():
            job, self.reference_job = self.reference_job, None
            try:
                ref = job.result()
                if ref.revision != self.revision:
                    return
                if self.active is None and ref.timestamp > self.last_capture_end:
                    self.reference = ref
                elif self.active is not None and ref.timestamp < self.active['started'] and self.active['reference'] is None:
                    self.active['reference'] = ref
                else:
                    return
                ref.image_path = self._picture(self.directory/'references', f'before_stone_{ref.timestamp:.6f}',
                                               self._stone_picture(ref.frame, ref.quad))
                ref.frame_path = self._picture(self.directory/'references', f'before_frame_{ref.timestamp:.6f}', ref.frame)
                self.emit('reference', photo=self._stone_picture(ref.frame, ref.quad), path=ref.image_path,
                          timestamp=ref.timestamp, status=ref.before['status'])
                if self.active is None:
                    self._phase('BEFORE_READY', 'Before-stone image captured and saved. Ready for rubbing.')
            except Exception as exc:
                self.emit('status', message=f'Reference evidence unavailable: {evidence_reason(exc)}. Recording continues.')
        if self.mask_job and self.mask_job[0].done():
            future, cycle_id = self.mask_job
            self.mask_job = None
            a = self.active
            if a is not None and a['id'] == cycle_id:
                try:
                    result = future.result()
                    if a['acid_onset'] is not None and result.get('timestamp', float('inf')) >= a['acid_onset']:
                        return
                    if result['status'] == 'NEW_RUB':
                        a['mask_candidate'] = result
                    else:
                        a['mask_reason'] = evidence_reason(result['message'])
                        a['mask_attempt_at'] = result.get('timestamp', self.last_ts)
                except Exception as exc:
                    a['mask_reason'] = evidence_reason(exc)
                    a['mask_attempt_at'] = self.last_photo

    @staticmethod
    def _stone_picture(frame, quad):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box_of(quad)
        return frame[max(0, y1-12):min(h, y2+12), max(0, x1-12):min(w, x2+12)].copy()

    def _confirm_candidate(self, a, frame, ts):
        candidate = a.get('mask_candidate')
        if candidate is None or a['mask'] is not None or a['returned'] is None or a['acid_onset'] is not None:
            return
        # A mask prepared as the jewel moves away must still look like the same
        # streak after return. A held object disappearing is not a persistent mark.
        transform = cv2.getPerspectiveTransform(self.pose.quad.astype(np.float32), DEST)
        previous = cv2.warpPerspective(candidate['frame'], candidate['frame_to_reference'], (512, 512))
        current = cv2.warpPerspective(frame, transform, (512, 512))
        delta = cv2.absdiff(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), cv2.cvtColor(current, cv2.COLOR_BGR2GRAY))
        pixels = delta[candidate['new_mask'] > 0]
        if not pixels.size or float(np.median(pixels)) > 20:
            a['mask_candidate'] = None
            a['mask_reason'] = 'Candidate mark did not remain consistent after jewel return.'
            return
        candidate['roi_mask'] = project_mask(candidate['new_mask'], frame.shape, transform)
        candidate['photo'] = draw_acid_roi(frame.copy(), candidate['roi_mask'])
        a['mask'], a['mask_status'], a['mask_reason'] = candidate, 'CONFIRMED_LIVE', ''
        a['mask_completed_at'] = time.time()
        old = a.get('previous_acid_mask')
        target = cv2.resize(candidate['new_mask'], (256,256), interpolation=cv2.INTER_NEAREST) > 0
        contaminated = old is not None and np.any(target & (old > 0))
        a['rub_contamination_possible'] = bool(contaminated)
        if contaminated:
            note = 'New rubbing overlaps previous acid; contamination possible.'
            if note not in a['issues']:
                a['issues'].append(note)
        self.emit('mask', jewel_id=a['jewel_id'], token=a['id'], mask=candidate['roi_mask'],
                  stone=box_of(self.pose.quad), photo=candidate['photo'], contamination_possible=bool(contaminated))

    def _detect(self, frame, ts, stone, cycle_id):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = stone
        pad = .12*max(x2-x1, y2-y1)
        x1, y1 = max(0, int(x1-pad)), max(0, int(y1-pad))
        x2, y2 = min(w, int(x2+pad)), min(h, int(y2+pad))
        if x2 <= x1 or y2 <= y1:
            return ts, cycle_id, []
        dets, _ = self.detector.predict(frame[y1:y2, x1:x2], .60)
        found = []
        for det in dets:
            box = tuple(int(v) for v in np.asarray(det.bbox)+[x1, y1, x1, y1])
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            if stone[0] <= cx <= stone[2] and stone[1] <= cy <= stone[3]:
                found.append(dict(bbox=box, confidence=det.confidence))
        return ts, cycle_id, found

    def step(self, snap):
        ts, frame = snap['timestamp'], snap['frame']
        self._collect_jobs()
        if self.recovery_job and self.recovery_job[0].done():
            future, cycle_id = self.recovery_job
            self.recovery_job = None
            a = self.active
            if a is not None and a['id'] == cycle_id and a['acid_onset'] is None:
                try:
                    observed, best, checked, rejected = future.result()
                    if a.get('fresh_gate') is not None:
                        a['fresh_gate'].rejections.extend(rejected)
                        a['fresh_gate'].rejections = a['fresh_gate'].rejections[-100:]
                        if rejected and not any(note.startswith('Recorded acid observations also') for note in a['issues']):
                            a['issues'].append('Recorded acid observations also required fresh-application evidence; previous/uncertain patches were ignored.')
                    if rejected:
                        a['fresh_acid_status'] = rejected[-1]['reason']
                    a['recovery_cursor'] = checked+.000001
                    if best is not None:
                        self._set_acid(a, observed, best, recovered=True)
                except Exception as exc:
                    a['issues'].append('Recorded acid check: '+str(exc))
        if self.acid_job and self.acid_job.done():
            job, self.acid_job = self.acid_job, None
            try:
                observed, cycle_id, found = job.result()
                self.last_acid_ts, self.last_acid_present = observed, bool(found)
                context, self.acid_context = self.acid_context, None
                self._acid_observation(observed, cycle_id, found, context)
            except Exception as exc:
                if self.active:
                    self.active['issues'].append('Live acid detector: '+str(exc))
        if ts <= self.last_ts or snap['mode'] not in ('ARMING', 'LOCKED'):
            return
        self.last_ts = ts
        if self.revision != snap['revision']:
            if self.active:
                self.active['issues'].append('Tracking ROI changed; the acid session remained active.')
                self.active['version'] = snap['revision']
                if self.mask_job is not None:
                    self.mask_job[0].cancel()
                    self.mask_job = None
            self.revision = snap['revision']
            self.pose, self.reference = StonePose(), None
            self.recent_hand.clear()
            self.placement = snap.get('placement')
            self.placement_auto = self.placement is None
            self.last_reference = -1e9
        self._visibility(snap, ts)
        stone = snap.get('stone')
        if stone is None and self.pose.quad is not None:
            stone = box_of(self.pose.quad)
        clear = self.pose.update(frame, ts, stone)
        if clear and (self.placement is None or (self.placement_auto and snap['mode'] == 'ARMING')):
            b = np.asarray(box_of(self.pose.quad), float)
            pad = .2*max(b[2:]-b[:2])
            b += [-pad, -pad, pad, pad]
            if self.placement is not None:
                # A hand can truncate the first stone observation. Grow only
                # during initial setup; never chase the stone during a test.
                previous = box_of(self.placement)
                b = np.array([min(b[0], previous[0]), min(b[1], previous[1]),
                              max(b[2], previous[2]), max(b[3], previous[3])])
            placement = np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
            if self.placement is None or not np.array_equal(placement, self.placement):
                self.placement = placement
                self.emit('placement', roi=self.placement)
        eligible_tracks = [tr for tr in snap['tracks'] if tr['id'] not in self.captured_ids]
        candidate = choose_hand_jewel(eligible_tracks, ts, snap['bench_moving'], self.recent_hand)
        current_hand = [tr for tr in eligible_tracks if not tr.get('acid_done')
                        and visibility(tr, ts, snap['bench_moving']) in HAND]
        pickup = any(visibility(tr, ts, snap['bench_moving']) in PICKUP_STATES for tr in current_hand)
        a = self.active
        if a and a['returned'] is not None and a['acid_onset'] is None and candidate is not None and candidate != a['jewel_id']:
            if not a.get('next_jewel_seen'):
                a['issues'].append('Another jewel was handled before acid was detected; the original acid session stayed active.')
            a['next_jewel_seen'] = candidate
            a['identity_status'] = 'AMBIGUOUS_AFTER_NEXT_JEWEL_HANDLING'
            a['acid_jewel_association'] = 'UNVERIFIED_MULTIPLE_JEWELS'
        missed_rubbing = snap.get('rubbing') and any(
            not tr.get('acid_done') and visibility(tr, ts, snap['bench_moving']) in ('MISSED', 'NOT_SEEN')
            for tr in eligible_tracks)
        if a is None and snap['mode'] == 'LOCKED' and (pickup or missed_rubbing):
            self._new_cycle(candidate, snap, ts)
            a = self.active
        if a is not None:
            if stone is not None:
                a['stone_observations'].append((ts, tuple(int(v) for v in stone)))
            if clear:
                a['poses'].append((ts, self.pose.quad.copy()))
            if snap.get('rubbing'):
                a['live_rub'] = True
                a['last_rub'] = ts
                a['rubbing_status'] = 'CONFIRMED_LIVE'
            own = next((tr for tr in snap['tracks'] if tr['id'] == a['jewel_id']), None)
            if own and visibility(own, ts, snap['bench_moving']) == 'MISSED':
                a['identity_status'] = 'MISSED_DURING_PROCESS'
            returned = (own is not None and visibility(own, ts, snap['bench_moving']) == 'VISIBLE'
                        and stone is not None and not (stone[0] <= own['pos'][0] <= stone[2]
                                                      and stone[1] <= own['pos'][1] <= stone[3]))
            if returned and own.get('box') is not None:
                b = own['box']
                intersection = max(0, min(b[2], stone[2])-max(b[0], stone[0]))*max(0, min(b[3], stone[3])-max(b[1], stone[1]))
                returned = intersection/max(1., (b[2]-b[0])*(b[3]-b[1])) < .05
            if own is None or a['identity_status'] == 'MISSED_DURING_PROCESS':
                # Keep the camera process moving, but never relabel this as a returned jewel.
                returned = clear and ts-a.get('last_rub', a['started']) >= 2
            # Prepare a candidate as soon as the stone clears, before waiting
            # for the tracker to mark the jewel's return. This work overlaps the
            # operator's handoff instead of adding a model pass after it.
            occupied = False
            if stone is not None:
                for tr in snap['tracks']:
                    b = tr.get('box')
                    if b is None or ts-tr['det_seen'] > .35:
                        continue
                    overlap = max(0, min(b[2], stone[2])-max(b[0], stone[0]))*max(0, min(b[3], stone[3])-max(b[1], stone[1]))
                    if overlap/max(1., (b[2]-b[0])*(b[3]-b[1])) >= .05:
                        occupied = True
            if (a['acid_onset'] is None and a['mask'] is None and a.get('mask_candidate') is None
                    and self.mask_job is None and clear and not occupied
                    and ts-self.last_photo >= .25 and a['reference'] is not None):
                self.last_photo = ts
                self.mask_job = (self.pool.submit(a['reference'].after, self.models, frame, ts,
                                                self.pose.quad.copy()), a['id'])
            if a['returned'] is None and returned and clear and inside_roi(self.pose.quad, self.placement):
                a['return_detected'] = a.get('return_detected', ts)
                if self.pose.stable_since is None or ts-self.pose.stable_since < .10:
                    return
                a['returned'], a['return_frame'] = a['return_detected'], frame
                a['decision_deadline'] = a['returned']+1.
                self._phase('MASK_CHECK', 'Jewel returned and stone inside ROI; checking the new rubbed mask.', a)
            if a['returned'] is not None:
                if clear and not occupied:
                    self._confirm_candidate(a, frame, ts)
                if a['reference'] is None:
                    a['mask_reason'] = 'No clear pre-rubbing reference was available.'
                if a.get('baseline_at') is None:
                    if a['mask'] is not None or a.get('mask_attempt_at', -1e9) >= a['returned'] or ts >= a['decision_deadline']:
                        self._baseline(a, frame, ts)
                    else:
                        self._phase('MASK_CHECK', 'New rubbed mask: checking. Baseline begins after this check.', a)
                if a['mask'] is not None and clear:
                    roi_mask = project_mask(a['mask']['new_mask'], frame.shape,
                                           cv2.getPerspectiveTransform(self.pose.quad.astype(np.float32), DEST))
                    self.emit('target', jewel_id=a['jewel_id'], token=a['id'], mask=roi_mask,
                              stone=box_of(self.pose.quad), uncertain=False)
                elif a['mask'] is not None:
                    self.emit('target', jewel_id=a['jewel_id'], token=a['id'], uncertain=True)
                if a['acid_onset'] is not None:
                    self._phase('REACTION', 'Acid DETECTED; acid search stopped. Recording the reaction.', a,
                                remaining=max(0., a['acid_onset']+5-ts))
                    # Broad surface coverage after application indicates hand/cloth interference.
                    if surface_covered(frame, stone):
                        a['coverage_since'] = a['coverage_since'] or ts
                    else:
                        a['coverage_since'] = None
                    if a['coverage_since'] and ts-a['coverage_since'] >= .3:
                        if a['reaction_end'] is None:
                            a['reaction_end'] = a['coverage_since']
                            a['issues'].append('Stone broadly covered/wiped during reaction; usable duration is recorded separately.')
                    if ts >= a['acid_onset']+5:
                        self._finish(ts)
                elif a.get('baseline_at') is not None:
                    if ts-a['baseline_at'] >= .3:
                        self._phase('WAIT_FOR_ACID',
                            ('Previous/baseline acid ignored; fresh application not confirmed. Session stays active.'
                             if 'IGNORED' in a.get('fresh_acid_status', '') else
                             'Fresh acid application not confirmed. Session stays active; checking live and recorded frames.'), a)
                    if (a.get('acid_baseline_checked') and ts-a['baseline_at'] >= 2 and self.recovery_job is None and self.detector is not None
                            and ts-a['last_recovery'] >= 2):
                        a['last_recovery'] = ts
                        self.recovery_job = (self.finish_pool.submit(self._search_recording,
                            dict(a, poses=list(a['poses']), fresh_gate=a['fresh_gate'].fork()), a['recovery_cursor'], ts-.2), a['id'])
            else:
                if returned:
                    self._phase('WAIT_FOR_PLACEMENT', 'Jewel returned; waiting for a stable clear stone inside the test ROI.', a)
                else:
                    self._phase('RUBBING', 'Following the held jewel; waiting for return and stone placement.', a)
        remaining_jewels = snap['mode'] == 'ARMING' or any(not tr.get('acid_done') for tr in eligible_tracks)
        armed = self.active is not None and self.active.get('baseline_at') is not None and self.active['acid_onset'] is None
        idle = self.active is None and remaining_jewels
        interval = .18 if armed else .5
        if ((armed or idle) and stone is not None and self.detector is not None and self.acid_job is None
                and ts-self.last_detection >= interval):
            self.last_detection = ts
            a = self.active
            quad = None if self.pose.quad is None else self.pose.quad.copy()
            baseline_sample = armed and not a.get('acid_baseline_checked', False)
            source, source_ts = frame, ts
            if baseline_sample:
                source = a['return_frame']
                source_ts = a['returned']
                quad = self._quad_at(a, source_ts)
            self.acid_context = (source, quad, clear and self.pose.stable_since is not None
                                 and ts-self.pose.stable_since >= .1,
                                 quad is not None and self.pose.last_clear is not None and ts-self.pose.last_clear <= .65,
                                 baseline_sample)
            self.acid_job = self.detector_pool.submit(self._detect, source, source_ts,
                box_of(quad) if baseline_sample and quad is not None else stone, a['id'] if a else None)
        if (idle and not current_hand and clear and self.reference_job is None
                and ts-self.last_reference >= 2 and ts-self.last_acid_ts < 1):
            self.last_reference = ts
            self._phase('BEFORE_CAPTURE', 'Capturing the clear stone before rubbing.')
            self.reference_job = self.pool.submit(RubReference, self.models, frame, ts, self.pose.quad.copy(), self.revision)
        if idle and self.reference is None and self.reference_job is None and ts-self.last_notice >= 2:
            self.last_notice = ts
            self._phase('WAIT_FOR_STONE', 'Waiting for a clear stone to save the before image; recording continues.')

    def _finish(self, ts, issue=None):
        a, self.active = self.active, None
        if a is None:
            return
        a['finished'] = ts
        if a['jewel_id'] is not None:
            self.captured_ids.add(a['jewel_id'])
        a.setdefault('stop_search', threading.Event()).set()
        self.last_capture_end = ts
        if issue:
            a['issues'].append(issue)
        intervals = []
        for jewel in self.jewels.values():
            intervals.extend(jewel['visibility'])
        intervals.extend(self.intervals.values())
        a['visibility'] = [dict(i, start=max(i['start'], a['started']), end=min(i['end'], ts))
                           for i in intervals if i['end'] > a['started'] and i['start'] < ts and i['status'] != 'VISIBLE']
        self.reference = None
        complete = a['acid_onset'] is not None and ts >= a['acid_onset']+5
        self._phase('PROCESSING', 'Reaction capture finished; calculating the acid result.' if complete else
                    'Session stopped; saving incomplete test evidence.', a)
        self.emit('captured', jewel_id=a['jewel_id'], token=a['id'], capture_complete=complete)
        self.finish_pool.submit(self._recover_and_save, a)

    @staticmethod
    def _quad_at(a, ts):
        poses = a['poses']
        return min(poses, key=lambda p: abs(p[0]-ts))[1] if poses else None

    def _recover_and_save(self, a):
        try:
            self._save(a)
        except Exception as exc:
            # Preserve a visible terminal record even if a model/file operation failed.
            a['issues'].append('Recovery/save error: '+str(exc))
            record = a.get('saved_record') or self._record(a)
            self._commit(record)

    def _record(self, a):
        keys = ('jewel_id', 'started', 'returned', 'acid_onset', 'finished', 'rubbing_status',
                'mask_status', 'mask_reason', 'identity_status', 'ph_status', 'issues', 'visibility', 'pickup_image',
                'baseline_at', 'mask_decision_at', 'baseline_start_image', 'phase_events', 'acid_checks',
                'fresh_acid_status', 'fresh_acid_evidence', 'baseline_acid_detections', 'acid_jewel_association')
        record = {key: a.get(key) for key in keys}
        record.update(prediction=None, acid_status='NOT_DETECTED', record_path=str(a['directory']/'manifest.json'),
                      detector_bbox=list(a['detector_bbox']) if a.get('detector_bbox') else None,
                      actual_label=None, schema='jewel_continuous_prediction_v1',
                      alignment_mode='tracked_corners_limited_motion_assumption',
                      decision_deadline=a.get('decision_deadline'), mask_completed_at=a.get('mask_completed_at'))
        record['capture_complete'] = a['acid_onset'] is not None and a['finished'] >= a['acid_onset']+4.99
        gate = a.get('fresh_gate')
        record['fresh_acid_rejections'] = list(gate.rejections) if gate else []
        return record

    def _save(self, a):
        # Allow the independent recorder to persist the newest queued evidence.
        self.journal.flush_through(a['finished'])
        start, end = a['started'], a['finished']
        if (a['acid_onset'] is None and a['returned'] is not None and self.detector is not None
                and a.get('fresh_gate') is not None):
            if not a.get('acid_baseline_checked'):
                quad = self._quad_at(a, a['returned'])
                if quad is not None and a.get('return_frame') is not None:
                    _, _, baseline_found = self._detect(a['return_frame'], a['returned'], box_of(quad), a['id'])
                    for detection in baseline_found:
                        known = footprint(detection['bbox'], quad)
                        a['fresh_gate'].old_mask |= known
                        a['previous_acid_mask'] |= known
                    a['baseline_acid_detections'] = baseline_found
                    a['acid_baseline_checked'] = True
            recovery = dict(a, fresh_gate=a['fresh_gate'].fork())
            # The active search is cancelled at stop; this bounded export uses
            # its own cancellation token and cannot bypass the freshness gate.
            recovery['stop_search'] = threading.Event()
            observed, best, _, rejected = self._search_recording(recovery, a['returned'], end)
            a['fresh_gate'].rejections.extend(rejected)
            if best is not None:
                a['acid_onset'], a['detector_bbox'] = observed, best['bbox']
                a['acid_recovered'] = True
                a['fresh_acid_status'] = 'FRESH_APPLICATION_DETECTED'
                a['fresh_acid_evidence'] = {key: best.get(key) for key in ('freshness', 'fresh_changed_fraction')}
        onset = a['acid_onset']
        # Recover the rub from pre-acid frames only; liquid must not become a new mark.
        if a['mask'] is None and a['reference'] is not None:
            until = onset-.05 if onset is not None else end
            entries = self.journal.between(a['returned'] or start, until, fps=2)
            indices = np.linspace(0, len(entries)-1, min(4, len(entries)), dtype=int)
            for index in indices:
                entry = entries[index]
                try:
                    result = a['reference'].after(self.models, self.journal.read(entry), entry['timestamp'],
                                                 self._quad_at(a, entry['timestamp']))
                    if result['status'] == 'NEW_RUB':
                        a['mask'], a['mask_status'], a['mask_reason'] = result, 'RECOVERED_FROM_CLIP', ''
                        break
                    a['mask_reason'] = evidence_reason(result['message'])
                except Exception as exc:
                    a['mask_reason'] = evidence_reason(exc)
        if not a['live_rub'] and a['mask'] is not None:
            a['rubbing_status'] = ('NEW_MARK_CONFIRMED_LIVE' if a['mask_status'] == 'CONFIRMED_LIVE'
                                   else 'NEW_MARK_RECOVERED_FROM_CLIP')
        record = self._record(a)
        a['saved_record'] = record
        if a['reference'] is not None:
            record['before_image'] = getattr(a['reference'], 'image_path', None) or self._picture(
                a['directory'], 'before_stone', self._stone_picture(a['reference'].frame, a['reference'].quad))
            record['before_timestamp'] = a['reference'].timestamp
            record['before_status'] = a['reference'].before['status']
        else:
            record['before_status'] = 'NOT_CAPTURED'
            record['issues'].append('A clear before-stone reference was not captured before rubbing.')
        if a['mask'] is not None:
            record['mask_image'] = self._picture(a['directory'], 'new_rub', a['mask']['photo'])
            cv2.imwrite(str(a['directory']/'new_rub_mask.png'), a['mask']['roi_mask'])
        else:
            a['issues'].append('New rubbed mask could not be confirmed: '+a['mask_reason'])
        contact = recover_contact(self.journal, a, self.rub_observer, self._picture)
        record.update(contact)
        if not a['live_rub'] and contact['rubbing_contact_status'] == 'CONTACT_AND_MOTION_RECOVERED':
            a['rubbing_status'] = record['rubbing_status'] = 'RUBBING_RECOVERED_FROM_CLIP'
        if contact['rubbing_contact_status'] == 'CONTACT_NOT_VISIBLE':
            a['issues'].append(contact['rubbing_contact_reason'])
        if not a['live_rub'] and a['mask'] is None:
            a['issues'].append('Jewel/rubbing was not visible or no usable new mark was detected; no repeat was requested.')
        record['rubbing_clip'] = self.journal.clip(start-1, a['returned'] or end, a['directory']/'pickup_rubbing_return.avi')
        if onset is not None:
            reaction_end = min(onset+5, end, a['reaction_end'] if a['reaction_end'] is not None else float('inf'))
            record['acid_status'] = 'RECOVERED_FROM_CLIP' if a.get('acid_recovered') else 'DETECTED_LIVE_ONE_HIT'
            record['acid_clip'] = self.journal.clip(onset-2, min(onset+5, end), a['directory']/'acid_2s_before_5s_after.avi')
            h, w = (a['return_frame'] if a['return_frame'] is not None else a['reference'].frame).shape[:2]
            x1, y1, x2, y2 = a['detector_bbox']
            px, py = .45*(x2-x1), .45*(y2-y1)
            box = (max(0, int(x1-px)), max(0, int(y1-py)), min(w, int(x2+px)), min(h, int(y2+py)))
            if a['mask'] is not None:
                onset_quad = self._quad_at(a, onset)
                mask = project_mask(a['mask']['new_mask'], (h, w, 3),
                                    cv2.getPerspectiveTransform(onset_quad.astype(np.float32), DEST))
                smaller = min(np.count_nonzero(mask), max(1, (x2-x1)*(y2-y1)))
                overlap = np.count_nonzero(mask[y1:y2, x1:x2])/max(1, smaller)
                record['acid_new_mask_overlap'] = float(overlap)
                if overlap < .5:
                    a['issues'].append('Detected acid did not sufficiently overlap the new rubbed mask; acid model output is retained separately.')
            else:
                record['acid_new_mask_overlap'] = None
            record['acid_target_status'] = ('COMPARISON_UNAVAILABLE' if a['mask'] is None else
                'ACID_OVERLAPS_NEW_RUB' if record['acid_new_mask_overlap'] >= .5 else 'ACID_AT_DIFFERENT_LOCATION')
            onset_quad = self._quad_at(a, onset)
            shift = np.float32([[1, 0, -box[0]], [0, 1, -box[1]], [0, 0, 1]])
            def frames(lo, hi):
                result = []
                for entry in self.journal.between(lo, hi, fps=5):
                    frame = self.journal.read(entry)
                    quad = self._quad_at(a, entry['timestamp'])
                    if quad is not None and onset_quad is not None:
                        transform = shift @ cv2.getPerspectiveTransform(quad.astype(np.float32), onset_quad.astype(np.float32))
                        crop = cv2.warpPerspective(frame, transform, (box[2]-box[0], box[3]-box[1]))
                    else:
                        crop = frame[box[1]:box[3], box[0]:box[2]]
                    if crop.size:
                        result.append(crop.copy())
                return result
            baseline = frames(onset-2, onset-.001)
            transition = frames(onset, min(onset+.5, reaction_end))
            observation = frames(onset, reaction_end)
            record['frame_counts'] = dict(baseline=len(baseline), application=len(transition), reaction=len(observation))
            record['reaction_bbox'] = list(box)
            record['reaction_crop_alignment'] = 'stone_corners_to_acid_onset'
            record['acid_window_mode'] = '2s_before_first_detection_5s_after_source_timestamp'
            record['usable_reaction_seconds'] = max(0., reaction_end-onset)
            if reaction_end < onset+4.9:
                a['issues'].append(f'Only {max(0., reaction_end-onset):.2f} seconds of reaction evidence available before coverage/session end.')
            if a['returned'] is None or onset-2 < a['returned']:
                a['issues'].append('The two-second baseline includes jewel return/hand or unsettled footage.')
            if baseline:
                record['baseline_image'] = self._picture(a['directory'], 'baseline', baseline[-1])
            if observation:
                record['acid_image'] = self._picture(a['directory'], 'acid_reaction', observation[-1])
            detection_frames = self.journal.between(onset-.15, onset+.15)
            if detection_frames:
                entry = min(detection_frames, key=lambda e: abs(e['timestamp']-onset))
                detection = self.journal.read(entry)
                comparison = detection.copy()
                if a['mask'] is not None:
                    current_quad = self._quad_at(a, entry['timestamp'])
                    rub_mask = project_mask(a['mask']['new_mask'], comparison.shape,
                        cv2.getPerspectiveTransform(current_quad.astype(np.float32), DEST))
                    contours, _ = cv2.findContours(rub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(comparison, contours, -1, (255,0,255), 2)
                cv2.rectangle(comparison, (x1,y1), (x2,y2), (255,255,0), 2)
                crop = self._stone_picture(comparison, self._quad_at(a, entry['timestamp']))
                legend = np.full((75, max(500, crop.shape[1]), 3), 245, np.uint8)
                cv2.putText(legend, 'MAGENTA: new rub | CYAN: acid', (8,25), cv2.FONT_HERSHEY_SIMPLEX, .55, (40,40,40), 1)
                cv2.putText(legend, record['acid_target_status'], (8,53), cv2.FONT_HERSHEY_SIMPLEX, .5, (40,40,40), 1)
                canvas = np.full((crop.shape[0]+75, legend.shape[1], 3), 245, np.uint8)
                canvas[:crop.shape[0], :crop.shape[1]] = crop
                canvas[crop.shape[0]:] = legend
                record['acid_rub_comparison_image'] = self._picture(a['directory'], 'rub_and_acid_locations', canvas)
                cv2.rectangle(detection, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(detection, 'ACID DETECTED', (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,0), 2)
                record['acid_detection_image'] = self._picture(a['directory'], 'acid_detected', detection)
            if baseline and transition and observation and self.classifier is not None:
                # Preserve the exact supported runtime output, even with process warnings.
                record['prediction'] = self.classifier.predict(baseline, transition, observation)
            else:
                a['issues'].append('Acid model could not run because baseline/application/reaction frames or the model were unavailable.')
            for folder, images in [('baseline', baseline), ('application', transition), ('reaction', observation)]:
                for index, image in enumerate(images):
                    self._picture(a['directory']/folder, f'{index:04d}', image)
        else:
            record['acid_clip'] = self.journal.clip((a['returned'] or start)-2, end, a['directory']/'acid_search.avi')
            a['issues'].append('Acid was not detected live or in recorded footage; no acid model result was fabricated.')
            a['issues'].append('Fresh application not confirmed; old/baseline acid and uncertain observations did not start a reaction timer.')
        old = a.get('previous_acid_mask')
        if old is not None:
            record['previous_acid_mask_image'] = str(a['directory']/'previous_acid_stone_relative.png')
            cv2.imwrite(record['previous_acid_mask_image'], old)
            if a['mask'] is not None:
                target = cv2.resize(a['mask']['new_mask'], (256,256), interpolation=cv2.INTER_NEAREST) > 0
                record['rub_previous_acid_overlap'] = float(np.count_nonzero(target & (old > 0))/max(1,np.count_nonzero(target)))
                if record['rub_previous_acid_overlap'] > 0:
                    note = 'New rubbing overlaps previous acid; contamination possible.'
                    if note not in a['issues']:
                        a['issues'].append(note)
        self._commit(record)

    def _commit(self, record):
        try:
            atomic_json(record['record_path'], record)
        except OSError as exc:
            record['issues'].append('Prediction archive write failed: '+str(exc))
        with self.report_lock:
            existing = next((i for i, test in enumerate(self.tests) if test['record_path'] == record['record_path']), None)
            if existing is None:
                self.tests.append(record)
            else:
                self.tests[existing] = record
        self.emit('result', record=record)
        if self.active is None and record.get('capture_complete'):
            from session_evidence import prediction_text
            self._phase('DONE', prediction_text(record.get('prediction'))+'; result saved.')
        try:
            self.report()
        except Exception as exc:
            self.emit('status', message=f'PDF write failed: {exc}. The prediction and recorded evidence are saved.')
