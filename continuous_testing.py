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
import json
import pickle
import os
from collections import deque

import cv2
import numpy as np

from new_rub_detection.flow import RubCheck
from new_rub_detection.geometry import DEST, project_mask
from rub_preparation import clear_surface, draw_acid_roi
from session_evidence import FrameJournal, atomic_json, write_report
from acid_freshness import AcidHistory, FreshAcidGate, footprint, normalized, DEST as ACID_DEST
from acid_application import AcidApplicationVerifier
from acid_area import AcidAreaMeasurement, region_box, compare_area
from rubbing_recovery import recover_contact
from sop_evidence import contact_region, validate_mark, local_motion

HAND = {'IN_HAND', 'UNDER_HAND'}
# Trial policy: covered jewels remain tracked, but cannot start a test cycle.
PICKUP_STATES = {'IN_HAND'}
PHASE_TITLES = {
    'BEFORE_CAPTURE': 'CAPTURING BEFORE STONE', 'BEFORE_READY': 'BEFORE STONE CAPTURED',
    'WAIT_FOR_STONE': 'WAITING FOR CLEAR STONE', 'RUBBING': 'RUBBING CONTACT OBSERVED',
    'PICKUP': 'JEWEL IN HAND / CONTACT NOT CONFIRMED',
    'WAIT_FOR_PLACEMENT': 'WAITING FOR STONE INSIDE TEST ROI',
    'MASK_CHECK': 'CHECKING NEW RUBBED MASK', 'BASELINE': 'BASELINE CAPTURE',
    'WAIT_FOR_ACID': 'WAITING FOR ACID', 'REACTION': '5 SECOND REACTION',
    'ACID_CANDIDATE': 'RECORDING / VERIFYING ACID CANDIDATE',
    'PROCESSING': 'PROCESSING ACID RESULT', 'DONE': 'DONE',
    'INCOMPLETE':'FINALIZED / INCOMPLETE EVIDENCE',
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
    if (acid and acid.get('detected') and acid.get('bbox')
            and acid.get('area_status','').startswith('ESTIMATED_') and state.get('phase') == 'REACTION'):
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


def reaction_covered(frame, box):
    """Reject sustained skin/cloth coverage of the actual classifier region."""
    if box is None:return False
    h,w=frame.shape[:2]
    x1,y1,x2,y2=box
    px,py=.45*(x2-x1),.45*(y2-y1)
    x1,y1=max(0,int(x1-px)),max(0,int(y1-py))
    x2,y2=min(w,int(x2+px)),min(h,int(y2+py))
    crop=frame[y1:y2,x1:x2]
    if not crop.size:return True
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    skin=(hsv[:,:,0]<=14)&(hsv[:,:,1]>=45)&(hsv[:,:,2]>80)
    cloth=(hsv[:,:,1]<40)&(hsv[:,:,2]>150)
    return float(skin.mean())>.3 or float(cloth.mean())>.8


def reaction_region_covered(frame, box, source_quad, current_quad, clear):
    """Coverage must follow the same stone-relative region as reaction crops."""
    if not clear or source_quad is None or current_quad is None or box is None:
        return True
    x1,y1,x2,y2=box
    corners=np.float32([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]])
    transform=cv2.getPerspectiveTransform(np.asarray(source_quad,np.float32),np.asarray(current_quad,np.float32))
    points=cv2.perspectiveTransform(corners,transform)[0]
    if not np.isfinite(points).all():
        return True
    lo,hi=np.floor(points.min(0)).astype(int),np.ceil(points.max(0)).astype(int)
    return reaction_covered(frame,tuple(map(int,[*lo,*hi])))


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
        result['photo_timestamp']=timestamp
        result['photo_quad']=quad
        if result['status'] == 'NEW_RUB':
            result['roi_mask'] = project_mask(result['new_mask'], frame.shape, result['frame_to_reference'])
        return result


class ContinuousMonitor(threading.Thread):
    def __init__(self, directory, snapshot, models, acid_models, emit, rub_observer=None, inactivity_seconds=None, pickup_debounce=None):
        super().__init__(name='continuous-tests', daemon=True)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal = FrameJournal(self.directory)
        self.snapshot,self.models,self.acid_models,self._emit_callback=snapshot,models,acid_models,emit
        self.emit=self._emit_versioned
        self.cycle_versions={}
        self.rub_observer = rub_observer
        self.acid_history = AcidHistory()
        self.acid_context = None
        config=dict(acid_inactivity_seconds=12.,next_pickup_debounce_seconds=.3,
                    max_recovery_acid_frames=60,max_recovery_contact_frames=24)
        config_path=Path(__file__).with_name('sop_config.json')
        if config_path.is_file():
            config.update(json.loads(config_path.read_text()))
        self.inactivity_seconds=max(.1,float(config['acid_inactivity_seconds'] if inactivity_seconds is None else inactivity_seconds))
        self.pickup_debounce=max(0.,float(config['next_pickup_debounce_seconds'] if pickup_debounce is None else pickup_debounce))
        self.recovery_acid_frames=max(1,int(config['max_recovery_acid_frames']))
        self.recovery_contact_frames=max(1,int(config['max_recovery_contact_frames']))
        self.next_pickup=None
        self.clear_history=deque(maxlen=8)
        self.reference_owner=None
        self.detector_lock=threading.Lock()
        self.report_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='audit-pdf')
        self.report_future=None
        self.report_dirty=False
        self.final_lock=threading.Lock()
        self.final_pending=0
        self.deferred_dir=self.directory/"deferred_recovery"
        self.deferred_dir.mkdir(exist_ok=True)

        self.stop_event = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='live-rub-photo')
        self.detector_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='live-acid-roi')
        self.finish_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='clip-recovery')
        self.report_lock = threading.Lock()
        self.report_schedule_lock=threading.Lock()
        self.tests, self.jewels = [], {}
        self.intervals, self.recent_hand = {}, {}
        self.pose, self.reference = StonePose(), None
        self.reference_job = self.mask_job = self.acid_job = None
        self.area_job = None
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
        self.last_cycle_id=None

    def _emit_versioned(self,kind,**data):
        record=data.get('record')
        revision=record.get('revision') if record is not None else self.cycle_versions.get(data.get('token'),self.revision)
        data.setdefault('revision',revision)
        self._emit_callback(kind,**data)

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
            self.drain_finalization()
            self.finish_pool.shutdown(wait=True)
            self.report_pool.shutdown(wait=True)
            self._close_intervals(self.last_ts)
            try:
                self.report()
            except Exception as exc:
                self.emit('status', message=f'PDF could not be written: {exc}. Recorded frames are saved.')

    def stop(self):
        self.stop_event.set()

    def request_report(self):
        # Coalesce repeated requests: only one PDF writer runs at a time.
        with self.report_schedule_lock:
            self.report_dirty=True
            if self.report_future is None or self.report_future.done():
                self.report_dirty=False
                self.report_future=self.report_pool.submit(self.report)

    def report(self):
        with self.report_lock:
            with self.journal.lock:
                problems = copy.deepcopy(self.journal.problems)
            path = write_report(self.directory,[copy.deepcopy(j) for j in list(self.jewels.values())],copy.deepcopy(self.tests),problems)
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

    @staticmethod
    def _residue_snapshot(a, residue):
        old=a.get('residue_warning')
        if old in a['issues']:a['issues'].remove(old)
        a['previous_acid_state']=residue
        a['residue_warning']=None
        if residue['prior_location_known'] and residue['status']!='VISUALLY_CLEARED':
            warning=('Previous acid was still visible before rubbing; contamination possible.'
                     if residue['status']=='PREVIOUS_ACID_VISIBLE' else
                     'Previous acid clearance was not verified before rubbing; contamination possible.')
            a['residue_warning']=warning
            a['issues'].append(warning)

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
        self.last_cycle_id=directory.name
        self.cycle_versions[directory.name]=snap['revision']
        self.active.update(placement=None if self.placement is None else self.placement.copy(),contact=None, contact_region=None, mask_evidence_status="NOT_ESTABLISHED",
                           last_activity=ts, activity_frame=None, rejected_mask=None, unverified_mask=None,
                           pickup_image_timestamp=ts,recovery_contact_frames=self.recovery_contact_frames,
                           previous_acid_mask=self.acid_history.mask.copy(), fresh_gate=None,
                           fresh_acid_status='WAITING_FOR_BASELINE', stone_observations=[],
                           application_verifier=None, acid_candidate_events=[],
                           contact_closed_at=None, contact_close_reason=None, return_probe=None)
        self._residue_snapshot(self.active,self.acid_history.snapshot(ts))
        if self.active['reference'] is None and self.reference_job is None:
            candidates=[c for c in self.clear_history if ts-3<=c[0]<ts]
            if candidates:
                source_ts,source_frame,quad=candidates[-1]
                self.reference_owner=directory.name
                self.reference_job=self.pool.submit(RubReference,self.models,source_frame,source_ts,quad,snap['revision'])
        if jid is None:
            self.active['issues'].append('Jewel identity was missed or multiple jewels were under hands; no identity was guessed.')
        if self.model_error:
            self.active['issues'].append('Model preparation: '+self.model_error)
        self.emit('begin', jewel_id=jid, token=directory.name, timestamp=ts)
        self._phase('PICKUP', 'Following pickup; rubbing contact is not yet confirmed.', self.active)

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
                  acid_state=('APPLICATION LOCKED / MEASURING AREA' if a.get('acid_onset') is not None else
                              'CANDIDATE / RECORDING / NOT VERIFIED' if a.get('application_verifier') is not None and a['application_verifier'].pending else
                              'PREVIOUS ACID IGNORED; WAITING FOR FRESH' if 'IGNORED' in a.get('fresh_acid_status', '') else
                              'FRESH APPLICATION NOT CONFIRMED') if a and a.get('baseline_at') is not None else None,
                  residue_warning=a.get('residue_warning') if a else None,
                  **details)

    def _baseline(self, a, frame, ts):
        if a.get('baseline_at') is not None:
            return
        self._close_contact(a, ts, 'BASELINE_STARTED')
        a['baseline_at'] = a['mask_decision_at'] = ts
        a['recovery_cursor'] = a['returned']
        a['fresh_gate'] = FreshAcidGate(self.acid_history.mask,
            a['return_frame'] if a.get('return_frame') is not None else frame, self.pose.quad)
        a['application_verifier'] = None
        a['fresh_acid_status'] = 'FRESH_APPLICATION_NOT_CONFIRMED'
        a['acid_baseline_checked'] = False
        a['baseline_start_image'] = self._picture(a['directory'], 'baseline_start', frame)
        self.emit('baseline', jewel_id=a['jewel_id'], token=a['id'], timestamp=ts,
                  photo=self._stone_picture(frame, self.pose.quad), path=a['baseline_start_image'])
        self.emit('mask_decision', jewel_id=a['jewel_id'], token=a['id'], detected=a['mask'] is not None,
                  reason=a['mask_reason'], timestamp=ts,
                  contamination_possible=a.get('rub_contamination_possible', False))
        self._phase('BASELINE', 'Mask check finished. Baseline saved; waiting for acid without ending the session.', a)

    def _candidate_acid(self,a,observed,best,recovered=False):
        # Resolve a possible J2 pickup before assigning a later application to J1.
        if self.next_pickup is not None and observed>=self.next_pickup['start']:
            a['pending_acid']=(observed,best,recovered)
            return
        self._set_acid(a,observed,best,recovered)

    def _application_verifier(self, a):
        verifier = a.get('application_verifier')
        gate = a.get('fresh_gate')
        if verifier is None and gate is not None and gate.baseline is not None:
            verifier = a['application_verifier'] = AcidApplicationVerifier(gate.baseline, gate.old_mask)
        return verifier

    def _verify_application(self, a, frame, quad, timestamp, clear, geometry_known):
        if a['acid_onset'] is not None:
            return
        verifier = self._application_verifier(a)
        if verifier is None or not verifier.pending:
            return
        result = verifier.observe(frame, quad, timestamp, clear, geometry_known)
        a['acid_candidate_events'] = list(verifier.events)
        if result is not None:
            observed, best = result
            self._candidate_acid(a, observed, best)
        elif verifier.pending:
            a['fresh_acid_status'] = 'CANDIDATE_RECORDING_UNVERIFIED'
        else:
            a['fresh_acid_status'] = 'FRESH_APPLICATION_NOT_CONFIRMED'

    def _set_acid(self, a, observed, best, recovered=False):
        if a.get('acid_onset') is not None:
            return  # First qualifying application owns this cycle permanently.
        self._close_contact(a, observed, 'ACID_APPLICATION')
        a['acid_onset'], a['detector_bbox'] = observed, best['bbox']
        a['detector_confidence'], a['acid_recovered'] = best['confidence'], recovered
        a['acid_source_frame']=best.get('source_frame')
        if best.get('candidate_events') is not None:
            a['acid_candidate_events'] = best['candidate_events']
        for key in ('application_candidate_at', 'application_verified_at', 'application_timing_status',
                    'application_verification', 'clear_patch_frames'):
            a[key] = best.get(key)
        a['fresh_acid_status'] = 'FRESH_APPLICATION_DETECTED'
        a['fresh_acid_evidence'] = {key: best.get(key) for key in ('freshness', 'fresh_changed_fraction',
            'application_verification', 'clear_patch_frames')}
        quad = best.get('quad')
        if quad is None:
            quad = self._quad_at(a, observed)
        a['acid_source_quad'] = None if quad is None else np.asarray(quad).copy()
        gate = a.get('fresh_gate')
        if gate is not None and gate.baseline is not None and quad is not None:
            a['acid_area'] = AcidAreaMeasurement(gate.baseline, gate.old_mask, observed, best, quad)
        a['acid_area_requests'] = 0
        self.acid_history.remember([best], quad,baseline=gate.baseline if gate is not None else None)
        a.setdefault('stop_search', threading.Event()).set()
        mask_cutoff = best.get('application_candidate_at')
        if mask_cutoff is None:
            mask_cutoff = observed
        if a.get('mask') is not None and a['mask']['timestamp'] >= mask_cutoff:
            a['mask'], a['mask_status'] = None, 'UNCONFIRMED'
            a['mask_evidence_status']='REJECTED_ACID_OVERLAP'
            a['mask_reason'] = 'Mask frame overlaps possible acid application; searching earlier recorded frames.'
            self.emit('clear_mask', jewel_id=a['jewel_id'], token=a['id'])
        self.emit('acid_detection', jewel_id=a.get('jewel_id'), token=a['id'], detected=True,
                  bbox=best['bbox'], confidence=best['confidence'], timestamp=observed, recovered=recovered)
        self._phase('REACTION', 'Acid confirmed once. Recording the five-second reaction and measuring acid area.', a,
                    remaining=max(0., observed+5-self.last_ts))

    def _search_recording(self, a, start, end):
        checked = start
        entries=self.journal.between(start,end,fps=5)
        indices=np.unique(np.linspace(0,len(entries)-1,min(self.recovery_acid_frames,len(entries)),dtype=int))
        gate = a.get('fresh_gate')
        verifier = (AcidApplicationVerifier(gate.baseline, gate.old_mask)
                    if gate is not None and gate.baseline is not None else None)
        for index in indices:
            entry=entries[index]
            if a['stop_search'].is_set():
                break
            quad = self._quad_at(a, entry['timestamp'])
            if quad is None:
                continue
            frame = self.journal.read(entry)
            observed, _, found = self._detect(frame, entry['timestamp'], box_of(quad), a['id'])
            checked = observed
            found=[d for d in found if cv2.pointPolygonTest(np.asarray(quad,np.float32),
                ((d['bbox'][0]+d['bbox'][2])/2,(d['bbox'][1]+d['bbox'][3])/2),False)>=0]
            if gate is not None:
                best, _ = gate.check(frame, quad, found, observed,
                                     geometry_known=self._pose_known(a, observed))
                if best is not None and verifier is not None:
                    verifier.offer(frame, quad, best, observed)
                if verifier is not None:
                    result = verifier.observe(frame, quad, observed,
                        clear=not surface_covered(frame, box_of(quad)), geometry_known=self._pose_known(a, observed))
                    if result is not None:
                        supported, verified = result
                        verified['candidate_events'] = list(verifier.events)
                        return supported, verified, checked, gate.rejections
        return None, None, checked, a['fresh_gate'].rejections if a.get('fresh_gate') else []

    @staticmethod
    def _pose_known(a, timestamp):
        return bool(a.get('poses')) and min(abs(p[0]-timestamp) for p in a['poses']) <= .65

    def _acid_observation(self, observed, cycle_id, found, context):
        a = self.active
        if context is None:
            return  # A detection without its own source geometry cannot arm capture.
        frame, quad, clear, geometry_known = context[:4]
        if a is not None and a['id'] == cycle_id and a['acid_onset'] is not None:
            if a.get('reaction_end') is not None and observed > a['reaction_end']:
                return
            area = a.get('acid_area')
            if area is not None and area.observe(frame, quad, found, observed, clear, geometry_known):
                self.acid_history.remember_region(area.mask,area.baseline)
                self.emit('acid_area', jewel_id=a['jewel_id'], token=a['id'], detected=True,
                          bbox=region_box(area.mask, quad, frame.shape), timestamp=observed,
                          confidence=a['detector_confidence'], area_status='ESTIMATED_PERSISTENT_FOOTPRINT')
            return
        if quad is not None:
            found=[d for d in found if cv2.pointPolygonTest(np.asarray(quad,np.float32),
                ((d['bbox'][0]+d['bbox'][2])/2,(d['bbox'][1]+d['bbox'][3])/2),False)>=0]
        baseline_sample = len(context) > 4 and context[4]
        if baseline_sample and a is not None and a['id'] == cycle_id and a.get('fresh_gate') is not None:
            # Baseline predictions may include the new rubbed mark. They are
            # excluded from this cycle's trigger, but are not residue history.
            for detection in found:
                a['fresh_gate'].old_mask |= footprint(detection['bbox'],quad)
            a['baseline_acid_detections'] = found
            a['acid_baseline_checked'] = True
            return
        idle_sample = a is None or observed < (a.get('returned') or float('inf'))
        if idle_sample:
            if a is None or observed < a['started']:
                self.acid_history.observe_idle(frame, quad, found, observed, clear)
                state=self.acid_history.snapshot(observed)
                self.emit('residue',timestamp=observed,**state)
                if a and a.get('fresh_gate') is None:
                    a['previous_acid_mask'] = self.acid_history.mask.copy()
                    self._residue_snapshot(a,state)
            return
        if a['id'] != cycle_id or a.get('baseline_at') is None or a['acid_onset'] is not None:
            return
        gate = a.get('fresh_gate')
        if gate is None:
            return
        best, reason = gate.check(frame, quad, found, observed, geometry_known)
        a['fresh_acid_status'] = reason
        if best is not None:
            verifier = self._application_verifier(a)
            if verifier is not None and verifier.offer(frame, quad, best, observed):
                self.emit('acid_detection', jewel_id=a['jewel_id'], token=a['id'], detected=False,
                          provisional=True, acid_present=True, freshness='CANDIDATE_RECORDING_UNVERIFIED',
                          bbox=best['bbox'], confidence=best['confidence'], timestamp=observed)
            self._verify_application(a, frame, quad, observed, clear, geometry_known)
            return
        # Never declare cleaning from frames captured after this jewel's pickup
        # or rubbing. Residue checking belongs exclusively to clear-stone idle.
        a.setdefault('acid_checks', []).append(dict(timestamp=observed, detected=False,
                                                   acid_present=bool(found), reason=reason))
        a['acid_checks'] = a['acid_checks'][-200:]
        self.emit('acid_detection', jewel_id=a['jewel_id'], token=a['id'], detected=False,
                  acid_present=bool(found), freshness=reason, bbox=None, confidence=0., timestamp=observed)

    def _collect_jobs(self):
        if self.reference_job and self.reference_job.done():
            job, self.reference_job = self.reference_job, None
            owner,self.reference_owner=self.reference_owner,None
            try:
                ref = job.result()
                if ref.revision != self.revision:
                    return
                if self.active is None and ref.timestamp > self.last_capture_end:
                    self.reference = ref
                elif (self.active is not None and ref.timestamp < self.active['started'] and self.active['reference'] is None
                      and (owner==self.active['id'] or (owner is None and ref.timestamp>=self.active['started']-3))):
                    self.active['reference'] = ref
                else:
                    return
                ref.image_path = self._picture(self.directory/'references', f'before_stone_{ref.timestamp:.6f}',
                                               self._stone_picture(ref.frame, ref.quad))
                ref.frame_path = self._picture(self.directory/'references', f'before_frame_{ref.timestamp:.6f}', ref.frame)
                self.emit('reference', photo=self._stone_picture(ref.frame, ref.quad), path=ref.image_path,
                          timestamp=ref.timestamp, status=ref.before['status'])
                if self.active is None:
                    self._phase('BEFORE_READY','Before-stone photo saved; existing marks recorded. Automatic capture continues.')
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

    @staticmethod
    def _contact_overlay(photo,quad,region):
        picture=photo.copy()
        if region is not None:
            projected=project_mask(region,picture.shape,cv2.getPerspectiveTransform(np.asarray(quad,np.float32),DEST))
            contours,_=cv2.findContours(projected,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(picture,contours,-1,(255,220,0),2)
        return picture

    def _confirm_candidate(self, a, frame, ts):
        candidate = a.get('mask_candidate')
        if candidate is None or a['mask'] is not None or a['returned'] is None or a['acid_onset'] is not None:
            return
        state,reason=validate_mark(candidate,frame,self.pose.quad,ts,a.get('contact_region'),self.placement)
        a['mask_evidence_status'],a['mask_reason']=state,reason
        if state=='CHECKING':
            return
        if state!='VERIFIED':
            candidate['validation_state'],candidate['validation_reason']=state,reason
            a['unverified_mask' if state.startswith('UNVERIFIED') else 'rejected_mask']=candidate
            a['mask_candidate']=None
            a['mask_attempt_at']=ts
            return
        transform=cv2.getPerspectiveTransform(self.pose.quad.astype(np.float32),DEST)
        candidate['roi_mask']=project_mask(candidate['new_mask'],frame.shape,transform)
        candidate['photo']=draw_acid_roi(frame.copy(),candidate['roi_mask'])
        candidate['photo_timestamp']=ts
        candidate['photo_quad']=self.pose.quad.copy()
        a['mask'],a['mask_status']=candidate,'CONFIRMED_LIVE'
        self._close_contact(a, ts, 'NEW_MASK_CONFIRMED')
        a['mask_completed_at']=ts
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

    def _close_contact(self, a, timestamp, reason):
        if a.get('contact_closed_at') is not None:
            return
        a['contact_closed_at'],a['contact_close_reason']=timestamp,reason
        self.emit('contact_gate',token=a['id'],jewel_id=a['jewel_id'],open=False,
                  timestamp=timestamp,reason=reason)

    @staticmethod
    def _reliable_return(a, own, visible, timestamp):
        if own is None or a.get('identity_status') == 'MISSED_DURING_PROCESS':
            return visible  # Existing identity-missed placement policy stays explicit.
        seen=own.get('det_seen',-1e9)
        if not visible or not a['started']<=seen<=timestamp or timestamp-seen>.4:
            a['return_probe']=None
            return False
        probe=a.get('return_probe')
        if probe is None or timestamp-probe['last_frame']>.75:
            probe=a['return_probe']=dict(start=timestamp,last_seen=seen,count=1,last_frame=timestamp)
        else:
            if seen>probe['last_seen']:
                probe['last_seen']=seen;probe['count']+=1
            probe['last_frame']=timestamp
        return probe['count']>=2 and timestamp-probe['start']>=.12

    def _contact(self,a,contact,owner_on_bed=False):
        if (owner_on_bed or a.get('contact_closed_at') is not None
                or a.get('returned') is not None or a.get('mask') is not None
                or a.get('baseline_at') is not None):
            return
        if not contact or contact['timestamp']<a['started'] or (a['acid_onset'] is not None and contact['timestamp']>=a['acid_onset']):
            return
        if contact.get('cycle_token') is not None and contact['cycle_token']!=a['id']:
            return
        if a.get('last_contact_ts',-1)>=contact['timestamp']:
            return
        a['last_contact_ts']=contact['timestamp']
        quad=self._quad_at(a,contact['timestamp'])
        region=contact_region(contact,quad)
        was_confirmed=a['live_rub']
        if contact.get('confirmed') and not was_confirmed:
            a['contact_region']=region
        elif contact.get('confirmed') or not was_confirmed:
            a['contact_region']=region if a.get('contact_region') is None else a['contact_region']|region
        if contact.get('confirmed'):
            a['live_rub']=True
            a['last_rub']=contact['timestamp']
            a['rubbing_status']='CONFIRMED_LIVE'
            a['last_activity']=contact['timestamp']
        old=a.get('contact')
        if (old is None or (contact.get('confirmed') and not old.get('confirmed'))
                or (bool(contact.get('confirmed'))==bool(old.get('confirmed')) and contact.get('area',0)>old.get('area',0)*1.2)):
            a['contact']=dict(contact)
            frame=contact['frame']
            image=frame.copy()
            ox,oy=contact['origin']
            mask=contact['mask']
            image[oy:oy+mask.shape[0],ox:ox+mask.shape[1]][mask>0]=(0,215,255)
            self.emit('contact',token=a['id'],jewel_id=a['jewel_id'],timestamp=contact['timestamp'],
                      box=contact['box'],confirmed=bool(contact.get('confirmed')),
                      photo=self._stone_picture(image,quad if quad is not None else np.float32([
                          contact['stone'][:2],(contact['stone'][2],contact['stone'][1]),
                          contact['stone'][2:],(contact['stone'][0],contact['stone'][3])])) )

    def _inactivity(self,a,snap,stone,previous_ts):
        ts=snap['timestamp']
        if previous_ts>0 and ts-previous_ts>1:
            # A missing stream is unknown time, not observed inactivity.
            a['last_activity']+=ts-previous_ts
        if stone is not None and ts-a.get('last_activity_sample',-1e9)>=.15:
            a['last_activity_sample']=ts
            a['activity_frame'],motion=local_motion(a.get('activity_frame'),snap['frame'],stone,allow_broad=True)
            if motion is not None:
                a['last_activity']=ts
        return ts-a['last_activity']>=self.inactivity_seconds

    def _detect(self, frame, ts, stone, cycle_id):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = stone
        pad = .12*max(x2-x1, y2-y1)
        x1, y1 = max(0, int(x1-pad)), max(0, int(y1-pad))
        x2, y2 = min(w, int(x2+pad)), min(h, int(y2+pad))
        if x2 <= x1 or y2 <= y1:
            return ts, cycle_id, []
        with self.detector_lock:
            dets, _ = self.detector.predict(frame[y1:y2, x1:x2], .60)
        found = []
        for det in dets:
            box = tuple(int(v) for v in np.asarray(det.bbox)+[x1, y1, x1, y1])
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            if stone[0] <= cx <= stone[2] and stone[1] <= cy <= stone[3]:
                found.append(dict(bbox=box, confidence=det.confidence))
        return ts, cycle_id, found

    def _measure_area_job(self, frame, ts, quad, cycle_id, area):
        _, _, found = self._detect(frame, ts, box_of(quad), cycle_id)
        area.observe(frame, quad, found, ts)
        return ts, cycle_id, area

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
                        self._candidate_acid(a, observed, best, recovered=True)
                except Exception as exc:
                    a['issues'].append('Recorded acid check: '+str(exc))
        if self.area_job and self.area_job.done():
            job,self.area_job=self.area_job,None
            try:
                measured_at,cycle_id,area=job.result()
                a=self.active
                if (a is not None and a['id']==cycle_id and a['acid_onset']==area.onset
                        and (a.get('reaction_end') is None or measured_at<=a['reaction_end'])):
                    a['acid_area']=area
                    if np.any(area.mask):self.acid_history.remember_region(area.mask,area.baseline)
            except Exception as exc:
                if self.active:self.active['issues'].append('Acid-area measurement: '+str(exc))
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
        previous_ts=self.last_ts
        self.last_ts = ts
        if self.report_dirty:
            self.request_report()
        if self.revision != snap['revision']:
            if self.active:
                self._finish(previous_ts,'Tracking ROI/reset changed; old cycle finalized without reusing its evidence.')
                if self.mask_job is not None:
                    self.mask_job[0].cancel()
                    self.mask_job = None
            self.revision = snap['revision']
            self.pose, self.reference = StonePose(), None
            self.clear_history.clear()
            if self.reference_job is not None:
                self.reference_job.cancel()
                self.reference_job=None
                self.reference_owner=None
            self.recent_hand.clear()
            self.placement = snap.get('placement')
            self.placement_auto = self.placement is None
            self.last_reference = -1e9
        self._visibility(snap, ts)
        stone = snap.get('stone')
        if stone is None and self.pose.quad is not None:
            stone = box_of(self.pose.quad)
        clear = self.pose.update(frame, ts, stone)
        if clear and (not self.clear_history or ts-self.clear_history[-1][0]>=.2):
            self.clear_history.append((ts,frame,self.pose.quad.copy()))
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
        # Handoff requires repeated actual, fresh in-hand detections of J2.
        reliable=[tr for tr in eligible_tracks if tr.get('in_hand') and visibility(tr,ts,snap['bench_moving'])=='IN_HAND' and
                  0<=ts-tr['det_seen']<=.4 and (a is None or tr['id']!=a['jewel_id'])]
        if a and a['acid_onset'] is None and len(reliable)==1:
            tr=reliable[0]
            if self.next_pickup is None or self.next_pickup['id']!=tr['id'] or ts-self.next_pickup['last_frame']>.45:
                self.next_pickup=dict(id=tr['id'],start=ts,last_seen=tr['det_seen'],count=1,last_frame=ts)
            elif tr['det_seen']>self.next_pickup['last_seen']:
                self.next_pickup['last_seen']=tr['det_seen']
                self.next_pickup['count']+=1
            self.next_pickup['last_frame']=ts
            if self.next_pickup['count']>=2 and ts-self.next_pickup['start']>=self.pickup_debounce:
                boundary=self.next_pickup['start']
                self._finish(boundary,'Acid not confirmed before the next jewel was reliably picked up.',reason='NEXT_JEWEL_PICKUP')
                self._new_cycle(tr['id'],snap,boundary)
                self.active['pickup_image_timestamp']=ts
                a=self.active
                self.next_pickup=None
        else:
            self.next_pickup=None
        if a is not None and self.next_pickup is None and a.get('pending_acid') is not None:
            observed,best,recovered=a.pop('pending_acid')
            self._set_acid(a,observed,best,recovered)
        if a is None and snap['mode'] == 'LOCKED' and pickup:
            self._new_cycle(candidate, snap, ts)
            a = self.active
        if a is not None:
            if stone is not None:
                a['stone_observations'].append((ts, tuple(int(v) for v in stone)))
            if clear:
                a['poses'].append((ts, self.pose.quad.copy()))
            if a.get('baseline_at') is not None and a['acid_onset'] is None:
                self._verify_application(a,frame,self.pose.quad,ts,clear,
                    clear and self.pose.last_clear is not None and ts-self.pose.last_clear <= .65)
            if a['acid_onset'] is None and self._inactivity(a,snap,stone,previous_ts):
                self._finish(ts,'Acid not confirmed after genuine SOP inactivity.',reason='INACTIVITY_TIMEOUT')
                return
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
            self._contact(a,snap.get('contact'),owner_on_bed=returned)
            reliable_return=self._reliable_return(a,own,returned and clear,ts)
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
            application_pending = a.get('application_verifier') is not None and bool(a['application_verifier'].pending)
            if (not application_pending and a['acid_onset'] is None and a['mask'] is None and a.get('mask_candidate') is None
                    and self.mask_job is None and clear and not occupied
                    and ts-self.last_photo >= .5 and a['reference'] is not None):
                self.last_photo = ts
                self.mask_job = (self.pool.submit(a['reference'].after, self.models, frame, ts,
                                                self.pose.quad.copy()), a['id'])
            if a['returned'] is None and reliable_return and clear and inside_roi(self.pose.quad, self.placement):
                a['return_detected'] = a.get('return_detected', ts)
                if self.pose.stable_since is None or ts-self.pose.stable_since < .10:
                    return
                a['returned'], a['return_frame'] = a['return_detected'], frame
                self._close_contact(a,ts,'JEWEL_RETURNED')
                a['return_frame_timestamp']=ts
                a['decision_deadline'] = a['returned']+1.
                self._phase('MASK_CHECK', 'Jewel returned and stone inside ROI; checking the new rubbed mask.', a)
            if a['returned'] is not None:
                if clear and not occupied and not application_pending:
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
                    self._phase('REACTION', 'Acid confirmed once; recording reaction and measuring estimated acid area.', a,
                                remaining=max(0., a['acid_onset']+5-ts))
                    # Broad surface coverage after application indicates hand/cloth interference.
                    source_quad=a.get('acid_source_quad')
                    if source_quad is None:
                        source_quad=self._quad_at(a,a['acid_onset'])
                    area = a.get('acid_area')
                    coverage_box = a['detector_bbox']
                    if area is not None and np.any(area.mask):
                        coverage_box = region_box(area.mask,source_quad,frame.shape)
                        if clear:
                            self.emit('acid_area',jewel_id=a['jewel_id'],token=a['id'],detected=True,
                                bbox=region_box(area.mask,self.pose.quad,frame.shape),timestamp=ts,
                                confidence=a['detector_confidence'],area_status='ESTIMATED_PERSISTENT_FOOTPRINT')
                    if surface_covered(frame, stone) or reaction_region_covered(frame,coverage_box,
                            source_quad,self.pose.quad,clear):
                        a['coverage_since'] = a['coverage_since'] or ts
                    else:
                        a['coverage_since'] = None
                    if a['coverage_since'] and ts-a['coverage_since'] >= .3:
                        if a['reaction_end'] is None:
                            a['reaction_end'] = max(a['acid_onset'],a['coverage_since']-.001)
                            a['issues'].append('Reaction-region hand/cloth/glare check failed; usable duration is recorded separately.')
                    if ts >= a['acid_onset']+5:
                        self._finish(ts)
                elif a.get('baseline_at') is not None:
                    provisional = a.get('application_verifier') is not None and bool(a['application_verifier'].pending)
                    if ts-a['baseline_at'] >= .3:
                        self._phase('ACID_CANDIDATE' if provisional else 'WAIT_FOR_ACID',
                            'Recording possible application; checking stone-relative movement and a persistent clear patch.' if provisional else
                            ('Previous/baseline acid ignored; fresh application not confirmed. Session stays active.'
                             if 'IGNORED' in a.get('fresh_acid_status', '') else
                             'Fresh acid application not confirmed. Session stays active; checking live and recorded frames.'), a)
                    if (not provisional and a.get('acid_baseline_checked') and ts-a['baseline_at'] >= 2 and self.recovery_job is None and self.detector is not None
                            and ts-a['last_recovery'] >= 2):
                        a['last_recovery'] = ts
                        self.recovery_job = (self.finish_pool.submit(self._search_recording,
                            dict(a, poses=list(a['poses']), fresh_gate=a['fresh_gate'].fork()), a['recovery_cursor'], ts-.2), a['id'])
            else:
                if returned:
                    self._phase('WAIT_FOR_PLACEMENT', 'Jewel returned; waiting for a stable clear stone inside the test ROI.', a)
                else:
                    self._phase('RUBBING' if a['live_rub'] else 'PICKUP',
                                'Following the held jewel; waiting for return and stone placement.',a)
        remaining_jewels = snap['mode'] == 'ARMING' or any(not tr.get('acid_done') for tr in eligible_tracks)
        armed = self.active is not None and self.active.get('baseline_at') is not None and self.active['acid_onset'] is None
        idle = self.active is None and (remaining_jewels or np.any(self.acid_history.mask))
        measuring = (self.active is not None and self.active['acid_onset'] is not None
                     and self.active.get('acid_area') is not None
                     and clear and self.pose.stable_since is not None
                     and ts-self.pose.stable_since >= .1
                     and ts <= self.active['acid_onset']+5
                     and (self.active.get('reaction_end') is None or ts <= self.active['reaction_end'])
                     and self.active.get('acid_area_requests', 0) < 6)
        interval = .6 if measuring else .18 if armed else .5
        if (measuring and stone is not None and self.detector is not None
                and self.area_job is None and self.acid_job is None and ts-self.last_detection>=interval):
            a=self.active
            a['acid_area_requests']=a.get('acid_area_requests',0)+1
            self.last_detection=ts
            self.area_job=self.detector_pool.submit(self._measure_area_job,frame,ts,
                self.pose.quad.copy(),a['id'],a['acid_area'].fork())
        if ((armed or idle) and stone is not None and self.detector is not None and self.acid_job is None
                and self.area_job is None
                and ts-self.last_detection >= interval):
            self.last_detection = ts
            a = self.active
            quad = None if self.pose.quad is None else self.pose.quad.copy()
            baseline_sample = armed and not a.get('acid_baseline_checked', False)
            source, source_ts = frame, ts
            if baseline_sample:
                source = a['return_frame']
                source_ts = a.get('return_frame_timestamp',a['returned'])
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
            self.reference_owner=None
            self.reference_job = self.pool.submit(RubReference, self.models, frame, ts, self.pose.quad.copy(), self.revision)
        if idle and self.reference is None and self.reference_job is None and ts-self.last_notice >= 2:
            self.last_notice = ts
            self._phase('WAIT_FOR_STONE', 'Waiting for a clear stone to save the before image; recording continues.')

    def _finish(self, ts, issue=None, reason=None):
        a, self.active = self.active, None
        if a is None:
            return
        self._close_contact(a,ts,'CYCLE_FINISHED')
        a['finished'] = ts
        a.pop('pending_acid',None)
        a['close_reason']=reason or ('REACTION_COMPLETE' if a['acid_onset'] is not None else 'SESSION_STOP')
        self.next_pickup=None
        if self.mask_job is not None:
            self.mask_job[0].cancel()
            self.mask_job=None
        if self.recovery_job is not None:
            self.recovery_job[0].cancel()
            self.recovery_job=None
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
        # Keep a genuinely newer pre-pickup reference for the following jewel.
        if self.reference is a.get('reference'):
            self.reference = None
        complete = a['acid_onset'] is not None and ts >= a['acid_onset']+5
        self._phase('PROCESSING', 'Reaction capture finished; calculating the acid result.' if complete else
                    'Session stopped; saving incomplete test evidence.', a)
        self.emit('captured', jewel_id=a['jewel_id'], token=a['id'], capture_complete=complete)
        self._queue_finalization(a)

    @staticmethod
    def _quad_at(a, ts):
        poses = a['poses']
        return min(poses, key=lambda p: abs(p[0]-ts))[1] if poses else None

    def _queue_finalization(self,a):
        # At most two finalization snapshots retain camera-sized arrays in RAM.
        # Overload goes to a durable local spool, never an unbounded model queue.
        with self.final_lock:
            if self.final_pending>=2:
                state=dict(a)
                state.pop('stop_search',None)
                temp=self.deferred_dir/(a['id']+'.tmp')
                with temp.open('wb') as handle:pickle.dump(state,handle,protocol=5)
                os.replace(temp,self.deferred_dir/(a['id']+'.pkl'))
                return
            self.final_pending+=1
        self.finish_pool.submit(self._finalize_one,a)

    def _finalize_one(self,a):
        try:
            self._recover_and_save(a)
        finally:
            with self.final_lock:
                self.final_pending-=1
            self._schedule_deferred()

    def _schedule_deferred(self):
        # Only files written by this process's private session spool are loaded.
        with self.final_lock:
            paths=sorted(self.deferred_dir.glob('*.pkl'))
            if not paths or self.final_pending>=2:return
            path=paths[0]
            with path.open('rb') as handle:a=pickle.load(handle)
            path.unlink()
            a['stop_search']=threading.Event()
            self.final_pending+=1
        try:self.finish_pool.submit(self._finalize_one,a)
        except RuntimeError:
            # Explicit test/executor shutdown: finish the already loaded item.
            self._finalize_one(a)

    def drain_finalization(self):
        while True:
            self._schedule_deferred()
            with self.final_lock:
                pending=self.final_pending
            if not pending and not list(self.deferred_dir.glob('*.pkl')):return
            time.sleep(.01)

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
                'fresh_acid_status', 'fresh_acid_evidence', 'baseline_acid_detections', 'acid_jewel_association',
                'close_reason','mask_evidence_status','pickup_image_timestamp')
        keys += ('application_candidate_at','application_verified_at','application_timing_status',
                 'application_verification','clear_patch_frames','acid_candidate_events',
                 'contact_closed_at','contact_close_reason','previous_acid_state','residue_warning')
        record = {key: a.get(key) for key in keys}
        record.update(cycle_id=a['id'],revision=a.get('version'),prediction=None, acid_status='NOT_DETECTED', record_path=str(a['directory']/'manifest.json'),
                      detector_bbox=list(a['detector_bbox']) if a.get('detector_bbox') else None,
                      actual_label=None, schema='jewel_continuous_prediction_v1',
                      alignment_mode='tracked_corners_limited_motion_assumption',
                      decision_deadline=a.get('decision_deadline'), mask_completed_at=a.get('mask_completed_at'))
        record['rubbed_region_kind'] = vars(self.models).get('mark_evidence_kind',
            getattr(type(self.models), 'mark_evidence_kind', 'SEGMENTATION'))
        model_path = vars(self.models).get('rub_model_path')
        if model_path is not None:
            record['rubbed_model'] = str(model_path)
        record['capture_complete'] = a['acid_onset'] is not None and a['finished'] >= a['acid_onset']+4.99
        gate = a.get('fresh_gate')
        record['fresh_acid_rejections'] = list(gate.rejections) if gate else []
        return record

    def _recover_acid_area(self, a, onset, end):
        """Bounded finalizer work; never arms capture or runs on the camera loop."""
        area = a.get('acid_area')
        gate = a.get('fresh_gate')
        quad = self._quad_at(a, onset)
        if area is None and gate is not None and gate.baseline is not None and quad is not None:
            area = a['acid_area'] = AcidAreaMeasurement(gate.baseline, gate.old_mask,
                onset, dict(bbox=a['detector_bbox']), quad)
        if area is None or np.any(area.mask) or self.detector is None:
            return area
        # Replay with a fresh measurement object so older recorded timestamps
        # cannot be rejected by a partially sampled live object's last_ts.
        area = AcidAreaMeasurement(area.baseline, area.old_mask,
            onset, dict(bbox=a['detector_bbox']), quad)
        entries = self.journal.between(onset, min(end, onset+5), fps=3)
        indices = np.unique(np.linspace(0, len(entries)-1, min(8, len(entries)), dtype=int))
        for index in indices:
            entry = entries[index]
            ts = entry['timestamp']
            current = self._quad_at(a, ts)
            if current is None or not self._pose_known(a, ts):
                continue
            frame = self.journal.read(entry)
            if surface_covered(frame, box_of(current)):
                continue
            try:
                _, _, found = self._detect(frame, ts, box_of(current), a['id'])
            except Exception as exc:
                a['issues'].append('Acid-area recovery unavailable: '+str(exc))
                break
            area.observe(frame, current, found, ts)
        a['acid_area'] = area
        return area

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
                a['acid_source_frame']=best.get('source_frame')
                for key in ('application_candidate_at','application_verified_at','application_timing_status',
                            'application_verification','clear_patch_frames'):
                    a[key] = best.get(key)
                if best.get('candidate_events') is not None:
                    a['acid_candidate_events'] = best['candidate_events']
                a['fresh_acid_status'] = 'FRESH_APPLICATION_DETECTED'
                a['fresh_acid_evidence'] = {key: best.get(key) for key in ('freshness', 'fresh_changed_fraction',
                    'application_verification','clear_patch_frames')}
        onset = a['acid_onset']
        mask_cutoff = a.get('application_candidate_at')
        if mask_cutoff is None:
            mask_cutoff = onset
        if onset is not None and a.get('mask') is not None and a['mask']['timestamp'] >= mask_cutoff:
            a['mask'],a['mask_status']=None,'UNCONFIRMED'
            a['mask_evidence_status']='REJECTED_ACID_OVERLAP'
            a['mask_reason']='Mask frame overlaps possible acid application; searching earlier recorded frames.'
        if a.get('application_timing_status') == 'ONSET_OBSCURED_USING_FIRST_CLEAR_PATCH':
            a['issues'].append('Application onset was obscured or moving; reaction timing starts at the first supported clear patch, not an unverified tool detection.')
        verifier = a.get('application_verifier')
        if onset is None and verifier is not None and verifier.pending:
            a['fresh_acid_status'] = 'FRESH_APPLICATION_NOT_CONFIRMED'
            reasons = sorted({c['reason'] for c in verifier.pending})
            a['issues'].append('Acid candidates remained unverified: '+', '.join(reasons)+'. Applicator/hand visibility and patch persistence are visual checks only.')
        contact=recover_contact(self.journal,a,self.rub_observer,self._picture)
        recovered_region=contact.pop('_contact_region',None)
        if recovered_region is not None and not a['live_rub']:
            a['contact_region']=recovered_region if a.get('contact_region') is None else a['contact_region']|recovered_region
        if contact['rubbing_verified']:
            a['rubbing_status']='CONFIRMED_LIVE' if a['live_rub'] else 'RUBBING_RECOVERED_FROM_CLIP'
        # Mark recovery obeys the same persistence and contact constraints as live.
        if a['mask'] is None and a['reference'] is not None:
            until=mask_cutoff-.05 if onset is not None else end
            entries=self.journal.between(a['returned'] or start,until,fps=5)
            indices=np.unique(np.linspace(0,len(entries)-1,min(4,len(entries)),dtype=int))
            candidates=[c for c in (a.get('unverified_mask'),a.get('rejected_mask')) if c is not None and c['timestamp']<until]
            for index in indices:
                entry=entries[index]
                try:
                    result=a['reference'].after(self.models,self.journal.read(entry),entry['timestamp'],self._quad_at(a,entry['timestamp']))
                    if result['status']=='NEW_RUB':candidates.append(result)
                    else:a['mask_reason']=evidence_reason(result['message'])
                except Exception as exc:a['mask_reason']=evidence_reason(exc)
            for candidate in candidates:
                candidate['clear_frames']=[]
                later=[e for e in entries if e['timestamp']>=candidate['timestamp']][:8]
                for entry in later:
                    quad=self._quad_at(a,entry['timestamp'])
                    if quad is None or not self._pose_known(a,entry['timestamp']):continue
                    state,reason=validate_mark(candidate,self.journal.read(entry),quad,entry['timestamp'],
                                               a.get('contact_region'),a.get('placement'))
                    a['mask_evidence_status'],a['mask_reason']=state,reason
                    if state=='VERIFIED':
                        a['mask'],a['mask_status']=candidate,'RECOVERED_FROM_CLIP'
                        a['mask_completed_at']=entry['timestamp']
                        break
                    if state.startswith('REJECTED'):
                        candidate['validation_state'],candidate['validation_reason']=state,reason
                        a['rejected_mask']=candidate
                        break
                    if state.startswith('UNVERIFIED'):
                        candidate['validation_state'],candidate['validation_reason']=state,reason
                        a['unverified_mask']=candidate
                if a['mask'] is not None:break
        if a['mask'] is None and a.get('mask_evidence_status')=='CHECKING':
            if a.get('unverified_mask') is not None:
                a['mask_evidence_status']='UNVERIFIED_CONTACT_LOCATION'
                a['mask_reason']='Persistent new-looking mark has no verified jewellery-contact location.'
            elif a.get('rejected_mask') is not None:
                a['mask_evidence_status']=a['rejected_mask'].get('validation_state','REJECTED')
                a['mask_reason']=a['rejected_mask'].get('validation_reason','Rejected mark candidate.')
            else:
                a['mask_evidence_status']='NOT_ESTABLISHED'
                a['mask_reason']='Insufficient independent clear frames before acid/session end.'
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
            record['mask_timestamp']=a['mask'].get('photo_timestamp',a['mask']['timestamp'])
            quad=a['mask'].get('photo_quad',a['mask']['quad'])
            picture=a['mask']['photo'].copy()
            record['mask_image']=self._picture(a['directory'],'new_rub',self._stone_picture(picture,quad))
            cv2.imwrite(str(a['directory']/'new_rub_mask.png'), a['mask']['roi_mask'])
        else:
            a['issues'].append('New rubbed mask could not be confirmed: '+a['mask_reason'])
        record.update(contact)
        if contact.get('rubbing_return_precedes_contact'):
            a['issues'].append('Tracking reported return before the saved live contact; the rubbing evidence interval includes that later pre-acid contact.')
        if not contact['rubbing_verified']:
            a['issues'].append(contact['rubbing_contact_reason'])
        candidate=a.get('rejected_mask') or a.get('unverified_mask')
        if candidate is not None and a['mask'] is None:
            photo=candidate.get('photo')
            if photo is not None:
                quad=candidate.get('photo_quad',candidate['quad'])
                picture=photo.copy()
                record['candidate_mask_timestamp']=candidate.get('photo_timestamp',candidate['timestamp'])
                record['candidate_mask_image']=self._picture(a['directory'],'unverified_mark_candidate',
                    self._stone_picture(picture,quad))
        contact_end=contact['rubbing_evidence_interval'][1]
        record['rubbing_clip'] = self.journal.clip(start-1, max(a['returned'] or end, contact_end), a['directory']/'pickup_rubbing_return.avi')
        if onset is not None:
            reaction_end = min(onset+5, end, a['reaction_end'] if a['reaction_end'] is not None else float('inf'))
            record['acid_status'] = ('RECOVERED_FROM_CLIP' if a.get('acid_recovered') else
                'VERIFIED_LIVE_PERSISTENT_PATCH' if a.get('application_verification') else 'DETECTED_LIVE_ONE_HIT')
            record['acid_clip'] = self.journal.clip(onset-2, min(onset+5, end), a['directory']/'acid_2s_before_5s_after.avi')
            h, w = (a['return_frame'] if a['return_frame'] is not None else a['reference'].frame).shape[:2]
            area = self._recover_acid_area(a, onset, reaction_end)
            measured = area is not None and np.any(area.mask)
            onset_quad = self._quad_at(a, onset)
            record['acid_area_status'] = 'ESTIMATED_PERSISTENT_VISUAL_ACID_AREA' if measured else 'AREA_UNAVAILABLE'
            record['acid_area_samples'] = area.samples if area is not None else 0
            record['acid_area_rejections'] = dict(area.rejections) if area is not None else {}
            record['acid_area_updated_at'] = area.updated_at if measured else None
            area_box = region_box(area.mask, onset_quad, (h,w)) if measured and onset_quad is not None else None
            record['acid_area_bbox'] = list(area_box) if area_box else None
            x1, y1, x2, y2 = area_box or a['detector_bbox']
            px, py = .45*(x2-x1), .45*(y2-y1)
            box = (max(0, int(x1-px)), max(0, int(y1-py)), min(w, int(x2+px)), min(h, int(y2+py)))
            rub = cv2.resize(a['mask']['new_mask'], (256,256), interpolation=cv2.INTER_NEAREST) if a['mask'] is not None else None
            record.update(compare_area(area.mask if measured else None, rub))
            if not measured:
                a['issues'].append('Application confirmed, but the acid area could not be measured in successive clear frames; location comparison unavailable. Reaction crop uses the initial detector box.')
            elif record['acid_target_status'] in ('ACID_AT_DIFFERENT_LOCATION','ACID_PARTIALLY_OVERLAPS_NEW_RUB'):
                a['issues'].append('Estimated persistent acid footprint '+('does not overlap' if record['acid_target_status']=='ACID_AT_DIFFERENT_LOCATION' else 'partially overlaps')+' the new rubbed mask; this is not a chemical boundary measurement.')
            if measured:
                record['acid_area_mask'] = self._picture(a['directory'], 'acid_area_stone_relative', area.mask)
            onset_quad = self._quad_at(a, onset)
            shift = np.float32([[1, 0, -box[0]], [0, 1, -box[1]], [0, 0, 1]])
            reaction_times=[]
            rejected_reaction=[]
            def frames(lo, hi, verify_reaction=False):
                result = []
                for entry in self.journal.between(lo, hi, fps=5):
                    frame = self.journal.read(entry)
                    quad = self._quad_at(a, entry['timestamp'])
                    if quad is not None and onset_quad is not None:
                        transform = shift @ cv2.getPerspectiveTransform(quad.astype(np.float32), onset_quad.astype(np.float32))
                        crop = cv2.warpPerspective(frame, transform, (box[2]-box[0], box[3]-box[1]))
                    else:
                        crop = frame[box[1]:box[3], box[0]:box[2]]
                    if verify_reaction:
                        known=self._pose_known(a,entry['timestamp'])
                        covered=(not crop.size or reaction_covered(crop,(0,0,crop.shape[1],crop.shape[0])))
                        if not known or covered:
                            rejected_reaction.append(dict(timestamp=entry['timestamp'],
                                reason='REACTION_GEOMETRY_UNCERTAIN' if not known else 'REACTION_REGION_COVERED'))
                            continue
                        reaction_times.append(entry['timestamp'])
                    if crop.size:
                        result.append(crop.copy())
                return result
            baseline = frames(onset-2, onset-.001)
            transition = frames(onset, min(onset+.5, reaction_end))
            observation = frames(onset, reaction_end,verify_reaction=True)
            record['frame_counts'] = dict(baseline=len(baseline), application=len(transition), reaction=len(observation))
            record['reaction_bbox'] = list(box)
            record['reaction_crop_alignment'] = 'stone_corners_to_acid_onset'
            record['acid_window_mode'] = '2s_before_first_detection_5s_after_source_timestamp'
            times=reaction_times
            record['reaction_rejected_frames']=rejected_reaction
            gaps=[b-a for a,b in zip(times,times[1:])]
            complete=(not rejected_reaction and len(times)>=10 and times[0]-onset<=.35 and reaction_end-times[-1]<=.35
                      and max(gaps,default=0)<=.6 and reaction_end>=onset+4.9)
            record['reaction_coverage_status']='COMPLETE' if complete else 'PARTIAL'
            typical=float(np.median(gaps)) if gaps else .2
            missing=sum(max(0,g-typical) for g in gaps if g>.6)
            span=times[-1]-times[0]+typical-missing if times else 0
            if times:
                span-=sum(typical for e in rejected_reaction if times[0]<e['timestamp']<times[-1])
            record['usable_reaction_seconds']=max(0.,min(reaction_end-onset,span))
            if not complete:
                a['issues'].append('Reaction recording incomplete: insufficient duration/frames or a source-frame gap; no material prediction generated.')
            if rejected_reaction:
                a['issues'].append(f'{len(rejected_reaction)} saved reaction frames were excluded for coverage or uncertain stone geometry; see reaction_rejected_frames in the manifest.')
            if reaction_end < onset+4.9:
                a['issues'].append(f'Only {max(0., reaction_end-onset):.2f} seconds of reaction evidence available before coverage/session end.')
            if a['returned'] is None or onset-2 < a['returned']:
                a['issues'].append('The two-second baseline includes jewel return/hand or unsettled footage.')
            if baseline:
                record['baseline_image'] = self._picture(a['directory'], 'baseline', baseline[-1])
            if observation:
                record['reaction_first_image']=self._picture(a['directory'],'reaction_first',observation[0])
                if times:
                    record['reaction_first_timestamp']=times[0]
                    record['reaction_last_timestamp']=times[-1]
                record['acid_image'] = self._picture(a['directory'], 'acid_reaction', observation[-1])
            detection_frames = self.journal.between(onset-.15,onset+.15)
            detection=a.get('acid_source_frame')
            detection_ts=onset
            if detection is not None:
                record['acid_confirmation_source_frame']=self._picture(a['directory'],'acid_confirmation_source',detection)
            if measured:
                detection=area.source_frame
                detection_ts=area.updated_at
            if detection is None and detection_frames:
                entry=min(detection_frames,key=lambda e:abs(e['timestamp']-onset))
                detection=self.journal.read(entry)
                detection_ts=entry['timestamp']
            if detection is not None:
                detection=detection.copy()
                placement_quad=area.source_quad if measured else self._quad_at(a,detection_ts)
                record['acid_source_frame']=self._picture(a['directory'],'acid_application_source',detection)
                record['acid_placement_timestamp']=detection_ts
                comparison = detection.copy()
                if a['mask'] is not None:
                    current_quad = placement_quad
                    rub_mask = project_mask(a['mask']['new_mask'], comparison.shape,
                        cv2.getPerspectiveTransform(current_quad.astype(np.float32), DEST))
                    contours, _ = cv2.findContours(rub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(comparison, contours, -1, (255,0,255), 2)
                if measured:
                    area_quad=placement_quad
                    placement_box=region_box(area.mask,area_quad,comparison.shape)
                    area_image=project_mask(area.mask,comparison.shape,
                        cv2.getPerspectiveTransform(area_quad.astype(np.float32),ACID_DEST))
                    contours,_=cv2.findContours(area_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(comparison,contours,-1,(255,255,0),2)
                else:
                    placement_box=None
                crop = self._stone_picture(comparison, placement_quad)
                legend = np.full((75, max(500, crop.shape[1]), 3), 245, np.uint8)
                cv2.putText(legend, 'MAGENTA: new rub | '+('CYAN: estimated visible acid' if measured else 'ACID AREA UNAVAILABLE'), (8,25), cv2.FONT_HERSHEY_SIMPLEX, .45, (40,40,40), 1)
                cv2.putText(legend, record['acid_target_status'], (8,53), cv2.FONT_HERSHEY_SIMPLEX, .5, (40,40,40), 1)
                canvas = np.full((crop.shape[0]+75, legend.shape[1], 3), 245, np.uint8)
                canvas[:crop.shape[0], :crop.shape[1]] = crop
                canvas[crop.shape[0]:] = legend
                record['acid_rub_comparison_image'] = self._picture(a['directory'], 'rub_and_acid_locations', canvas)
                if placement_box is not None:
                    dx1,dy1,dx2,dy2=placement_box
                    cv2.rectangle(detection, (dx1, dy1), (dx2, dy2), (255, 255, 0), 2)
                    cv2.putText(detection, 'ESTIMATED VISIBLE ACID AREA',
                        (dx1, max(20, dy1-8)), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,0), 2)
                record['acid_detection_image'] = self._picture(a['directory'], 'acid_detected', detection)
            if baseline and transition and observation and self.classifier is not None and record['reaction_coverage_status']=='COMPLETE':
                # Preserve the exact supported runtime output, even with process warnings.
                record['prediction'] = self.classifier.predict(baseline, transition, observation)
            else:
                a['issues'].append('Acid model not run: a complete usable reaction, baseline/application frames, or model were unavailable.')
            for folder, images in [('baseline', baseline), ('application', transition), ('reaction', observation)]:
                for index, image in enumerate(images):
                    self._picture(a['directory']/folder, f'{index:04d}', image)
        else:
            record['acid_clip'] = self.journal.clip((a['returned'] or start)-2, end, a['directory']/'acid_search.avi')
            a['issues'].append('Acid application was not verified in live or recorded footage; no acid model result was fabricated.')
            a['issues'].append('Fresh application not confirmed; old/baseline acid and uncertain observations did not start a reaction timer.')
        candidate_times = [e['candidate_at'] for e in a.get('acid_candidate_events', []) if e.get('candidate_at') is not None]
        if candidate_times:
            first_candidate = min(candidate_times)
            record['first_acid_candidate_at'] = first_candidate
            record['acid_candidate_clip'] = (self.journal.clip(first_candidate-2, min(end,onset+5),
                a['directory']/'provisional_acid_candidate.avi') if onset is not None and first_candidate < onset-2
                else record['acid_clip'])
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
        if (self.active is None and record.get('revision')==self.revision
                and record.get('cycle_id')==self.last_cycle_id):
            from session_evidence import prediction_text
            self._phase('DONE' if record.get('prediction') and record.get('capture_complete') else 'INCOMPLETE',
                        prediction_text(record.get('prediction'))+'; audit saved.')
        try:
            self.request_report()
        except RuntimeError:
            self.report()  # Tests/explicit shutdown may already have drained the report executor.
