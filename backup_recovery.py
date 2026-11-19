"""Recorded-only backup transport and host integration; never called by a camera.

The existing finalizer owns ordering/publication. A persistent subprocess owns
all additional models, CPU limits and embedding cache. No live model is sent to
the subprocess, and a timeout terminates native inference before publication.
"""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
DEFAULTS = dict(enabled=True, encoder='weights/edgesam/edge_sam_3x_encoder.onnx',
                decoder='weights/edgesam/edge_sam_3x_decoder.onnx', threads=1,
                nice=15, max_cpus=1, timeout_seconds=45., initial_frames=4,
                max_frames=16, box_padding=.20, min_score=.85,
                min_sharpness=8., max_pose_age=.4, contact_frames=12,
                max_queue_seconds=120., embedding_cache_size=4)


def configuration(values=None):
    result = dict(DEFAULTS, **(values or {}))
    if 'JEWEL_EDGESAM_ENABLED' in os.environ:
        result['enabled'] = os.environ['JEWEL_EDGESAM_ENABLED'].lower() in ('1', 'true', 'yes')
    for key in ('encoder', 'decoder'):
        result[key] = str((ROOT / result[key]).resolve())
    result['threads'] = max(1, min(2, int(result['threads'])))
    result['initial_frames'] = max(2, min(4, int(result['initial_frames'])))
    result['max_frames'] = max(result['initial_frames'], min(32, int(result['max_frames'])))
    for key in ('timeout_seconds', 'max_queue_seconds'):
        result[key] = max(.1, float(result[key]))
    result['box_padding'] = max(0., min(1., float(result['box_padding'])))
    return result


class BackupWorker:
    def __init__(self, config):
        self.config = config
        self.process = None

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdin.close()
            process.stdout.close()

    def analyze(self, request, directory):
        directory = Path(directory)
        request_file = directory / 'backup_request.json'
        request_file.write_text(json.dumps(request), encoding='utf-8')
        deadline = time.monotonic() + self.config['timeout_seconds']
        try:
            if self.process is None or self.process.poll() is not None:
                self.close()
                env = dict(os.environ, OMP_NUM_THREADS=str(self.config['threads']),
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', QT_QPA_PLATFORM='offscreen')
                # Keep diagnostics out of the line-oriented response pipe.
                with (directory / 'backup_worker.log').open('a') as errors:
                    self.process = subprocess.Popen([sys.executable, '-u', str(ROOT/'backup_worker.py')],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, env=env, text=True, bufsize=1)
            self.process.stdin.write(json.dumps(dict(request=str(request_file), config=self.config))+'\n')
            self.process.stdin.flush()
            ready, _, _ = select.select([self.process.stdout], [], [], max(0., deadline-time.monotonic()))
            if not ready:
                raise TimeoutError('EdgeSAM backup exceeded its worker deadline')
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('Recorded backup worker exited; see backup_worker.log')
            response = json.loads(line)
            if response.get('error'):
                raise RuntimeError(response['error'])
            return response
        except Exception as exc:
            self.close()
            return dict(status='REVIEW_REQUIRED', reason=str(exc), source='EXISTING_RECOVERY_RETAINED',
                        timed_out=isinstance(exc, TimeoutError), timings={})


def required(attempt):
    area = attempt.get('acid_area')
    return (attempt.get('mask') is None or attempt.get('mask_evidence_status') != 'VERIFIED'
            or attempt.get('acid_onset') is None or area is None or not area.mask.any())


def recover(monitor, attempt, contact):
    """Only invoked after the original OpenCV/model-based finalizer recovery."""
    import cv2
    import numpy as np
    from session_evidence import atomic_json
    config = monitor.backup_config
    if not config['enabled'] or not required(attempt):
        return
    directory = attempt['directory']
    queued = attempt.get('backup_queued_at', time.monotonic())
    if time.monotonic()-queued > config['max_queue_seconds']:
        response = dict(status='REVIEW_REQUIRED', reason='Recorded recovery queue deadline exceeded', timings={})
    else:
        reference = attempt.get('reference')
        before = None
        if reference is not None:
            before = dict(file=monitor._picture(directory, 'backup_before_source', reference.frame),
                          timestamp=reference.timestamp, quad=reference.quad.tolist())
        mask = attempt.get('mask')
        rub = None
        if mask is not None:
            rub = dict(file=monitor._picture(directory, 'backup_existing_rub_mask', mask['new_mask']),
                       timestamp=mask['timestamp'])
        region = attempt.get('contact_region')
        region_path = monitor._picture(directory, 'backup_contact_region', region) if region is not None else None
        gate = attempt.get('fresh_gate')
        baseline = (monitor._picture(directory, 'backup_acid_baseline', gate.baseline)
                    if gate is not None and gate.baseline is not None else None)
        entries = monitor.journal.between(attempt['started'], attempt['finished'], fps=5)
        # Absolute frame references are exact recorder samples, not reconstructed
        # video FPS times; the replay reader also supports timestamp sidecars.
        entries = [dict(entry, file=str(monitor.directory/entry['file'])) for entry in entries]
        request = dict(cycle_id=attempt['id'], directory=str(directory), entries=entries, before=before,
            stone_observations=[dict(timestamp=ts,quad=[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]]) for ts,b in attempt.get('stone_observations',[])],
            poses=[dict(timestamp=ts, quad=np.asarray(q).tolist()) for ts, q in attempt['poses']],
            started=attempt['started'], finished=attempt['finished'], returned=attempt.get('returned'),
            acid_onset=attempt.get('acid_onset'), application_candidate_at=attempt.get('application_candidate_at'),
            acid_baseline=baseline, contact_region=region_path, rubbing_verified=contact.get('rubbing_verified', False),
            contact_samples=contact.get('rubbing_contact_samples', []), rub=rub,
            placement=None if attempt.get('placement') is None else np.asarray(attempt['placement']).tolist(),
            need_acid_area=attempt.get('acid_area') is None or not attempt['acid_area'].mask.any())
        if monitor.backup_worker is None:
            monitor.backup_worker = BackupWorker(config)
        response = monitor.backup_worker.analyze(request, directory)
    attempt['backup'] = response
    attempt['evidence_source'] = response.get('source', 'EXISTING_RECOVERY_RETAINED')
    for key,value in response.get('timings',{}).items():
        attempt['performance_timings'][key]=attempt['performance_timings'].get(key,0)+value
    if response.get('contact'):
        contact.update(response['contact'])
        attempt['rubbing_status'] = 'RUBBING_RECOVERED_FROM_CLIP'
    recovered_rub = response.get('rub')
    if recovered_rub and attempt.get('mask') is None:
        frame = cv2.imread(recovered_rub['source_frame'])
        new_mask = cv2.imread(recovered_rub['mask'], cv2.IMREAD_GRAYSCALE)
        quad = np.float32(recovered_rub['quad'])
        from new_rub_detection.geometry import DEST, project_mask
        transform = cv2.getPerspectiveTransform(quad, DEST)
        roi = project_mask(new_mask, frame.shape, transform)
        photo = frame.copy()
        photo[roi > 0] = cv2.addWeighted(photo, .4, np.full_like(photo, (255, 0, 255)), .6, 0)[roi > 0]
        attempt['mask'] = dict(new_mask=new_mask, roi_mask=roi, frame=frame, photo=photo,
            quad=quad, timestamp=recovered_rub['timestamp'], frame_to_reference=transform,
            photo_timestamp=recovered_rub['timestamp'], photo_quad=quad,
            rub_display_kind='EDGESAM_SEGMENTATION' if recovered_rub['source']=='EDGESAM_BACKUP' else 'RECORDED_MODEL_SEGMENTATION')
        attempt.update(mask_status='RECOVERED_FROM_CLIP', mask_evidence_status='VERIFIED',
            mask_reason='Recorded before/after change persisted and matched model-supported jewellery contact.',
            mask_completed_at=recovered_rub['verified_at'])
    recovered_acid = response.get('acid')
    if recovered_acid:
        from acid_area import AcidAreaMeasurement
        from acid_freshness import normalized
        frame = cv2.imread(recovered_acid['source_frame'])
        quad = np.float32(recovered_acid['quad'])
        area_mask = cv2.imread(recovered_acid['mask'], cv2.IMREAD_GRAYSCALE)
        baseline = cv2.imread(recovered_acid['baseline'])
        if attempt.get('acid_onset') is None:
            onset = recovered_acid['timestamp']
            attempt.update(acid_onset=onset, detector_bbox=tuple(recovered_acid['bbox']),
                acid_recovered=True, acid_source_frame=frame, acid_source_quad=quad,
                application_candidate_at=onset, application_verified_at=recovered_acid['verified_at'],
                application_timing_status='RECORDED_FIRST_CLEAR_PATCH_ONSET',
                application_verification='RECORDED_NEW_PERSISTENT_PATCH_WITH_'+recovered_acid['source'],
                clear_patch_frames=recovered_acid['clear_frames'], baseline_at=recovered_acid['baseline_timestamp'],
                return_frame=frame)
            # This source region has passed the same application verifier as the
            # live path. Existing pad_bbox(.45) and temporal preprocessing own all
            # subsequent classification; EdgeSAM padding never enters that path.
            attempt['confirmed_classification_region'] = dict(bbox=recovered_acid['bbox'],
                quad=quad.tolist(), timestamp=onset, verification=attempt['application_verification'],
                confidence=recovered_acid.get('score', 0.))
        area = AcidAreaMeasurement(baseline, np.zeros((256,256),np.uint8),
            attempt['acid_onset'], dict(bbox=attempt['detector_bbox']), quad)
        area.mask = area_mask
        area.updated_at = recovered_acid['verified_at']
        area.source_frame = cv2.imread(recovered_acid['verified_source_frame'])
        area.source_quad = np.float32(recovered_acid['verified_quad'])
        area.samples = recovered_acid['clear_frames']
        attempt['acid_area'] = area
    atomic_json(directory/'backup_status.json', response)
