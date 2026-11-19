"""Bounded recorded-frame analysis. No camera, UI events or live model ownership."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np

from acid_application import AcidApplicationVerifier
from acid_area import region_box
from acid_freshness import normalized, changed_pixels, footprint
from new_rub_detection.flow import RubCheck
from new_rub_detection.geometry import DEST, project_mask
from rub_preparation import clear_surface
from rubbing_recovery import recover_contact
from sop_evidence import validate_mark, contact_match


def save(directory, name, image):
    path = Path(directory)/f'{name}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    if image is None or not image.size or not cv2.imwrite(str(path), image):
        raise OSError('Could not write backup evidence: '+str(path))
    return str(path)


class RecordedSource:
    """Exact JPEG journal entries or video indices with authoritative timestamps."""
    def __init__(self, entries):
        self.entries = sorted(entries, key=lambda e:e['timestamp'])
        self.decode_seconds = 0.
        self.decoded = 0
        self.videos = {}

    def between(self, start, end, fps=None):
        result, last = [], -float('inf')
        for entry in self.entries:
            if start <= entry['timestamp'] <= end and (fps is None or entry['timestamp']-last >= 1/fps-.001):
                result.append(entry)
                last = entry['timestamp']
        return result

    def read(self, entry):
        start = time.perf_counter()
        try:
            if entry.get('video'):
                path = entry['video']
                if path not in self.videos:
                    self.videos[path] = cv2.VideoCapture(path)
                video = self.videos[path]
                video.set(cv2.CAP_PROP_POS_FRAMES, entry['frame_index'])
                ok, frame = video.read()
                if not ok:
                    raise OSError('Saved video frame unavailable')
            else:
                frame = cv2.imread(entry['file'])
                if frame is None:
                    raise OSError('Recorded frame unavailable: '+entry['file'])
            self.decoded += 1
            return frame
        finally:
            self.decode_seconds += time.perf_counter()-start

    def close(self):
        for video in self.videos.values():
            video.release()


def components(mask, limit=2):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    indices = sorted(range(1, count), key=lambda i:stats[i, cv2.CC_STAT_AREA], reverse=True)
    return [(labels == i).astype(np.uint8)*255 for i in indices[:limit]
            if 30 <= stats[i, cv2.CC_STAT_AREA] <= .15*mask.size]


class AnalysisEngine:
    def __init__(self, config, refiner=None, models=None, contact_observer=None):
        self.config = config
        self.refiner = refiner
        self.models = models
        self.contact_observer = contact_observer
        self.refiner_error = None
        self.gold = None
        self.acid_detector = None

    def _models(self):
        if self.models is None:
            from rub_preparation import RubPhotoModels
            self.models = RubPhotoModels()
        return self.models

    def _refine(self, image, seed, diagnostics):
        if self.refiner is None and self.refiner_error is None:
            try:
                from edgesam_backend import EdgeSAM
                self.refiner = EdgeSAM(self.config['encoder'], self.config['decoder'],
                    self.config['threads'], self.config['embedding_cache_size'])
            except Exception as exc:
                self.refiner_error = str(exc)
        if self.refiner_error:
            diagnostics['edgesam_unavailable'] = self.refiner_error
            return None
        return self.refiner.refine(image, seed, self.config['box_padding'], self.config['min_score'])

    def _observer(self, frame, stone):
        if self.contact_observer is not None:
            return self.contact_observer(frame, stone)
        if self.gold is None:
            import torch
            torch.set_num_threads(self.config['threads'])
            from int8_models import OpenVINOYolo, JEWEL_MODEL
            model = OpenVINOYolo(JEWEL_MODEL)
            # Reuse the production contact/colour checks without its shared
            # scheduler or model instance. The child has its own CPU affinity.
            def predict(image, **kwargs):
                kwargs.pop('_live_job', None)
                return model.predict(image, **kwargs)
            self.gold = SimpleNamespace(model=SimpleNamespace(predict=predict))
        from jewel_tracker_5 import Worker
        return Worker.recorded_rub_evidence(self.gold, frame, stone)

    def analyze(self, job):
        start = time.perf_counter()
        source = RecordedSource(job['entries'])
        result = dict(status='REVIEW_REQUIRED', source='EXISTING_RECOVERY_RETAINED',
                      reason='Required recorded evidence remains unverified', timings={}, diagnostics={})
        previous_stats = dict(getattr(self.refiner, 'stats', {}))
        try:
            self._analyze(job, source, result)
        except Exception as exc:
            result['diagnostics']['analysis_error'] = str(exc)
        finally:
            source.close()
        stats = getattr(self.refiner, 'stats', {})
        result['timings'].update(video_decoding_seconds=source.decode_seconds,
            decoded_frames=source.decoded, total_backup_seconds=time.perf_counter()-start,
            edgesam_inference_seconds=sum(stats.get(key, 0)-previous_stats.get(key, 0)
                for key in ('encoder_seconds', 'decoder_seconds')),
            edgesam_encoder_calls=stats.get('encoder_calls',0)-previous_stats.get('encoder_calls',0),
            edgesam_decoder_calls=stats.get('decoder_calls',0)-previous_stats.get('decoder_calls',0),
            embedding_cache_hits=stats.get('embedding_cache_hits',0)-previous_stats.get('embedding_cache_hits',0))
        result['models'] = getattr(self.refiner, 'models', {})
        result['timings'].setdefault('rubbing_verification_seconds', 0.)
        result['timings'].setdefault('acid_verification_seconds', 0.)
        return result

    def _analyze(self, job, source, result):
        config, directory = self.config, Path(job['directory'])
        directory.mkdir(parents=True, exist_ok=True)
        diagnostics = result['diagnostics']
        # Refuse cross-cycle or unordered replay metadata; never extend into the
        # following jewel to manufacture a complete reaction window.
        entries = source.between(job['started'], job['finished'])
        poses = job.get('poses', [])
        if not poses or len(entries) < 3:
            result['reason'] = 'Insufficient timestamped frames or known stone poses'
            return
        reference = job.get('before')
        if reference is None or reference['timestamp'] >= job['started']:
            result['reason'] = 'No clear pre-operation stone reference'
            return
        before = source.read(reference)
        reference_quad = np.float32(reference['quad'])
        before_stone = cv2.fillConvexPoly(np.zeros(before.shape[:2],np.uint8), reference_quad.astype(np.int32),255)
        clear_surface(before, before_stone)
        before_image = cv2.warpPerspective(before, cv2.getPerspectiveTransform(reference_quad,DEST),(512,512))
        contact = cv2.imread(job['contact_region'],0) if job.get('contact_region') else None
        contact_ok = job.get('rubbing_verified', False)
        rubbing_start = time.perf_counter()
        if not contact_ok:
            stone_observations = [(p['timestamp'], tuple(np.r_[np.min(p['quad'],axis=0),np.max(p['quad'],axis=0)]))
                                  for p in (job.get('stone_observations') or poses)]
            cycle = dict(started=job['started'], finished=job['finished'], returned=job.get('returned'),
                acid_onset=job.get('acid_onset'), contact=None, directory=directory,
                overlap_trigger_required=False, recovery_contact_frames=config['contact_frames'],
                stone_observations=stone_observations)
            recovered = recover_contact(source, cycle, self._observer, save)
            contact = recovered.pop('_contact_region', contact)
            contact_ok = recovered['rubbing_verified']
            diagnostics['recorded_contact_status'] = recovered['rubbing_contact_status']
            diagnostics['recorded_contact_errors'] = recovered.get('rubbing_recovery_errors', [])
            if contact_ok:
                result['contact'] = recovered
        # Bounded clear-frame selection uses recorded poses and local image
        # quality. Samples that fail visibility cannot contribute persistence.
        cache, rejected = {}, Counter()
        def load(index):
            if index in cache:
                return cache[index]
            if len(cache) >= config['max_frames']:
                return None
            entry = entries[index]
            timestamp = entry['timestamp']
            pose = min(poses, key=lambda p:abs(p['timestamp']-timestamp))
            if abs(pose['timestamp']-timestamp) > config['max_pose_age']:
                rejected['unknown_pose'] += 1
                cache[index] = None
                return None
            quad = np.float32(pose['quad'])
            if quad.shape != (4,2) or not np.isfinite(quad).all() or abs(cv2.contourArea(quad))<80:
                cache[index] = None
                return None
            frame = source.read(entry)
            stone = cv2.fillConvexPoly(np.zeros(frame.shape[:2],np.uint8),quad.astype(np.int32),255)
            try:
                clear_surface(frame,stone)
            except RuntimeError:
                rejected['hand_coverage'] += 1
                cache[index] = None
                return None
            matrix = cv2.getPerspectiveTransform(quad, DEST)
            image = cv2.warpPerspective(frame,matrix,(512,512))
            hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            if ((hsv[:,:,2]>150)&(hsv[:,:,1]<40))[10:-10,10:-10].mean()>.35:
                rejected['broad_glare_or_cloth'] += 1
                cache[index] = None
                return None
            if cv2.Laplacian(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),cv2.CV_32F).var()<config['min_sharpness']:
                rejected['blur'] += 1
                cache[index] = None
                return None
            # Same recorded corner continuity as live; large unexplained
            # orientation/scale changes remain review cases, never ECC guesses.
            if cv2.contourArea(quad,oriented=True)*cv2.contourArea(reference_quad,oriented=True)<=0:
                cache[index] = None
                return None
            item = dict(frame=frame, quad=quad, image=image, timestamp=timestamp, matrix=matrix,
                        source_frame=save(directory,f'backup_source_{index:05d}',frame))
            cache[index] = item
            return item

        initial = set(np.linspace(0,len(entries)-1,config['initial_frames'],dtype=int).tolist())
        anchor = job.get('returned')
        if anchor is not None:
            initial.add(min(range(len(entries)),key=lambda i:abs(entries[i]['timestamp']-anchor)))
        initial = sorted(initial)[:config['initial_frames']]
        diagnostics['initial_frame_indices'] = initial
        rub = job.get('rub')
        rub_item = None
        if rub:
            index = min(range(len(entries)),key=lambda i:abs(entries[i]['timestamp']-rub['timestamp']))
            rub_item = load(index)
        cutoff = job.get('application_candidate_at') or job.get('acid_onset') or job['finished']
        # Model candidates are tried before any EdgeSAM request. A single
        # persistent mark with independent contact support is required.
        todo = list(initial)
        todo += [int(i) for i in np.linspace(0,len(entries)-1,8,dtype=int) if i not in initial]
        for index in todo:
            if rub_item is not None or entries[index]['timestamp'] >= cutoff:
                continue
            item = load(index)
            if item is None:
                continue
            change = changed_pixels(before_image,item['image'])
            gray_delta = cv2.cvtColor(item['image'],cv2.COLOR_BGR2GRAY).astype(float)-cv2.cvtColor(before_image,cv2.COLOR_BGR2GRAY)
            gray_delta -= np.median(gray_delta[16:-16,16:-16])
            hsv = cv2.cvtColor(item['image'],cv2.COLOR_BGR2HSV)
            seed = change & (gray_delta>10) & (hsv[:,:,1]<90)
            seed[:10]=seed[-10:]=False
            seed[:,:10]=seed[:,-10:]=False
            seeds = components(seed)
            if not seeds or seed.mean()>.15 or contact is None:
                continue
            candidate = None
            try:
                check = RubCheck()
                models = self._models()
                check.before(before,before_stone,models.marks(before),reference_quad)
                model_result = check.after(item['frame'],
                    cv2.fillConvexPoly(np.zeros(item['frame'].shape[:2],np.uint8),item['quad'].astype(np.int32),255),
                    models.marks(item['frame']),item['quad'], frame_to_reference=item['matrix'])
                if model_result['status']=='NEW_RUB':
                    candidate = model_result['new_mask']
            except Exception as exc:
                diagnostics['rub_model_error'] = str(exc)
            origin, score = 'RECORDED_MODEL_RECOVERY', 0.
            if candidate is None:
                for seed_mask in seeds:
                    if (contact_match(seed_mask,contact) or 0)<.25:
                        continue
                    refined = self._refine(item['image'],seed_mask,diagnostics)
                    if refined:
                        candidate,score = refined['mask'],refined['score']
                        origin = 'EDGESAM_BACKUP'
                        diagnostics['rub_prompts'] = {k:v for k,v in refined.items() if k!='mask'}
                        break
            if candidate is None:
                continue
            candidate = dict(new_mask=candidate, timestamp=item['timestamp'],frame=item['frame'],
                             frame_to_reference=item['matrix'], clear_frames=[])
            state = 'CHECKING'
            for later in range(index,min(index+5,len(entries))):
                if entries[later]['timestamp']>=cutoff:
                    break
                current = load(later)
                if current is None:
                    continue
                state,reason = validate_mark(candidate,current['frame'],current['quad'],current['timestamp'],
                                             contact,job.get('placement'))
                if state=='VERIFIED':
                    if not contact_ok:
                        diagnostics['mark_only'] = 'Persistent mark, but rubbing motion remains unverified'
                        break
                    result['rub'] = dict(mask=save(directory,'backup_rub_mask',candidate['new_mask']),
                        timestamp=item['timestamp'],verified_at=current['timestamp'],quad=item['quad'].tolist(),
                        source_frame=item['source_frame'],source=origin,score=score)
                    rub_item = item
                    break
                if state.startswith('REJECTED'):
                    diagnostics['rub_rejection'] = reason
                    break
        result['timings']['rubbing_verification_seconds'] = time.perf_counter()-rubbing_start
        acid_start = time.perf_counter()
        onset = job.get('acid_onset')
        baseline_path = job.get('acid_baseline')
        baseline = cv2.imread(baseline_path) if baseline_path else None
        baseline_ts = (rub_item['timestamp'] if rub_item is not None else None)
        if baseline is None and rub_item is not None:
            baseline = normalized(rub_item['frame'],rub_item['quad'])
            baseline_path = save(directory,'backup_verified_preacid_baseline',baseline)
        if baseline is not None and (rub_item is not None or onset is not None):
            acid_indices = sorted(set(initial + [i for i in range(len(entries))
                if onset is not None and onset<=entries[i]['timestamp']<=onset+.6]))
            # If sparse sampling missed a stage, expand only after initial
            # candidates failed. The global cache bounds all clear-frame work.
            if rub_item is not None:
                acid_indices += [int(i) for i in np.linspace(0,len(entries)-1,8,dtype=int)
                                 if i not in acid_indices]
            for index in acid_indices:
                if baseline_ts is not None and entries[index]['timestamp']<=baseline_ts+.1:
                    continue
                if onset is not None and not onset<=entries[index]['timestamp']<=onset+5:
                    continue
                item = load(index)
                if item is None:
                    continue
                image = normalized(item['frame'],item['quad'])
                changed = changed_pixels(baseline,image)
                if changed.mean()>.25:
                    continue
                hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                seeds = components(changed & ~((hsv[:,:,1]<40)&(hsv[:,:,2]>150)))
                for seed_mask in seeds:
                    # OpenCV visual candidates precede EdgeSAM. An acid region
                    # without existing detector support always needs refinement.
                    refined = None
                    try:
                        if self.acid_detector is None:
                            from acid_test.onnx_detector import OnnxAcidDetector
                            self.acid_detector = OnnxAcidDetector(Path(__file__).with_name('acid.onnx'))
                        detections,_ = self.acid_detector.predict(item['frame'],.30)
                        for detection in detections:
                            support = footprint(detection.bbox,item['quad'])>0
                            if np.count_nonzero(support & (seed_mask>0))/max(1,np.count_nonzero(seed_mask))>=.65:
                                refined = dict(mask=seed_mask, score=float(detection.confidence), source='RECORDED_MODEL_RECOVERY')
                                break
                    except Exception as exc:
                        diagnostics['acid_model_error'] = str(exc)
                    if refined is None:
                        refined = self._refine(image,seed_mask,diagnostics)
                    if refined is None:
                        continue
                    origin = refined.get('source','EDGESAM_BACKUP')
                    region = refined['mask']
                    if onset is None and baseline_ts is not None:
                        # Narrow the observed onset bracket without decoding the
                        # whole clip. Unknown/occluded samples cannot move it back.
                        lo = next((i for i,e in enumerate(entries) if e['timestamp']>baseline_ts),index)
                        hi = index
                        while lo < hi:
                            mid = (lo+hi)//2
                            earlier = load(mid)
                            if earlier is None:
                                break
                            delta = changed_pixels(baseline,normalized(earlier['frame'],earlier['quad']))
                            if np.count_nonzero(delta & (region>0))/max(1,np.count_nonzero(region))>=.65:
                                hi = mid
                            else:
                                lo = mid+1
                        if lo==hi:
                            earlier = load(lo)
                            if earlier is not None:
                                index,item = lo,earlier
                    bbox = region_box(region,item['quad'],item['frame'].shape)
                    if bbox is None:
                        continue
                    verifier = AcidApplicationVerifier(baseline,np.zeros((256,256),np.uint8), min_clear_frames=3)
                    verifier.offer(item['frame'],item['quad'],dict(bbox=bbox,confidence=refined['score']),item['timestamp'])
                    verified = None
                    accepted = []
                    for later in range(index,min(index+5,len(entries))):
                        current = load(later)
                        if current is None:
                            continue
                        # Confirm the EdgeSAM boundary on independent recorded
                        # images, not repeated prompts against the same pixels.
                        current_image = normalized(current['frame'],current['quad'])
                        current_changed = changed_pixels(baseline,current_image)
                        support = (region>0)&current_changed
                        if support.sum()/max(1,np.count_nonzero(region))<.65:
                            continue
                        if verified is None:
                            verified = verifier.observe(current['frame'],current['quad'],current['timestamp'])
                        accepted.append(current)
                        if verified and len(accepted)>=3:
                            break
                    diagnostics['acid_candidate_events'] = verifier.events
                    if not verified or len(accepted)<3:
                        continue
                    supported, best = verified
                    first = min(accepted,key=lambda p:abs(p['timestamp']-supported))
                    last = accepted[-1]
                    # Refinement must remain a new patch, not its unchanged
                    # surroundings or an old mark retained by the prompt box.
                    if np.count_nonzero((region>0)&changed)/max(1,np.count_nonzero(region))<.5:
                        continue
                    result['acid'] = dict(mask=save(directory,'backup_acid_mask',region),
                        source=origin,score=refined['score'],timestamp=supported,
                        verified_at=last['timestamp'], clear_frames=len(accepted),
                        baseline=baseline_path,baseline_timestamp=baseline_ts,
                        quad=first['quad'].tolist(),bbox=list(best['bbox']),source_frame=first['source_frame'],
                        verified_source_frame=last['source_frame'],verified_quad=last['quad'].tolist())
                    diagnostics['acid_prompts'] = {k:v for k,v in refined.items() if k!='mask'}
                    break
                if result.get('acid'):
                    break
        result['timings']['acid_verification_seconds'] = time.perf_counter()-acid_start
        diagnostics.update(selected_frame_indices=[int(i) for i in sorted(cache)], rejected_clear_frames=dict(rejected),
                           clear_frames=sum(item is not None for item in cache.values()))
        if result.get('rub') or result.get('acid'):
            result['source'] = ('EDGESAM_BACKUP' if result.get('acid',{}).get('source')=='EDGESAM_BACKUP' or result.get('rub',{}).get('source')=='EDGESAM_BACKUP'
                                else 'RECORDED_MODEL_RECOVERY')
        if (rub_item is not None and contact_ok and (result.get('acid') or (onset is not None and not job.get('need_acid_area')))):
            result.update(status='VERIFIED',reason='Recorded contact, new mark and persistent acid region verified')
