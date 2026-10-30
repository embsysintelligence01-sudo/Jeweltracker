"""Verify a classifier ROI from full-frame detector boxes, never area masks."""
import cv2
import numpy as np

from acid_freshness import normalized, footprint
from acid_test.flow import bbox_iou
from acid_application import visible_patch


def extract_crop(frame, box, source_quad, current_quad):
    """Map only the output crop; never resize/mask the classifier's pixels."""
    if source_quad is not None and current_quad is not None:
        transform=cv2.getPerspectiveTransform(np.asarray(current_quad,np.float32),
                                            np.asarray(source_quad,np.float32))
        if not np.allclose(transform,np.eye(3),atol=1e-5):
            shift=np.float32([[1,0,-box[0]],[0,1,-box[1]],[0,0,1]])
            return cv2.warpPerspective(frame,shift@transform,(box[2]-box[0],box[3]-box[1]))
    return frame[box[1]:box[3],box[0]:box[2]].copy()


class ClassificationCrop:
    def __init__(self, baseline, target=None):
        self.baseline=baseline
        self.target=target
        self.previous=[]
        self.best=None
        self.checks=[]

    def observe(self, frame, quad, found, timestamp, clear=True):
        if not clear or quad is None or self.baseline is None:
            self.previous=[]
            return
        image=normalized(frame,quad)
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        skin=(hsv[:,:,0]<=14)&(hsv[:,:,1]>=45)&(hsv[:,:,2]>80)
        valid=[]
        for detection in found[:8]:
            region=footprint(detection['bbox'],quad)>0
            if region.sum()<5 or region.mean()>.3:continue
            # Verify the current patch/box, not its freshness versus baseline.
            # Otherwise an accepted application can be blocked again at cropping.
            patch, _ = visible_patch(image,region)
            if patch.sum()<5 or skin[region].mean()>.2:continue
            # Wrong-location acid still gets its own material observation;
            # comparison with the rubbed mark is a separate audit outcome.
            ys,xs=np.where(region)
            local=(int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max()))
            candidate=dict(timestamp=timestamp,bbox=tuple(detection['bbox']),local=local,
                quad=np.asarray(quad,np.float32).copy(),confidence=float(detection['confidence']))
            valid.append(candidate)
            for previous in self.previous:
                if not .03<=timestamp-previous['timestamp']<=1.25:continue
                if bbox_iou(local,previous['local'])<.45:continue
                score=(local[2]-local[0])*(local[3]-local[1])*candidate['confidence']
                if self.best is None or score>self.best['score']:
                    self.best=dict(candidate,score=score,status='VERIFIED_FULL_FRAME_DETECTOR_PAIR',
                                   first_timestamp=previous['timestamp'])
        # Isolated model misses do not erase a preceding clear observation.
        self.previous=[p for p in self.previous if timestamp-p['timestamp']<=1.25]+valid
        self.previous=self.previous[-16:]
        self.checks.append(dict(timestamp=timestamp,detected=len(found),supported=len(valid)))
