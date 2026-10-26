"""Headless color/tracking SOP extracted from ph_paper_auto_sop.py.

Keep the supplied fresh-yellow and sustained-pink decision thresholds.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

PAPER_CLASS_ID = 0
CONFIDENCE = 0.35
FRESH_FRAMES = 8
REACTION_FRAMES = 2
# A pipette or hand can briefly hide the strip. Keep its baseline and progress
# through this many missed detections; only a longer absence starts a new test.
MISSING_GRACE_FRAMES = 8
PINK_THRESHOLD = 0.10
PINK_DELTA_MIN = 0.15
MIN_CONNECTED_PINK = 0.15
MIN_YELLOW_DURING_REACTION = 0.20
MIN_COLOR_COVERAGE = 0.45
MAX_DARK_FRACTION = 0.12
OCCLUSION_YELLOW_DROP = 0.25


YELLOW_LO = (17, 70, 90)
YELLOW_HI = (34, 255, 245)
# The reacted paper samples from this camera are magenta (mostly H=171..179).
# Excluding the red 0-degree wrap rejects the light-red nail sample, while the
# higher saturation floor rejects pale skin highlights.
PINK_LO = (168, 80, 70)
PINK_HI = (179, 255, 245)


def rect_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    aa = max(0,a[2]-a[0]) * max(0,a[3]-a[1])
    bb = max(0,b[2]-b[0]) * max(0,b[3]-b[1])
    return intersection / max(1, aa+bb-intersection)


def plausible_match(a, b):
    if a is None:
        return True
    aw, ah = a[2]-a[0], a[3]-a[1]
    bw, bh = b[2]-b[0], b[3]-b[1]
    aa, bb = aw*ah, bw*bh
    if min(aa,bb) == 0 or max(aa,bb)/min(aa,bb) > 3.0:
        return False
    ac = ((a[0]+a[2])/2,(a[1]+a[3])/2)
    bc = ((b[0]+b[2])/2,(b[1]+b[3])/2)
    dist = math.dist(ac,bc)
    return rect_iou(a,b)>0.02 or dist < max(80.0,min(200.0, 2.0*math.hypot(aw,ah)))


def all_paper_detections(model, image, class_id, confidence):
    """Return oriented pH-paper detections and their polygon masks."""
    return model.detect(image, class_id, confidence)


def bbox_strip_band(crop, yellow, pink):
    """Approximate elongated paper band in bbox-only mode.

    Infer direction/width from yellow pixels, then extend band across box.
    This heuristic helps suppress hands but is NOT equivalent to segmentation.
    """
    hh,ww=crop.shape[:2]
    y_points=cv2.findNonZero(yellow.astype(np.uint8))
    if y_points is None or len(y_points)<max(20,int(0.008*ww*hh)):
        return np.ones((hh,ww),dtype=bool)  # preserve ability to flag pre-used
    rect=cv2.minAreaRect(y_points)
    (cx,cy),(rw,rh),angle=rect
    # If yellow region is square/round, orientation cannot be recovered.
    major=max(rw,rh); minor=min(rw,rh)
    if major<12 or minor<3 or major / max(1,minor) < 1.3:
        return np.ones((hh,ww),dtype=bool)
    quad=cv2.boxPoints(rect)
    # Long axis vector of the oriented rectangle.
    edges=[quad[(i+1)%4]-quad[i] for i in range(4)]
    v=max(edges,key=lambda e:float(np.dot(e,e)))
    v=v/max(1e-6,float(np.linalg.norm(v)))
    per=np.array([-v[1],v[0]])
    yy,xx=np.indices((hh,ww),dtype=np.float32)
    perpendicular=np.abs((xx-cx)*per[0]+(yy-cy)*per[1])
    # Slight margin for imperfect yellow boundary and colored tip.
    band=perpendicular<=max(5,minor*0.66)
    return band


def analyze_paper(crop, shape_mask=None):
    """Measure color against the complete OBB paper area, not colored pixels only."""
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    yellow=cv2.inRange(hsv,YELLOW_LO,YELLOW_HI)>0
    pink=cv2.inRange(hsv,PINK_LO,PINK_HI)>0
    if shape_mask is not None:
        inside=shape_mask>0
    else:
        inside=bbox_strip_band(crop,yellow,pink)
    _,_,v=cv2.split(hsv)
    yellow &= inside
    pink &= inside
    dark=inside & (v<70)
    inside_count=int(np.count_nonzero(inside))
    yellow_count=int(np.count_nonzero(yellow))
    pink_count=int(np.count_nonzero(pink))
    color_count=int(np.count_nonzero(yellow | pink))
    sufficient=inside_count>=max(80,int(0.02*crop.shape[0]*crop.shape[1]))

    # Full paper area is the denominator. An occluding black/skin region can no
    # longer make a small remaining pink patch look like a large percentage.
    yellow_fraction=yellow_count/max(1,inside_count)
    pink_fraction=pink_count/max(1,inside_count)
    dark_fraction=float(np.count_nonzero(dark))/max(1,inside_count)
    coverage=color_count/max(1,inside_count)

    # A real drop normally creates one coherent patch. Scattered noise does not
    # pass merely because many unrelated pixels add up to the threshold.
    component_count,_,component_stats,_=cv2.connectedComponentsWithStats(
        pink.astype(np.uint8),8
    )
    largest_pink_pixels=(
        int(component_stats[1:,cv2.CC_STAT_AREA].max()) if component_count>1 else 0
    )
    largest_pink_fraction=largest_pink_pixels/max(1,inside_count)
    lab=cv2.cvtColor(crop.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    yellow_lab=np.median(lab[yellow],axis=0) if np.any(yellow) else None
    pink_lab=np.median(lab[pink],axis=0) if np.any(pink) else None
    overlay=np.zeros_like(crop)
    overlay[yellow]=(0,255,255)
    overlay[pink]=(255,0,255)
    overlay[dark]=(255,0,0)
    return dict(sufficient=sufficient, yellow=yellow_fraction,pink=pink_fraction,
                dark=dark_fraction,coverage=coverage,pink_pixels=pink_count,
                largest_pink_pixels=largest_pink_pixels,
                largest_pink=largest_pink_fraction,yellow_pixels=yellow_count,
                inside_pixels=inside_count,
                yellow_lab=yellow_lab,pink_lab=pink_lab,overlay=overlay)


def delta_e_76(x,y):
    if x is None or y is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(x)-np.asarray(y)))


class AutoSOP:
    def __init__(self, fresh_frames=FRESH_FRAMES,
                 reaction_frames=REACTION_FRAMES,
                 pink_threshold=PINK_THRESHOLD):
        self.fresh_frames=fresh_frames
        self.reaction_frames=reaction_frames
        self.pink_threshold=pink_threshold
        self.reset()

    def reset(self):
        self.state='WAIT_FRESH'
        self.fresh_hits=0
        self.reaction_hits=0
        self.missing=0
        self.prev_bbox=None
        self.baseline_pink=0.0
        self.baseline_yellow=0.0
        self.baseline_lab=None
        self.baseline_image=None
        self.reaction_start_time=None
        self.end_detail=None
        self.saved=False
        self.last_diagnostic='Show unused yellow paper'

    def update(self, detection, metrics, timestamp):
        """Returns True once for validated transition (can trigger voice)."""
        if (detection is not None and self.prev_bbox is not None and
                not plausible_match(self.prev_bbox,detection['bbox'])):
            # A sudden box change is a tracking gap, not proof of a new strip.
            detection=None
        if detection is None:
            if self.prev_bbox is None:
                return False
            self.missing+=1
            if self.missing>MISSING_GRACE_FRAMES:
                self.reset()
                self.last_diagnostic='Paper absent too long; show unused yellow paper'
            else:
                self.last_diagnostic=(
                    f'Paper briefly hidden ({self.missing}/{MISSING_GRACE_FRAMES}); '
                    'keeping the same strip'
                )
            return False

        bbox=detection['bbox']
        self.prev_bbox=bbox
        self.missing=0

        if not metrics['sufficient']:
            self.last_diagnostic='Analysis paused: paper area is too small'
            self.fresh_hits=0 if self.state=='WAIT_FRESH' else self.fresh_hits
            self.reaction_hits=0
            return False
        if metrics['dark']>MAX_DARK_FRACTION:
            self.last_diagnostic=(
                f"Analysis paused: dark/black obstruction {metrics['dark']:.0%}"
            )
            self.fresh_hits=0 if self.state=='WAIT_FRESH' else self.fresh_hits
            self.reaction_hits=0
            return False
        if metrics['coverage']<MIN_COLOR_COVERAGE:
            self.last_diagnostic='Analysis paused: paper is covered or colors are unclear'
            self.fresh_hits=0 if self.state=='WAIT_FRESH' else self.fresh_hits
            self.reaction_hits=0
            return False

        yellow=metrics['yellow']; pink=metrics['pink']
        if self.state=='WAIT_FRESH':
            if yellow>=.70 and pink<=.05:
                self.fresh_hits+=1
                self.last_diagnostic=f'Fresh yellow {self.fresh_hits}/{self.fresh_frames}'
                if self.fresh_hits>=self.fresh_frames:
                    self.state='WATCH_REACTION'
                    self.baseline_pink=pink
                    self.baseline_yellow=yellow
                    self.baseline_lab=metrics['yellow_lab']
                    self.baseline_image=detection['crop'].copy()
                    self.reaction_start_time=timestamp
                    self.last_diagnostic='Fresh paper accepted; add acid while keeping it visible'
            else:
                self.fresh_hits=0
                self.last_diagnostic='Pre-colored/unclear paper: need unused yellow'

        elif self.state=='WATCH_REACTION':
            yellow_drop=self.baseline_yellow-yellow
            if yellow_drop>OCCLUSION_YELLOW_DROP and pink<self.pink_threshold:
                self.reaction_hits=0
                self.last_diagnostic='Analysis paused: hand/nail may be covering the paper'
                return False
            d_e=delta_e_76(self.baseline_lab,metrics['pink_lab'])
            passes=(pink>=self.pink_threshold and
                    pink-self.baseline_pink>=PINK_DELTA_MIN and
                    yellow>=MIN_YELLOW_DURING_REACTION and
                    metrics['pink_pixels']>=20 and d_e>=20)
            passes=passes and metrics['largest_pink']>=MIN_CONNECTED_PINK
            self.reaction_hits=self.reaction_hits+1 if passes else 0
            self.last_diagnostic=(
                f"Pink {pink:.0%} Yellow {yellow:.0%} Dark {metrics['dark']:.0%} "
                f'dE76 {d_e:.1f} '
                f'({self.reaction_hits}/{self.reaction_frames})'
            )
            if self.reaction_hits>=self.reaction_frames:
                self.state='VERIFIED'
                self.last_diagnostic='ACIDIC REACTION VERIFIED (not acid identity)'
                self.end_detail=dict(pink_fraction=pink,yellow_fraction=yellow,
                                     dark_fraction=metrics['dark'],
                                     largest_pink_fraction=metrics['largest_pink'],
                                     delta_e_76=d_e,
                                     elapsed_seconds=timestamp-self.reaction_start_time,
                                     detector_mode=detection['mode'])
                return True
        return False

