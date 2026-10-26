"""Automatic PyQt5 pH-paper test using OpenVINO OBB and HSV/LAB color checks.

Run with: .venv-openvino/bin/python ph_paper_auto_sop.py
Check CPU inference without a camera: add --check-model

The camera continuously detects the same pH paper. A stable fresh-yellow
baseline is accepted automatically, after which the user keeps the paper in
view and places the acid/sample on it. A sustained pink change is then detected
automatically. No location target or manual stage-confirmation button is used.
"""
from __future__ import annotations

import argparse
import os
import queue
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QRect, QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from openvino_obb import OpenVINOOBB

# Linux OpenCV wheels set Qt paths for their bundled Qt. Use PyQt5's runtime
# for this application's widgets, including when both wheels are installed.
if sys.platform.startswith('linux'):
    for variable in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
        if 'cv2' in os.environ.get(variable, ''):
            os.environ.pop(variable, None)


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "acidph_openvino" / "acidph.xml"
EVIDENCE_DIR = HERE / "ph_sop_evidence"
DEBUG_DIR = HERE / "ph_debug"
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


def say_async(words):
    def runner():
        try:
            import pyttsx3
            engine=pyttsx3.init()
            engine.say(words)
            engine.runAndWait()
        except Exception as exc:
            print('[voice unavailable]',exc)
    threading.Thread(target=runner,daemon=True).start()


def draw_label(frame, text, loc, color=(255,255,255), size=.66):
    x,y=loc
    cv2.putText(frame,text,(x+1,y+1),cv2.FONT_HERSHEY_SIMPLEX,size,(0,0,0),4,cv2.LINE_AA)
    cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,2,cv2.LINE_AA)


def save_evidence(directory, full_frame, detection, sop, metrics, timestamp):
    folder=Path(directory)/(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(folder/'full_frame.jpg'),full_frame)
    cv2.imwrite(str(folder/'baseline_crop.jpg'),sop.baseline_image)
    cv2.imwrite(str(folder/'reaction_crop.jpg'),detection['crop'])
    cv2.imwrite(str(folder/'reaction_hsv_mask.jpg'),metrics['overlay'])
    data={'result':'ACIDIC_REACTION_VERIFIED','reagent_identity_verified':False,
          'physical_liquid_contact_proven':False,
          'bbox':detection['bbox'],'detail':sop.end_detail,
          'note':'Vision-only pH indicator transition; physical droplet contact is not proven.'}
    (folder/'result.json').write_text(json.dumps(data,indent=2))
    print('Saved evidence:',folder)


def to_qimage(bgr):
    if bgr is None:
        return None
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    height,width,channels=rgb.shape
    return QImage(rgb.data,width,height,channels*width,QImage.Format_RGB888).copy()


def open_camera(index):
    backends=(cv2.CAP_DSHOW,cv2.CAP_ANY) if sys.platform=='win32' else (cv2.CAP_ANY,)
    for backend in backends:
        cap=cv2.VideoCapture(index,backend)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture()


class CameraScanner(QThread):
    found=pyqtSignal(list)

    def run(self):
        available=[]
        for index in range(6):
            cap=open_camera(index)
            try:
                if cap.isOpened():
                    ok,_=cap.read()
                    if ok:
                        available.append(index)
            finally:
                cap.release()
        self.found.emit(available)


class CameraWorker(QThread):
    frame_ready=pyqtSignal(object)
    notice=pyqtSignal(str)
    failed=pyqtSignal(str)
    ready=pyqtSignal()

    def __init__(self,camera_index,parent=None):
        super().__init__(parent)
        self.camera_index=camera_index
        self._stop=threading.Event()
        self._commands=queue.SimpleQueue()

    def command(self,name):
        self._commands.put(name)

    def stop(self):
        self._stop.set()

    def _save_images(self,frame,detection,metrics):
        DEBUG_DIR.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        items=[('camera',frame)]
        if detection is not None:
            items.append(('crop',detection['crop']))
        if metrics is not None:
            items.append(('hsv_mask',metrics['overlay']))
        saved=[]
        for name,image in items:
            path=DEBUG_DIR/f'{stamp}_{name}.png'
            if image is not None and cv2.imwrite(str(path),image):
                saved.append(path.name)
        if saved:
            self.notice.emit(f"Saved {', '.join(saved)} in {DEBUG_DIR}")
        else:
            self.notice.emit('Could not save images.')

    def run(self):
        cap=None
        try:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f'Model not found: {MODEL_PATH}')
            model=OpenVINOOBB(MODEL_PATH)
            names=model.names
            if isinstance(names,dict) and PAPER_CLASS_ID not in names:
                raise RuntimeError(f'Class {PAPER_CLASS_ID} is not in model classes: {names}')
            if isinstance(names,(list,tuple)) and PAPER_CLASS_ID>=len(names):
                raise RuntimeError(f'Class {PAPER_CLASS_ID} is not in model classes: {names}')
            if self._stop.is_set():
                return

            cap=open_camera(self.camera_index)
            if not cap.isOpened():
                raise RuntimeError(f'Could not open camera {self.camera_index}.')
            self.ready.emit()
            self.notice.emit('Camera ready (OpenVINO CPU). Show fresh yellow pH paper.')

            sop=AutoSOP()
            missed_frames=0
            previous_state=sop.state
            while not self._stop.is_set():
                ok,frame=cap.read()
                if not ok:
                    missed_frames+=1
                    if missed_frames>=10:
                        raise RuntimeError('Camera stopped providing frames.')
                    self.msleep(30)
                    continue
                missed_frames=0

                candidates=all_paper_detections(
                    model,frame,PAPER_CLASS_ID,CONFIDENCE
                )
                matched=[d for d in candidates if plausible_match(sop.prev_bbox,d['bbox'])]
                detection=max(matched,key=lambda d:d['score']) if matched else None
                metrics=analyze_paper(detection['crop'],detection['mask']) if detection else None

                while not self._commands.empty():
                    command=self._commands.get_nowait()
                    if command=='reset':
                        sop.reset()
                        self.notice.emit('Reset. Show fresh yellow pH paper again.')
                    elif command=='save':
                        self._save_images(frame,detection,metrics)

                just_passed=sop.update(detection,metrics,time.monotonic())
                if sop.state!=previous_state:
                    self.notice.emit(sop.last_diagnostic)
                    previous_state=sop.state
                if just_passed:
                    save_evidence(EVIDENCE_DIR,frame,detection,sop,metrics,time.monotonic())
                    self.notice.emit(
                        f'Acidic reaction verified. Evidence saved in {EVIDENCE_DIR}'
                    )

                display=frame.copy()
                confidence=None
                if detection is not None:
                    x1,y1,x2,y2=detection['bbox']
                    confidence=detection['score']
                    clear=bool(
                        metrics and metrics['sufficient'] and
                        metrics['dark']<=MAX_DARK_FRACTION and
                        metrics['coverage']>=MIN_COLOR_COVERAGE
                    )
                    color=(68,206,113) if clear else (0,179,255)
                    if detection['quad'].shape==(4,2):
                        cv2.polylines(display,[detection['quad']],True,color,2,cv2.LINE_AA)
                    else:
                        cv2.rectangle(display,(x1,y1),(x2,y2),color,2)
                    label=f'acidph {confidence:.2f}'
                    if metrics is not None:
                        label+=(f"  Yellow {metrics['yellow']:.0%}"
                                f"  Pink {metrics['pink']:.0%}"
                                f"  Dark {metrics['dark']:.0%}")
                    draw_label(display,label,(x1,max(24,y1-8)),color,.55)

                self.frame_ready.emit({
                    'frame':to_qimage(display),
                    'crop':to_qimage(detection['crop']) if detection else None,
                    'mask':to_qimage(metrics['overlay']) if metrics else None,
                    'phase':sop.state,
                    'detected':detection is not None,
                    'missing_frames':sop.missing,
                    'sufficient':bool(metrics and metrics['sufficient']),
                    'clear':bool(
                        metrics and metrics['sufficient'] and
                        metrics['dark']<=MAX_DARK_FRACTION and
                        metrics['coverage']>=MIN_COLOR_COVERAGE
                    ),
                    'yellow':metrics['yellow'] if metrics else 0.0,
                    'pink':metrics['pink'] if metrics else 0.0,
                    'dark':metrics['dark'] if metrics else 0.0,
                    'coverage':metrics['coverage'] if metrics else 0.0,
                    'confidence':confidence,
                    'fresh_hits':sop.fresh_hits,
                    'reaction_hits':sop.reaction_hits,
                    'diagnostic':sop.last_diagnostic,
                })
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')
        finally:
            if cap is not None:
                cap.release()


class ImagePanel(QFrame):
    def __init__(self,placeholder,minimum_height=180):
        super().__init__()
        self.placeholder=placeholder
        self.panel_height=minimum_height
        self.current_image=None
        self.setMinimumHeight(minimum_height)
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.setObjectName('imagePanel')

    def set_image(self,image):
        self.current_image=image
        self.update()

    def sizeHint(self):
        return QSize(400,self.panel_height)

    def minimumSizeHint(self):
        return QSize(180,self.panel_height)

    def paintEvent(self,event):
        super().paintEvent(event)
        bounds=self.rect().adjusted(8,8,-8,-8)
        painter=QPainter(self)
        if self.current_image is None:
            painter.setPen(QColor('#a8b7c4'))
            painter.drawText(bounds,Qt.AlignCenter|Qt.TextWordWrap,self.placeholder)
        elif bounds.width()>0 and bounds.height()>0:
            image_size=self.current_image.size()
            image_size.scale(bounds.size(),Qt.KeepAspectRatio)
            target=QRect(
                bounds.x()+(bounds.width()-image_size.width())//2,
                bounds.y()+(bounds.height()-image_size.height())//2,
                image_size.width(),image_size.height()
            )
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawImage(target,self.current_image)
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Automatic pH Paper Guide')
        self.resize(1280,790)
        self.worker=None
        self.scanner=None
        self.closing=False
        self.phase='WAIT_FRESH'
        self.camera_running=False
        self.last_data=None

        root=QWidget()
        self.setCentralWidget(root)
        outer=QHBoxLayout(root)
        outer.setContentsMargins(18,18,18,18)
        outer.setSpacing(18)

        left=QVBoxLayout()
        left.setSpacing(12)
        outer.addLayout(left,3)
        title=QLabel('Automatic pH paper visual check')
        title.setObjectName('title')
        title.setWordWrap(True)
        left.addWidget(title)
        subtitle=QLabel('Continuous YOLO detection with automatic yellow-to-pink reaction monitoring')
        subtitle.setObjectName('muted')
        subtitle.setWordWrap(True)
        left.addWidget(subtitle)
        self.video=ImagePanel('Choose a camera and press Start camera',260)
        left.addWidget(self.video,5)

        previews=QGridLayout()
        previews.setSpacing(10)
        crop_title=QLabel('Detected pH paper')
        mask_title=QLabel('HSV yellow / pink / dark mask')
        crop_title.setObjectName('section')
        mask_title.setObjectName('section')
        crop_title.setWordWrap(True)
        mask_title.setWordWrap(True)
        previews.addWidget(crop_title,0,0)
        previews.addWidget(mask_title,0,1)
        self.crop=ImagePanel('Waiting for pH paper',120)
        self.mask=ImagePanel('Waiting for color pixels',120)
        previews.addWidget(self.crop,1,0)
        previews.addWidget(self.mask,1,1)
        left.addLayout(previews,2)

        right_box=QFrame()
        right_box.setObjectName('sidebar')
        right_scroll=QScrollArea()
        right_scroll.setObjectName('sidebarScroll')
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setFixedWidth(355)
        right_scroll.setWidget(right_box)
        outer.addWidget(right_scroll)
        right=QVBoxLayout(right_box)
        right.setContentsMargins(18,18,18,18)
        right.setSpacing(12)

        camera_title=QLabel('Camera')
        camera_title.setObjectName('section')
        right.addWidget(camera_title)
        camera_row=QHBoxLayout()
        self.camera_combo=QComboBox()
        self.camera_combo.setEditable(True)
        self.camera_combo.addItem('Camera 0',0)
        self.camera_combo.setToolTip('Select a discovered camera, or type its numeric index.')
        camera_row.addWidget(self.camera_combo,1)
        self.scan_button=QPushButton('Refresh')
        self.scan_button.clicked.connect(self.scan_cameras)
        camera_row.addWidget(self.scan_button)
        right.addLayout(camera_row)

        camera_controls=QHBoxLayout()
        self.start_button=QPushButton('Start camera')
        self.start_button.setObjectName('primary')
        self.start_button.clicked.connect(self.start_camera)
        camera_controls.addWidget(self.start_button)
        self.stop_button=QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_camera)
        camera_controls.addWidget(self.stop_button)
        right.addLayout(camera_controls)

        steps_title=QLabel('Automatic process')
        steps_title.setObjectName('section')
        right.addWidget(steps_title)
        self.step_labels=[]
        for text in [
            '1  Choose and start a camera',
            '2  Hold fresh yellow pH paper in view',
            '3  Keep it visible and place acid/sample on it',
            '4  Automatically identify a steady pink reaction',
        ]:
            label=QLabel(text)
            label.setWordWrap(True)
            label.setObjectName('step')
            right.addWidget(label)
            self.step_labels.append(label)

        self.instruction=QLabel()
        self.instruction.setWordWrap(True)
        self.instruction.setObjectName('instruction')
        right.addWidget(self.instruction)
        self.state_label=QLabel('State: Waiting for camera')
        self.state_label.setObjectName('state')
        right.addWidget(self.state_label)
        self.measure_label=QLabel('Yellow: --    Pink: --    Dark: --')
        right.addWidget(self.measure_label)
        self.readability_label=QLabel('Paper: --')
        self.readability_label.setWordWrap(True)
        right.addWidget(self.readability_label)
        self.progress=QProgressBar()
        right.addWidget(self.progress)

        bottom_row=QHBoxLayout()
        self.reset_button=QPushButton('Reset process')
        self.reset_button.clicked.connect(lambda:self.send_command('reset'))
        bottom_row.addWidget(self.reset_button)
        self.save_button=QPushButton('Save images')
        self.save_button.clicked.connect(lambda:self.send_command('save'))
        bottom_row.addWidget(self.save_button)
        right.addLayout(bottom_row)
        right.addStretch(1)
        self.notice_label=QLabel('Select a camera to begin.')
        self.notice_label.setWordWrap(True)
        self.notice_label.setObjectName('muted')
        right.addWidget(self.notice_label)
        caveat=QLabel(
            'Visual indication only. The camera observes a color transition; '
            'it does not prove physical contact, identify the reagent, or measure exact pH.'
        )
        caveat.setWordWrap(True)
        caveat.setObjectName('muted')
        right.addWidget(caveat)

        self.setStyleSheet("""
            QMainWindow, QWidget { background: #101820; color: #edf3f6;
                font-family: 'Segoe UI'; font-size: 14px; }
            QLabel#title { font-size: 27px; font-weight: 700; }
            QLabel#section { font-size: 16px; font-weight: 700; }
            QLabel#muted { color: #a8b7c4; font-size: 12px; }
            QFrame#imagePanel { background: #1b2731; border: 1px solid #344553;
                border-radius: 10px; color: #a8b7c4; }
            QFrame#sidebar { background: #192631; border-radius: 12px; }
            QScrollArea#sidebarScroll { background: #192631; border: 0;
                border-radius: 12px; }
            QLabel#step { background: #243440; border-radius: 7px; padding: 10px;
                color: #acbdc8; }
            QLabel#step[mode="active"] { background: #254d58; color: #f4fcff;
                border: 1px solid #49b9c2; }
            QLabel#step[mode="done"] { background: #204336; color: #b8f4cd; }
            QLabel#instruction { background: #243440; border-radius: 7px;
                padding: 12px; min-height: 54px; }
            QLabel#state { font-size: 16px; font-weight: 700; color: #7cdbc3; }
            QPushButton { background: #344958; color: white; border: 0;
                border-radius: 7px; padding: 10px; }
            QPushButton:hover { background: #416073; }
            QPushButton:disabled { background: #293640; color: #71838e; }
            QPushButton#primary { background: #177b90; font-weight: 700; }
            QPushButton#primary:hover { background: #1b93aa; }
            QComboBox { background: #263a48; border: 1px solid #4d6572;
                border-radius: 6px; padding: 8px; }
            QComboBox QAbstractItemView { background: #263a48; color: white; }
            QProgressBar { border: 1px solid #476071; border-radius: 6px;
                background: #263a48; text-align: center; }
            QProgressBar::chunk { background: #49b9c2; border-radius: 5px; }
        """)
        screen=QApplication.primaryScreen()
        if screen is not None:
            available=screen.availableGeometry()
            minimum=self.minimumSizeHint()
            self.resize(
                max(minimum.width(),min(1180,available.width()-40)),
                max(minimum.height(),min(760,available.height()-40))
            )
        self.update_guide()
        self.scan_cameras()

    def scan_cameras(self):
        if self.scanner is not None or self.worker is not None:
            return
        self.scan_button.setEnabled(False)
        self.notice_label.setText('Looking for cameras...')
        self.scanner=CameraScanner(self)
        self.scanner.found.connect(self.cameras_found)
        self.scanner.finished.connect(self.scanner_finished)
        self.scanner.start()

    def cameras_found(self,indices):
        current=self.selected_camera(default=0)
        self.camera_combo.clear()
        for index in indices:
            self.camera_combo.addItem(f'Camera {index}',index)
        if not indices:
            self.camera_combo.addItem('Camera 0',0)
            self.notice_label.setText('No camera found in scan. You can type a camera index.')
        else:
            self.notice_label.setText(f'Found {len(indices)} camera(s). Select one and start.')
        match=self.camera_combo.findData(current)
        if match>=0:
            self.camera_combo.setCurrentIndex(match)
        self.update_guide()

    def scanner_finished(self):
        if self.scanner is not None:
            self.scanner.deleteLater()
            self.scanner=None
        self.update_guide()
        if self.closing:
            self.close()

    def selected_camera(self,default=None):
        data=self.camera_combo.currentData()
        if self.camera_combo.currentText().strip()==f'Camera {data}':
            return data
        try:
            index=int(self.camera_combo.currentText().strip())
            if index>=0:
                return index
        except ValueError:
            pass
        return default

    def start_camera(self):
        if self.worker is not None or self.scanner is not None:
            return
        index=self.selected_camera()
        if index is None:
            QMessageBox.warning(self,'Camera index','Choose a camera or type a nonnegative index.')
            return
        self.camera_running=False
        self.last_data=None
        self.phase='WAIT_FRESH'
        self.notice_label.setText('Loading model and opening camera...')
        self.worker=CameraWorker(index,self)
        self.worker.ready.connect(self.camera_ready)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.notice.connect(self.notice_label.setText)
        self.worker.failed.connect(self.camera_failed)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()
        self.update_guide()

    def camera_ready(self):
        self.camera_running=True
        self.update_guide()

    def camera_failed(self,message):
        self.notice_label.setText(message)
        QMessageBox.critical(self,'Camera or model error',message)

    def stop_camera(self):
        if self.worker is not None:
            self.notice_label.setText('Stopping camera...')
            self.worker.stop()
            self.update_guide()

    def worker_finished(self):
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker=None
        self.camera_running=False
        self.last_data=None
        self.phase='WAIT_FRESH'
        self.video.set_image(None)
        self.crop.set_image(None)
        self.mask.set_image(None)
        self.update_guide()
        if self.closing:
            self.close()

    def send_command(self,name):
        if self.worker is not None and self.camera_running:
            self.worker.command(name)

    def on_frame(self,data):
        self.last_data=data
        self.phase=data['phase']
        self.video.set_image(data['frame'])
        self.crop.set_image(data['crop'])
        self.mask.set_image(data['mask'])
        self.update_guide()

    def update_guide(self):
        running=self.camera_running
        phase=self.phase
        active_by_phase={'WAIT_FRESH':1,'WATCH_REACTION':2,'VERIFIED':4,'REJECTED':3}
        active=active_by_phase.get(phase,1) if running else 0
        for index,label in enumerate(self.step_labels):
            mode='done' if index<active else 'active' if index==active else 'pending'
            if label.property('mode')!=mode:
                label.setProperty('mode',mode)
                label.style().unpolish(label)
                label.style().polish(label)

        if not running:
            instruction='Select a camera and start the live preview.'
            state='Waiting for camera'
        elif phase=='WAIT_FRESH':
            instruction=(
                'Hold unused yellow pH paper steady in view. Its fresh baseline '
                'will be accepted automatically.'
            )
            state='Detecting fresh yellow paper'
        elif phase=='WATCH_REACTION':
            instruction=(
                'Keep the same pH paper visible and place the acid/sample on it. '
                'The reaction is monitored automatically.'
            )
            state='Waiting for pink reaction'
        elif phase=='VERIFIED':
            instruction='A sustained pink change was detected. Reset to test a new paper.'
            state='Acidic reaction verified'
        else:
            instruction='The automatic check was rejected. Reset and show a fresh paper.'
            state='Test rejected'
        self.instruction.setText(instruction)
        self.state_label.setText(f'State: {state}')

        data=self.last_data
        if running and data is not None:
            if data['detected']:
                self.measure_label.setText(
                    f"Yellow: {data['yellow']:.0%}    Pink: {data['pink']:.0%}    "
                    f"Dark: {data['dark']:.0%}"
                )
                readable='Clear' if data['clear'] else 'Analysis paused: covered/dark'
                self.readability_label.setText(
                    f"Paper: {readable} · Color coverage: {data['coverage']:.0%} · "
                    f"YOLO confidence: {data['confidence']:.2f}"
                )
            else:
                self.measure_label.setText('Yellow: --    Pink: --    Dark: --')
                if data['missing_frames']:
                    self.readability_label.setText(
                        'Paper: Briefly hidden; keeping the same strip '
                        f"({data['missing_frames']}/{MISSING_GRACE_FRAMES} frames)"
                    )
                else:
                    self.readability_label.setText('Paper: Not detected')

            if phase=='WAIT_FRESH':
                self.progress.setRange(0,FRESH_FRAMES)
                self.progress.setValue(min(data['fresh_hits'],FRESH_FRAMES))
                self.progress.setFormat('Fresh-paper stability: %v / %m frames')
            else:
                self.progress.setRange(0,REACTION_FRAMES)
                value=REACTION_FRAMES if phase=='VERIFIED' else data['reaction_hits']
                self.progress.setValue(min(value,REACTION_FRAMES))
                self.progress.setFormat('Pink-reaction stability: %v / %m frames')
        else:
            self.measure_label.setText('Yellow: --    Pink: --    Dark: --')
            self.readability_label.setText('Paper: --')
            self.progress.setRange(0,FRESH_FRAMES)
            self.progress.setValue(0)
            self.progress.setFormat('Fresh-paper stability: %v / %m frames')

        self.reset_button.setEnabled(running)
        self.save_button.setEnabled(running)
        self.start_button.setEnabled(self.worker is None and self.scanner is None)
        self.stop_button.setEnabled(self.worker is not None)
        self.scan_button.setEnabled(self.worker is None and self.scanner is None)
        self.camera_combo.setEnabled(self.worker is None and self.scanner is None)

    def closeEvent(self,event):
        if self.worker is not None or self.scanner is not None:
            self.closing=True
            if self.worker is not None:
                self.worker.stop()
            event.ignore()
        else:
            event.accept()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-model',action='store_true',
                        help='Run a CPU inference check without opening a camera or window.')
    args=parser.parse_args()
    if args.check_model:
        model=OpenVINOOBB(MODEL_PATH)
        image=np.zeros((480,640,3),dtype=np.uint8)
        model.detect(image,PAPER_CLASS_ID,CONFIDENCE)  # Warm up the compiled model.
        start=time.perf_counter()
        for _ in range(5):
            detections=model.detect(image,PAPER_CLASS_ID,CONFIDENCE)
        print(json.dumps({
            'model':str(MODEL_PATH),'device':model.device,
            'input_shape':[1,3,model.height,model.width],
            'classes':model.names,'blank_image_detections':len(detections),
            'mean_ms':round((time.perf_counter()-start)*1000/5,2),
        },indent=2))
        return
    app=QApplication(sys.argv)
    window=MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__=='__main__':
    main()
