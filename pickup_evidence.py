"""Small fixed-region occlusion probes. No detector/model calls or frame writes."""
from __future__ import annotations
import cv2
import numpy as np


def area(box):
    b=np.asarray(box,float)
    return max(0.,b[2]-b[0])*max(0.,b[3]-b[1])


def coverage(a,b):
    return (max(0.,min(a[2],b[2])-max(a[0],b[0]))*
            max(0.,min(a[3],b[3])-max(a[1],b[1])))/max(1.,area(a))


def sample(frame,box):
    h,w=frame.shape[:2];x1,y1,x2,y2=np.rint(box).astype(int)
    x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
    if x2<=x1 or y2<=y1:return None
    return cv2.resize(frame[y1:y2,x1:x2],(32,32),interpolation=cv2.INTER_AREA)


class HomeAnchor:
    """An immutable original home, a settled current home, and reliable detections.

    A changed/covered patch only opens evidence. Departure requires clear empty
    background or reliable identity-supported observation away from home.
    """
    def __init__(self,box,timestamp,frame):
        self.original_box=np.asarray(box,float).copy()
        self.home_box=self.original_box.copy()
        self.last_box=self.original_box.copy();self.last_seen=timestamp
        self.template=None;self.background=None;self.enabled=False
        self.state='AT_HOME';self.presence='UNKNOWN';self.observed_at=timestamp
        self.event=0;self.started=None;self.departed_at=None;self.returned_at=None
        self.empty_since=None;self.empty_count=0;self.return_probe=None
        self._capture(frame,self.home_box)

    def _capture(self,frame,box):
        self.template=sample(frame,box)
        if self.template is None:return
        # Background is measured around the ORIGINAL clear jewel, never from
        # a hand-covered patch. A homogeneous detector false-positive is inert.
        b=np.asarray(box,float);pad=max(5.,.15*max(b[2:]-b[:2]))
        outer=sample(frame,b+[-pad,-pad,pad,pad])
        if outer is None:return
        ring=np.concatenate((outer[:3].reshape(-1,3),outer[-3:].reshape(-1,3),
                             outer[:, :3].reshape(-1,3),outer[:, -3:].reshape(-1,3)))
        self.background=np.median(ring,axis=0)
        fg=np.max(np.abs(self.template.astype(float)-self.background),axis=2)>25
        self.enabled=bool(np.mean(fg)>.025 and np.std(self.template.astype(float))>4)

    def clear_around(self,frame,box):
        b=np.asarray(box,float);pad=max(5.,.15*max(b[2:]-b[:2]))
        outer=sample(frame,b+[-pad,-pad,pad,pad])
        if outer is None or self.background is None:return False
        ring=np.concatenate((outer[:3].reshape(-1,3),outer[-3:].reshape(-1,3),
                             outer[:,:3].reshape(-1,3),outer[:,-3:].reshape(-1,3)))
        return float(np.mean(np.max(np.abs(ring.astype(float)-self.background),axis=1)<28))>=.75

    def _open(self,timestamp):
        if self.started is None or self.state in ('CANCELLED','RETURNED'):
            self.event+=1;self.started=timestamp;self.departed_at=self.returned_at=None
            self.empty_since=None;self.empty_count=0;self.return_probe=None

    def probe(self,frame,timestamp):
        if timestamp<=self.observed_at:return
        self.observed_at=timestamp
        if not self.enabled:return
        cur=sample(frame,self.home_box)
        if cur is None:self.presence='UNKNOWN';return
        a=cur.astype(np.float32);b=self.template.astype(np.float32)
        # Ignore small exposure shifts; don't normalize away a hand or removal.
        offset=np.clip(np.median(a-b,axis=(0,1)),-12,12)
        diff=np.mean(np.abs(a-b-offset),axis=2)
        present=float(np.mean(diff))<11 and float(np.mean(diff>24))<.18
        empty=float(np.mean(np.max(np.abs(a-self.background),axis=2)<24))>=.90
        if present:
            self.presence='PRESENT';self.empty_since=None;self.empty_count=0
            if self.started is not None and self.departed_at is None:self.state='CANCELLED'
            elif self.departed_at is None:self.state='AT_HOME'
        elif empty:
            self.presence='EMPTY';self._open(timestamp)
            if self.empty_since is None:self.empty_since=timestamp;self.empty_count=0
            self.empty_count+=1
            if self.empty_count>=2 and timestamp-self.empty_since>=.08:
                if self.departed_at is None:self.departed_at=timestamp
                self.state='DEPARTED' if self.returned_at is None else 'RETURNED'
            elif self.departed_at is None:self.state='OCCLUDED_HOME'
        else:
            self.presence='COVERED';self._open(timestamp)
            self.empty_since=None;self.empty_count=0
            if self.departed_at is None:self.state='OCCLUDED_HOME'

    def cancel_at_home(self,box,timestamp,frame):
        # A jewel can be nudged while a hand passes. A clear, full-size,
        # appearance-matching detection near home cancels that candidate after
        # two fresh observations; a partial/covered box cannot do so.
        previous=getattr(self,'home_clear_probe',None)
        centre=(np.asarray(box)[:2]+np.asarray(box)[2:])/2
        if previous is None or timestamp-previous['last']>.65 or np.linalg.norm(centre-previous['centre'])>5:
            self.home_clear_probe=dict(start=timestamp,last=timestamp,centre=centre);return
        if timestamp<=previous['last']:return
        previous['last']=timestamp
        if timestamp-previous['start']>=.12:
            self.state='CANCELLED';self.presence='PRESENT'
            self.home_box=np.asarray(box,float).copy();self._capture(frame,self.home_box)
            self.home_clear_probe=None

    def near_home(self,box):return coverage(self.home_box,box)>=.45

    def complete_box(self,box):
        ratio=area(box)/max(1.,area(self.last_box))
        old=np.sort(self.last_box[2:]-self.last_box[:2]);new=np.sort(np.asarray(box)[2:]-np.asarray(box)[:2])
        return .45<=ratio<=2.2 and bool(np.all(new/np.maximum(old,1.)>=.5))

    def looks_like(self,frame,box):
        patch=sample(frame,box)
        if patch is None or self.template is None:return False
        # Compare foreground colour only for the unique-lost-jewel fallback.
        # Existing SLIC/gallery decisions remain the stronger identity evidence.
        def colours(image):
            hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            mask=(np.max(np.abs(image.astype(float)-self.background),axis=2)>25).astype(np.uint8)
            if np.count_nonzero(mask)<12:return None
            hist=cv2.calcHist([hsv],[0,1],mask,[18,8],[0,180,0,256]);cv2.normalize(hist,hist)
            return hist
        a,b=colours(self.template),colours(patch)
        # A uniform skin-coloured false detection must not become a returned
        # jewel merely because its hue resembles gold.
        ref_texture=float(np.std(cv2.cvtColor(self.template,cv2.COLOR_BGR2GRAY)))
        cur_texture=float(np.std(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)))
        if cur_texture<max(2.,.25*ref_texture):return False
        return a is not None and b is not None and cv2.compareHist(a,b,cv2.HISTCMP_CORREL)>.75

    def reliable(self,box,timestamp,frame,on_bed,identity_supported):
        self.last_box=np.asarray(box,float).copy();self.last_seen=timestamp
        if self.started is None or self.state=='CANCELLED' or self.returned_at is not None:return
        away=not self.near_home(box)
        if self.departed_at is None and away and identity_supported:
            self.departed_at=timestamp;self.state='DEPARTED'
        if self.departed_at is None or not on_bed or not self.clear_around(frame,box):
            self.return_probe=None;return
        if not identity_supported or not self.looks_like(frame,box):
            self.return_probe=None;return
        previous=self.return_probe
        centre=(np.asarray(box)[:2]+np.asarray(box)[2:])/2
        tolerance=max(5.,.12*np.sqrt(max(1.,area(box))))
        if previous is None or timestamp-previous['last']>.65 or np.linalg.norm(centre-previous['centre'])>tolerance:
            self.return_probe=dict(start=timestamp,last=timestamp,count=1,centre=centre);return
        if timestamp<=previous['last']:return
        previous.update(last=timestamp,count=previous['count']+1,centre=centre)
        if previous['count']>=2 and timestamp-previous['start']>=.12:
            self.returned_at=timestamp;self.state='RETURNED'
            # Accept any settled bed position. The original anchor stays intact.
            self.home_box=np.asarray(box,float).copy();self._capture(frame,self.home_box)
            self.presence='PRESENT'

    def snapshot(self):
        return dict(enabled=self.enabled,state=self.state,event=self.event,started=self.started,
                    departed_at=self.departed_at,returned_at=self.returned_at,observed_at=self.observed_at,
                    home_presence=self.presence,original_home_box=self.original_box.tolist(),
                    home_box=self.home_box.tolist(),last_reliable_box=self.last_box.tolist(),last_reliable_at=self.last_seen)
