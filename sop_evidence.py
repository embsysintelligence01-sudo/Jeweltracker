"""Geometry and evidence checks shared by live and recorded SOP paths."""
import cv2
import numpy as np
from new_rub_detection.geometry import DEST, project_mask
from rub_preparation import clear_surface


def rectangle_quad(box):
    x1,y1,x2,y2=box
    return np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])


def contact_region(contact, quad=None):
    """Contact footprints use source-frame coordinates, never display coordinates."""
    if quad is None:
        quad=rectangle_quad(contact['stone'])
    matrix=cv2.getPerspectiveTransform(np.asarray(quad,np.float32),DEST)
    box=contact['box']
    if contact.get('mask') is not None:
        local=(contact['mask']>0).astype(np.uint8)*255
        ox,oy=contact['origin']
        shift=np.float32([[1,0,ox],[0,1,oy],[0,0,1]])
        region=cv2.warpPerspective(local,matrix@shift,(512,512),flags=cv2.INTER_NEAREST)
    else:
        region=np.zeros((512,512),np.uint8)
        points=cv2.perspectiveTransform(rectangle_quad(box)[None],matrix)[0]
        cv2.fillConvexPoly(region,np.rint(points).astype(np.int32),255)
    region[:3]=region[-3:]=0
    region[:,:3]=region[:,-3:]=0
    return region


def contact_match(mask, region):
    if region is None or not np.any(region):
        return None
    expanded=cv2.dilate(region,np.ones((17,17),np.uint8))>0
    return np.count_nonzero((mask>0)&expanded)/max(1,np.count_nonzero(mask))


def validate_mark(candidate, frame, quad, timestamp, contact=None, placement=None):
    """Require three distinct clear frames, positive appearance, and contact match.

    No extra inference is performed. Candidate coordinates remain relative to
    the saved stone reference. Missing contact is *unverified*, never a pass.
    """
    if timestamp < candidate['timestamp']:
        return 'CHECKING','Frame precedes the candidate.'
    stone=cv2.fillConvexPoly(np.zeros(frame.shape[:2],np.uint8),np.rint(quad).astype(np.int32),255)
    x,y,w,h=cv2.boundingRect(np.rint(quad).astype(np.int32))
    x1,y1=max(0,x),max(0,y)
    x2,y2=min(frame.shape[1],x+w),min(frame.shape[0],y+h)
    if x2<=x1 or y2<=y1:return 'REJECTED_OUTSIDE_STONE','Stone is outside the camera frame.'
    crop=frame[y1:y2,x1:x2]
    try:
        clear_surface(crop,stone[y1:y2,x1:x2])
    except RuntimeError:
        return 'CHECKING','Stone covered; waiting for independent clear frames.'
    transform=cv2.getPerspectiveTransform(np.asarray(quad,np.float32),DEST)
    projected=project_mask(candidate['new_mask'],frame.shape,transform)
    allowed=cv2.erode(stone,np.ones((3,3),np.uint8))
    if placement is not None:
        roi=cv2.fillPoly(np.zeros_like(stone),[np.asarray(placement,np.int32)],255)
        allowed &= roi
    pixels=projected>0
    if np.count_nonzero(pixels&(allowed>0))/max(1,np.count_nonzero(pixels))<.98:
        return 'REJECTED_OUTSIDE_STONE','Candidate extends beyond the stone or fixed test ROI.'
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    skin=(hsv[:,:,0]<=14)&(hsv[:,:,1]>=110)&(hsv[:,:,2]>60)
    if np.count_nonzero(pixels[y1:y2,x1:x2]&skin)/max(1,np.count_nonzero(pixels))>.10:
        return 'REJECTED_HAND','Candidate overlaps skin rather than a clear stone mark.'
    current=cv2.warpPerspective(frame,transform,(512,512))
    previous=cv2.warpPerspective(candidate['frame'],candidate['frame_to_reference'],(512,512))
    delta=cv2.absdiff(current,previous).mean(axis=2)[candidate['new_mask']>0]
    if not delta.size or float(np.median(delta))>20:
        return 'REJECTED_TRANSIENT','Candidate disappeared or changed after the hand/jewel moved away.'
    seen=candidate.setdefault('clear_frames',[])
    if not seen or timestamp-seen[-1]>=.04:
        seen.append(timestamp)
    if len(seen)<3 or seen[-1]-seen[0]<.12:
        return 'CHECKING','Waiting for persistence in three distinct clear frames.'
    overlap=contact_match(candidate['new_mask'],contact)
    candidate['contact_overlap']=overlap
    if overlap is None:
        return 'UNVERIFIED_CONTACT_LOCATION','New-looking mark persisted, but its rubbing-contact location is unverified.'
    if overlap<.25:
        return 'REJECTED_WRONG_LOCATION','Candidate mark is unrelated to the recorded jewellery-contact location.'
    return 'VERIFIED','New mark persisted inside the stone and matches jewellery contact.'


def local_motion(previous, frame, stone, allow_broad=False):
    """Cheap weak fallback; motion alone can never verify a jewel or rubbing."""
    x1,y1,x2,y2=map(int,stone)
    crop=frame[max(0,y1):min(frame.shape[0],y2),max(0,x1):min(frame.shape[1],x2)]
    if not crop.size:return None,None
    gray=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(128,128))
    gray=cv2.GaussianBlur(gray,(5,5),0).astype(np.float32)
    if previous is None:return gray,None
    delta=gray-previous
    delta-=np.median(delta)
    movement=(np.abs(delta)>18).astype(np.uint8)*255
    fraction=np.count_nonzero(movement)/movement.size
    if not .003<fraction<(.98 if allow_broad else .30):return gray,None
    contours,_=cv2.findContours(movement,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:return gray,None
    c=max(contours,key=cv2.contourArea)
    if cv2.contourArea(c)<12:return gray,None
    x,y,w,h=cv2.boundingRect(c)
    box=(x1+round(x*crop.shape[1]/128),y1+round(y*crop.shape[0]/128),
         x1+round((x+w)*crop.shape[1]/128),y1+round((y+h)*crop.shape[0]/128))
    return gray,dict(box=box,score=float(cv2.contourArea(c)))
