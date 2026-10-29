"""Bounded, ranked rubbing evidence inside one immutable cycle interval."""
import cv2
import numpy as np
from sop_evidence import contact_region, local_motion


def recover_contact(journal, cycle, observer, save_picture):
    start=cycle['started']
    end=min(cycle.get('returned') or cycle['finished'],
            cycle['acid_onset']-.05 if cycle.get('acid_onset') is not None else cycle['finished'])
    cutoff=cycle.get('contact_closed_at')
    if cutoff is not None:
        end=min(end,cutoff)
    live=cycle.get('contact')
    # Tracking can report a return before the operator finishes rubbing. The
    # timestamped live source is stronger evidence than that return heuristic.
    # Keep it only inside this cycle and strictly before acid application.
    valid_live=(live is not None and live.get('frame') is not None
                and start<=live['timestamp']<=cycle['finished']
                and (cutoff is None or live['timestamp']<=cutoff)
                and (cycle.get('acid_onset') is None or live['timestamp']<cycle['acid_onset']))
    return_precedes_contact=bool(valid_live and live['timestamp']>end)
    if valid_live:
        end=max(end,live['timestamp'])
    entries=journal.between(start,end,fps=6)
    limit=cycle.get('recovery_contact_frames',24)
    indices=np.unique(np.linspace(0,len(entries)-1,min(limit,len(entries)),dtype=int))
    observations=cycle.get('stone_observations',[])
    selected=None
    samples=[]
    region=None
    motions=[]
    previous=None
    gray=None
    geometry=None
    confirmed=False
    model_errors=[]
    if valid_live:
        source=min(entries,key=lambda e:abs(e['timestamp']-live['timestamp'])) if entries else None
        selected=dict(ev=live,frame=live['frame'],timestamp=live['timestamp'],
                      source_frame=source['file'] if source else None,stone=live['stone'],rank=3,score=live.get('area',0))
        region=cycle.get('contact_region')
        confirmed=bool(live.get('confirmed'))
    # A confirmed live source requires no repeated model work.
    for index in ([] if confirmed else indices):
        entry=entries[index]
        ts=entry['timestamp']
        if not observations:continue
        seen,stone=min(observations,key=lambda p:abs(p[0]-ts))
        if abs(seen-ts)>.4:continue
        frame=journal.read(entry)
        # A large pose jump invalidates a motion bridge instead of fabricating rub.
        if geometry is not None and np.linalg.norm(np.asarray(stone)-geometry)>40:
            gray=None
        gray,motion=local_motion(gray,frame,stone)
        geometry=np.asarray(stone)
        try:
            ev=observer(frame,stone) if observer is not None else None
        except Exception as exc:
            if len(model_errors)<3:model_errors.append(str(exc))
            ev=None
        if ev is not None:
            kind=ev.get('kind','SEGMENTATION')
            rank=3 if kind=='SEGMENTATION' else 2
            mask=ev.get('mask')
            score=np.count_nonzero(mask) if mask is not None else 1
            sample=dict(timestamp=ts,source_frame=entry['file'],contact_bbox=[int(v) for v in ev['box']],kind=kind)
            samples.append(sample)
            if rank==3:
                current=contact_region(dict(ev,stone=stone))
                region=current if region is None else region|current
                size=np.maximum(np.asarray(stone[2:])-stone[:2],1)
                point=(np.asarray(ev['c'])-stone[:2])/size*256
                if previous is not None and 0<ts-previous[0]<=.45:
                    motions.append(point-previous[1]);motions=motions[-6:]
                    distances=[np.linalg.norm(m) for m in motions]
                    reversals=any(np.dot(p,q)<-1 for p,q in zip(motions,motions[1:]))
                    if len(distances)>=3 and max(distances)>=2.5 and sum(distances)>=6 and reversals:
                        confirmed=True
                previous=ts,point
            else:
                previous=None;motions.clear()
            choice=dict(ev=ev,frame=frame,timestamp=ts,source_frame=entry['file'],stone=stone,rank=rank,score=score)
        elif motion is not None:
            previous=None;motions.clear()
            choice=dict(ev=dict(box=motion['box'],kind='LOCALIZED_MOTION'),frame=frame,
                        timestamp=ts,source_frame=entry['file'],stone=stone,rank=1,score=motion['score'])
        else:
            previous=None;motions.clear()
            continue
        if selected is None or (choice['rank'],choice['score'])>(selected['rank'],selected['score']):
            selected=choice
    status=('CONTACT_AND_MOTION_LIVE' if valid_live and live.get('confirmed') else
            'CONTACT_AND_MOTION_RECOVERED' if confirmed else
            'GOLD_CONTACT_VISIBLE' if selected and selected['rank']==3 else
            'UNVERIFIED_BOUNDING_BOX' if selected and selected['rank']==2 else
            'UNVERIFIED_LOCALIZED_MOTION' if selected else 'CONTACT_NOT_VISIBLE')
    reason={'CONTACT_AND_MOTION_LIVE':'Timestamped live segmentation contact and rubbing motion.',
            'CONTACT_AND_MOTION_RECOVERED':'Recorded segmentation contact and back-and-forth movement on the stone.',
            'GOLD_CONTACT_VISIBLE':'Segmentation contact visible; rubbing motion not confirmed.',
            'UNVERIFIED_BOUNDING_BOX':'Jewellery box inside stone; segmentation/contact and rubbing are unverified.',
            'UNVERIFIED_LOCALIZED_MOTION':'Localized movement only; jewellery and rubbing are unverified.',
            'CONTACT_NOT_VISIBLE':'No usable jewellery contact, box, or localized movement in this pickup-to-return interval.'}[status]
    result=dict(rubbing_contact_status=status,rubbing_contact_reason=reason,
                rubbing_evidence_interval=[start,end],rubbing_contact_samples=samples,
                rubbing_verified=status in ('CONTACT_AND_MOTION_LIVE','CONTACT_AND_MOTION_RECOVERED'),
                rubbing_return_precedes_contact=return_precedes_contact,
                rubbing_recovery_errors=model_errors,
                _contact_region=region)
    if selected:
        ev=selected['ev'];frame=selected['frame'];picture=frame.copy()
        box=tuple(int(v) for v in ev['box'])
        mask=ev.get('mask') if ev.get('kind','SEGMENTATION')=='SEGMENTATION' else None
        if mask is not None:
            ox,oy=ev['origin']
            picture[oy:oy+mask.shape[0],ox:ox+mask.shape[1]][mask>0]=(0,215,255)
            result['rubbing_contact_mask_image']=str(cycle['directory']/'rubbing_contact_mask.png')
            cv2.imwrite(result['rubbing_contact_mask_image'],(mask>0).astype(np.uint8)*255)
            result['rubbing_contact_mask_origin']=[int(ox),int(oy)]
        cv2.rectangle(picture,box[:2],box[2:],(255,220,0),2)
        result['rubbing_contact_box']=list(box)
        result['rubbing_contact_stone']=[int(v) for v in selected['stone']]
        result['rubbing_contact_kind']=ev.get('kind','SEGMENTATION')
        x1,y1,x2,y2=map(int,selected['stone'])
        crop=picture[max(0,y1-15):min(frame.shape[0],y2+15),max(0,x1-15):min(frame.shape[1],x2+15)]
        source=selected['source_frame']
        if live is not None and selected['ev'] is live:
            # The journal is sampled and may not contain this exact live frame.
            # Persist the original source rather than linking a nearby sample.
            source=save_picture(cycle['directory'],'rubbing_contact_source',selected['frame'])
        result.update(rub_image=save_picture(cycle['directory'],'rubbing_contact',crop),
                      rub_timestamp=selected['timestamp'],rub_source_frame=source,
                      rub_source_timestamp=selected['timestamp'])
    elif entries and observations:
        entry=entries[0];ts=entry['timestamp']
        _,stone=min(observations,key=lambda p:abs(p[0]-ts))
        frame=journal.read(entry);x1,y1,x2,y2=map(int,stone)
        crop=frame[max(0,y1):min(frame.shape[0],y2),max(0,x1):min(frame.shape[1],x2)]
        if crop.size:
            result.update(rub_image=save_picture(cycle['directory'],'contact_not_visible',crop),
                          rub_timestamp=ts,rub_source_frame=entry['file'])
    return result
