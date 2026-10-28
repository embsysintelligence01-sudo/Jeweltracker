"""Select stone/jewel contact pictures only inside this jewel's rub interval."""
import cv2
import numpy as np


def recover_contact(journal, cycle, observer, save_picture):
    start = cycle['started']
    end = min(cycle.get('returned') or cycle['finished'],
              cycle['acid_onset']-.05 if cycle.get('acid_onset') is not None else cycle['finished'])
    entries = journal.between(start, end, fps=6)
    # Bound model work on long missed/occluded intervals. Live inference keeps
    # priority in the shared model dispatcher.
    indices = np.unique(np.linspace(0, len(entries)-1, min(24, len(entries)), dtype=int))
    samples, selected, best_area, previous = [], None, 0, None
    motions = []
    confirmed = False
    observations = cycle.get('stone_observations', [])
    for index in indices:
        entry = entries[index]
        timestamp = entry['timestamp']
        if not observations or observer is None:
            continue
        seen, stone = min(observations, key=lambda p: abs(p[0]-timestamp))
        if abs(seen-timestamp) > .4:
            continue
        frame = journal.read(entry)
        ev = observer(frame, stone)
        if ev is None:
            previous = None
            motions.clear()
            continue
        area = np.count_nonzero(ev['mask'])
        samples.append(dict(timestamp=timestamp, source_frame=entry['file'],
                            contact_bbox=[int(v) for v in ev['box']]))
        size = np.maximum(np.asarray(stone[2:])-stone[:2], 1)
        point = (np.asarray(ev['c'])-stone[:2])/size*256
        if previous is not None and 0 < timestamp-previous[0] <= .45:
            motions.append(point-previous[1])
            motions = motions[-6:]
            distances = [np.linalg.norm(m) for m in motions]
            reversals = any(np.dot(p,q) < -1 for p,q in zip(motions, motions[1:]))
            if len(distances) >= 3 and max(distances) >= 2.5 and sum(distances) >= 6 and reversals:
                confirmed = True
        previous = timestamp, point
        if area > best_area:
            best_area = area
            picture = frame.copy()
            ox, oy = ev['origin']
            region = picture[oy:oy+ev['mask'].shape[0], ox:ox+ev['mask'].shape[1]]
            region[ev['mask'] > 0] = (0,215,255)
            x1,y1,x2,y2 = map(int, stone)
            h,w = frame.shape[:2]
            crop = picture[max(0,y1-15):min(h,y2+15), max(0,x1-15):min(w,x2+15)]
            if crop.size:
                selected = dict(image=crop.copy(), timestamp=timestamp, source_frame=entry['file'])
    result = dict(rubbing_contact_status='CONTACT_AND_MOTION_RECOVERED' if confirmed else
                  'GOLD_CONTACT_VISIBLE' if selected else 'CONTACT_NOT_VISIBLE',
                  rubbing_evidence_interval=[start,end], rubbing_contact_samples=samples,
                  rubbing_contact_reason='Gold-model overlap and back-and-forth movement on the stone.' if confirmed else
                  'Gold contact is visible; rubbing movement was not confirmed.' if selected else
                  'No usable stone/jewel contact frame was found in this jewel\'s pickup-to-return interval.')
    if selected:
        result.update(rub_image=save_picture(cycle['directory'], 'stone_gold_contact', selected['image']),
                      rub_timestamp=selected['timestamp'], rub_source_frame=selected['source_frame'])
    else:
        # A context photo is explicitly labelled NOT visible; never substitute
        # an unrelated frame, a later acid frame, or call return a rubbing image.
        for entry in entries:
            timestamp = entry['timestamp']
            if observations:
                seen, stone = min(observations, key=lambda p: abs(p[0]-timestamp))
                if abs(seen-timestamp) <= .4:
                    frame = journal.read(entry)
                    x1,y1,x2,y2 = map(int, stone)
                    crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                    if crop.size:
                        result.update(rub_image=save_picture(cycle['directory'], 'stone_contact_not_visible', crop),
                                      rub_timestamp=timestamp, rub_source_frame=entry['file'])
                        break
    return result
