"""Timestamped camera evidence and automatic reports, independent of inference."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from queue import Queue, Full
import json
import os
import threading
import time

import cv2


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)


def prediction_text(prediction):
    """Evidence warnings never replace the acid runtime's prediction."""
    if not prediction:
        return 'No acid result generated'
    label = {'gold_22k': '22K GOLD', 'non_gold': 'NON-GOLD',
             'invalid': 'INVALID (acid model)'}.get(prediction['prediction'], prediction['prediction'])
    return f"{label} ({prediction['confidence']:.1%})"


class FrameJournal:
    """Persist 15 FPS on a bounded writer queue; camera/model threads never wait.

    The original camera arrays are immutable. JPEG encoding and disk I/O happen
    only on the writer. Dropped samples and stream gaps remain explicit evidence.
    """
    def __init__(self, directory, fps=15, queue_size=45):
        self.directory = Path(directory)
        self.frames_dir = self.directory / 'frames'
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.queue = Queue(maxsize=queue_size)
        self.lock = threading.Lock()
        self.entries, self.problems = [], []
        self.last_offered = -float('inf')
        self.last_received = None
        self.closed = False
        self.thread = threading.Thread(target=self._write, name='evidence-recorder', daemon=True)
        self.thread.start()

    def offer(self, frame, timestamp):
        if self.closed or timestamp <= self.last_offered:
            return
        if self.last_received is not None and timestamp-self.last_received > .5:
            self.problem('camera_gap', self.last_received, timestamp)
        self.last_received = timestamp
        if timestamp-self.last_offered < 1/self.fps - .001:
            return
        self.last_offered = timestamp
        try:
            self.queue.put_nowait((timestamp, frame))
        except Full:
            self.problem('recording_queue_full', timestamp, timestamp)

    def problem(self, reason, start, end):
        with self.lock:
            if self.problems and self.problems[-1]['reason'] == reason and start-self.problems[-1]['end'] < .5:
                self.problems[-1]['end'] = end
                self.problems[-1]['count'] += 1
            else:
                self.problems.append(dict(reason=reason, start=start, end=end, count=1))

    def _write(self):
        with (self.directory/'frames.jsonl').open('a', encoding='utf-8') as index:
            while True:
                item = self.queue.get()
                try:
                    if item is None:
                        return
                    timestamp, frame = item
                    name = f'{timestamp:.6f}.jpg'
                    path = self.frames_dir/name
                    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise OSError('JPEG evidence write failed')
                    entry = dict(timestamp=timestamp, file=f'frames/{name}', shape=list(frame.shape))
                    with self.lock:
                        self.entries.append(entry)
                    index.write(json.dumps(entry)+'\n')
                    index.flush()
                except Exception as exc:
                    self.problem(f'recording_error: {exc}', item[0], item[0])
                finally:
                    self.queue.task_done()

    def between(self, start, end, fps=None):
        with self.lock:
            entries = [e.copy() for e in self.entries if start <= e['timestamp'] <= end]
        if fps is None:
            return entries
        selected, last = [], -float('inf')
        for entry in entries:
            if entry['timestamp']-last >= 1/fps-.001:
                selected.append(entry)
                last = entry['timestamp']
        return selected

    def read(self, entry):
        frame = cv2.imread(str(self.directory/entry['file']))
        if frame is None:
            raise OSError(f"Recorded frame unavailable: {entry['file']}")
        return frame

    def flush_through(self, timestamp, timeout=1.):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            with self.lock:
                latest = self.entries[-1]['timestamp'] if self.entries else -float('inf')
            if latest >= timestamp-1/self.fps or self.queue.unfinished_tasks == 0:
                return
            time.sleep(.01)

    def clip(self, start, end, output):
        """The timestamp sidecar is authoritative for irregular/dropped samples."""
        entries = self.between(start, end)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not entries:
            return dict(path=None, requested_start=start, requested_end=end, frames=0)
        first = self.read(entries[0])
        h, w = first.shape[:2]
        # Match playback to the observed time span when CPU/disk sampling is
        # below target FPS. Otherwise a seven-second capture plays too quickly.
        span = max(1/self.fps, min(end, entries[-1]['timestamp']+1/self.fps)-entries[0]['timestamp'])
        playback_fps = len(entries)/span
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'MJPG'), playback_fps, (w, h))
        if not writer.isOpened():
            raise OSError('Could not open evidence clip writer')
        written = []
        try:
            for entry in entries:
                image = self.read(entry)
                if image.shape != first.shape:
                    continue
                writer.write(image)
                written.append(entry['timestamp'])
        finally:
            writer.release()
        result = dict(path=str(output), requested_start=start, requested_end=end,
                      actual_start=written[0] if written else None,
                      actual_end=written[-1] if written else None,
                      frames=len(written), timestamps=written, playback_fps=playback_fps,
                      playback_seconds=len(written)/playback_fps)
        atomic_json(output.with_suffix('.json'), result)
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.put(None)
        self.thread.join()


def write_report(directory, jewels, tests, recording_problems=()):
    """Only actual tracker outputs: IDs, pictures, tracking, rub/mask and acid."""
    from html import escape
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

    directory = Path(directory)
    manifest = dict(schema='jewel_continuous_session_v1', generated=datetime.now().isoformat(),
                    jewels=jewels, tests=tests, recording_problems=list(recording_problems))
    atomic_json(directory/'session.json', manifest)
    styles = getSampleStyleSheet()
    story = []
    def text(value, style='BodyText'):
        story.append(Paragraph(escape(str(value)), styles[style]))
        story.append(Spacer(1, 7))
    def picture(path, caption):
        if not path or not Path(path).is_file():
            return
        width, height = ImageReader(str(path)).getSize()
        scale = min(480/width, 230/height, 1.)
        story.append(Image(str(path), width=width*scale, height=height*scale))
        text(caption)
    text('Jewel Tracking and Acid Test Report', 'Title')
    text(f"Session: {directory.name}")
    text(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    text(f'Jewels tracked: {len(jewels)} | Test records: {len(tests)}')
    rows = [['Jewel', 'Acid model result', 'Rubbing / mask evidence']]
    for jewel in jewels:
        records = [r for r in tests if r.get('jewel_id') == jewel['id']]
        record = records[-1] if records else {}
        rows.append([f"J{jewel['id']}", Paragraph(escape(prediction_text(record.get('prediction'))), styles['BodyText']),
                     Paragraph(escape(f"{record.get('rubbing_status', 'No test recorded')} / {record.get('mask_status', 'No mask recorded')}"), styles['BodyText'])])
    for record in tests:
        if record.get('jewel_id') is None:
            rows.append(['Unassigned', Paragraph(escape(prediction_text(record.get('prediction'))), styles['BodyText']),
                         'Tracking missed / identity unresolved'])
    table = Table(rows, colWidths=[58, 192, 230], repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0f2fe')),
                              ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('GRID', (0, 0), (-1, -1), .3, colors.lightgrey),
                              ('BOTTOMPADDING', (0, 0), (-1, -1), 9)]))
    story.append(table)
    text('Acid model output and process evidence are reported separately. Missing rubbing or tracking evidence does not change the acid model prediction.')
    for jewel in jewels:
        picture(jewel.get('image'), f"J{jewel['id']} at initial capture")
        for interval in jewel.get('visibility', []):
            if interval['status'] == 'MISSED':
                text(f"J{jewel['id']} tracking missed for {max(0, interval['end']-interval['start']):.2f} seconds.")
                picture(interval.get('image'), 'Frame when jewel tracking was missed')
    for record in tests:
        story.append(PageBreak())
        jid = record.get('jewel_id')
        text(f"J{jid} test" if jid is not None else 'Test with missed jewel identity', 'Heading1')
        text('Acid result: '+prediction_text(record.get('prediction')), 'Heading2')
        prediction = record.get('prediction') or {}
        if prediction.get('invalid_reason'):
            text('INVALID reason: '+prediction['invalid_reason'])
        for field in ('before_status', 'rubbing_status', 'rubbing_contact_status', 'rubbing_contact_reason',
                      'mask_status', 'mask_reason', 'fresh_acid_status', 'acid_status', 'acid_target_status', 'identity_status', 'acid_jewel_association', 'ph_status'):
            if record.get(field):
                text(f"{field.replace('_', ' ').capitalize()}: {record[field]}")
        if record.get('acid_new_mask_overlap') is not None:
            text(f"Acid/new-rub overlap: {record['acid_new_mask_overlap']:.1%} of the smaller region (50% threshold).")
        if record.get('acid_target_status') == 'ACID_AT_DIFFERENT_LOCATION':
            text('New rubbing was detected in one area and acid was detected elsewhere. Application to this jewel\'s rubbing mark was not verified. The model result above is retained from the acid location.')
        elif record.get('acid_target_status') == 'COMPARISON_UNAVAILABLE':
            text('Rubbing mask unavailable; acid location could not be compared with the new rubbing.')
        if record.get('rub_previous_acid_overlap', 0) > 0:
            text(f"New rubbing overlaps the previous acid footprint by {record['rub_previous_acid_overlap']:.1%}; contamination possible.")
        rejected = record.get('fresh_acid_rejections', [])
        if rejected:
            text(f'{len(rejected)} saved observations did not confirm a fresh application. Rejection reasons: '+
                 ', '.join(sorted({r['reason'] for r in rejected})))
        if record.get('rub_timestamp') is not None:
            text(f"Stone/jewel evidence source: {record.get('rub_source_frame')} at "
                 f"{datetime.fromtimestamp(record['rub_timestamp']):%Y-%m-%d %H:%M:%S.%f}; selected only from this jewel's pickup-to-return interval.")
        for name in ('before_timestamp', 'started', 'returned', 'mask_decision_at', 'baseline_at', 'acid_onset', 'finished'):
            if record.get(name) is not None:
                text(f"{name.replace('_', ' ').capitalize()}: {datetime.fromtimestamp(record[name]):%Y-%m-%d %H:%M:%S.%f}")
        for issue in record.get('issues', []):
            text('Evidence note: '+issue)
        for interval in record.get('visibility', []):
            text(f"J{interval['jewel_id']} {interval['status']}: {max(0, interval['end']-interval['start']):.2f} seconds "
                 f"({datetime.fromtimestamp(interval['start']):%H:%M:%S.%f} to {datetime.fromtimestamp(interval['end']):%H:%M:%S.%f})")
        for key, caption in [('before_image', 'Before stone / pre-rubbing reference'),
                             ('pickup_image', 'Jewel visibility during pickup/rubbing'),
                             ('rub_image', 'Stone/jewel contact evidence: '+record.get('rubbing_contact_status', 'Not verified')),
                             ('mask_image', 'New rubbed region'),
                             ('baseline_start_image', 'Baseline stage started after the mask decision'),
                             ('baseline_image', 'Pre-application evidence'),
                             ('acid_detection_image', 'Acid detected in recorded camera frame'),
                             ('acid_rub_comparison_image', 'Same stone frame: magenta = new rubbing; cyan = acid detector footprint'),
                             ('acid_image', 'Acid/reaction evidence')]:
            picture(record.get(key), caption)
        for key in ('rubbing_clip', 'acid_clip'):
            if record.get(key):
                clip = record[key]
                text(f"{key.replace('_', ' ').capitalize()}: {clip.get('path') or 'No recorded frames'}")
    for problem in recording_problems:
        text(f"Recording: {problem['reason']} ({problem['end']-problem['start']:.2f} seconds; {problem['count']} observations)")
    temp = directory/'report.tmp.pdf'
    SimpleDocTemplate(str(temp), title='Jewel Tracking and Acid Test Report').build(story)
    os.replace(temp, directory/'report.pdf')
    return str(directory/'report.pdf')
