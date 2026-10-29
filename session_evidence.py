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
    from audit_report import render_report
    directory=Path(directory)
    manifest=dict(schema='jewel_continuous_session_v1',generated=datetime.now().isoformat(),
                  jewels=jewels,tests=tests,recording_problems=list(recording_problems))
    atomic_json(directory/'session.json',manifest)
    temp=directory/'report.tmp.pdf'
    render_report(temp,directory,jewels,tests,recording_problems,prediction_text)
    os.replace(temp,directory/'report.pdf')
    return str(directory/'report.pdf')
