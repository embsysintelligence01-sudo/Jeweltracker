"""Timestamped camera evidence and automatic reports, independent of inference."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from queue import Queue, Full
import json
import os
import threading
from contextlib import contextmanager
import shutil
import uuid
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
    def __init__(self, directory, fps=15, queue_size=45, policy=None):
        self.directory = Path(directory)
        self.frames_dir = self.directory / 'frames'
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.queue = Queue(maxsize=queue_size)
        self.lock = threading.RLock()
        self.offer_lock = threading.Lock()
        self.policy = policy
        self.pins, self.leases = {}, {}
        self.watermark = None
        self.storage_status = {}
        self.pruned_frames = 0
        self._last_maintenance = -1e9
        self.entries, self.problems = [], []
        self.last_offered = -float('inf')
        self.last_received = None
        self.closed = False
        self.thread = threading.Thread(target=self._write, name='evidence-recorder', daemon=True)
        self.thread.start()

    def offer(self, frame, timestamp):
        # Constant-time producer path; all disk/retention work is on the writer.
        with self.offer_lock:
            if self.closed or timestamp <= self.last_offered:return
            if self.last_received is not None and timestamp-self.last_received > .5:
                self.problem('camera_gap', self.last_received, timestamp)
            self.last_received = timestamp
            if timestamp-self.last_offered < 1/self.fps - .001:return
            self.last_offered = timestamp
            try:self.queue.put_nowait((timestamp, frame))
            except Full:self.problem('recording_queue_full', timestamp, timestamp)

    def processed_through(self,timestamp):
        with self.lock:self.watermark=timestamp

    def pin_interval(self,token,start,end=None):
        with self.lock:self.pins[token]=(start,end)

    def finish_interval(self,token,end):
        with self.lock:
            if token in self.pins:self.pins[token]=(self.pins[token][0],end+.5)

    @contextmanager
    def lease(self,start,end):
        token=uuid.uuid4().hex
        with self.lock:self.leases[token]=(start,end)
        try:yield
        finally:
            with self.lock:self.leases.pop(token,None)

    def _maintenance(self,timestamp):
        if self.policy is None or not self.policy.enabled or timestamp-self._last_maintenance<1:return
        self._last_maintenance=timestamp
        free=shutil.disk_usage(self.directory).free
        self.storage_status=dict(free_bytes=free,low_disk=free<self.policy.disk_warning_free_gib*1024**3,
                                 pruned_idle_frames=self.pruned_frames)
        cutoff=timestamp-self.policy.idle_buffer_seconds
        with self.lock:
            if self.watermark is not None:cutoff=min(cutoff,self.watermark-self.policy.pre_roll_seconds)
            protected=list(self.pins.values())+list(self.leases.values())
            keep=[]
            for entry in self.entries:
                ts=entry['timestamp']
                if ts>=cutoff or any(lo<=ts and (hi is None or ts<=hi) for lo,hi in protected):
                    keep.append(entry);continue
                path=self.directory/entry['file']
                if path.is_symlink():raise OSError('Refusing to prune symlink frame')
                path.unlink(missing_ok=True);self.pruned_frames+=1
            changed=len(keep)!=len(self.entries)
            self.entries=keep
            if not changed:return False
            # The temporary catalogue is compacted with the buffer; it cannot
            # grow for days while idle. Active interval entries remain intact.
            temp=self.directory/'frames.pruning.tmp'
            with temp.open('w') as handle:
                for entry in keep:handle.write(json.dumps(entry)+'\n')
            os.replace(temp,self.directory/'frames.jsonl')
            return True

    def problem(self, reason, start, end):
        with self.lock:
            if self.problems and self.problems[-1]['reason'] == reason and start-self.problems[-1]['end'] < .5:
                self.problems[-1]['end'] = end
                self.problems[-1]['count'] += 1
            else:
                self.problems.append(dict(reason=reason, start=start, end=end, count=1))

    def _write(self):
        index=(self.directory/'frames.jsonl').open('a',encoding='utf-8')
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is None:return
                    timestamp, frame = item
                    name = f'{timestamp:.6f}.jpg'
                    path = self.frames_dir/name
                    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise OSError('JPEG evidence write failed')
                    entry = dict(timestamp=timestamp, file=f'frames/{name}', shape=list(frame.shape))
                    with self.lock:self.entries.append(entry)
                    index.write(json.dumps(entry)+'\n');index.flush()
                    if self._maintenance(timestamp):
                        index.close()
                        index=(self.directory/'frames.jsonl').open('a',encoding='utf-8')
                except Exception as exc:
                    self.problem(f'recording_error: {exc}', item[0], item[0])
                finally:self.queue.task_done()
        finally:index.close()

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
        with self.lock:frame = cv2.imread(str(self.directory/entry['file']))
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

    def clip(self,start,end,output):
        with self.lease(start,end):return self._clip(start,end,output)

    def _clip(self, start, end, output):
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
                    raise OSError('Evidence frame dimensions changed; clip incomplete')
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
        with self.offer_lock:
            send_stop=not self.closed
            self.closed=True
        if send_stop:self.queue.put(None)
        self.thread.join()


def write_report(directory, jewels, tests, recording_problems=()):
    """Only actual tracker outputs: IDs, pictures, tracking, rub/mask and acid."""
    from evidence_archive import report_guard
    with report_guard(directory):
        if (Path(directory)/'archive_commit.json').is_file():
            from evidence_archive import ArchiveManager
            ArchiveManager(directory).validate_commit()
            return str(Path(directory)/'report.pdf')
        return _write_report(directory,jewels,tests,recording_problems)


def _write_report(directory,jewels,tests,recording_problems=()):
    from audit_report import render_report
    directory=Path(directory)
    manifest=dict(schema='jewel_continuous_session_v1',generated=datetime.now().isoformat(),
                  jewels=jewels,tests=tests,recording_problems=list(recording_problems))
    atomic_json(directory/'session.json',manifest)
    temp=directory/'report.tmp.pdf'
    render_report(temp,directory,jewels,tests,recording_problems,prediction_text)
    os.replace(temp,directory/'report.pdf')
    return str(directory/'report.pdf')
