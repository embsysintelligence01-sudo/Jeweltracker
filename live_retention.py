"""Durable per-jewel cleanup while the camera continues to record.

Encoding never runs on the camera, inference, report, or recovery threads.
Only immutable, completely exported tests enter this queue. Raw camera frames
remain protected by FrameJournal pins until publication and report verification.
"""
from contextlib import nullcontext
from pathlib import Path
import copy
import json
import logging
import os
import re
import shutil
import threading
import time
import traceback

from evidence_archive import (SessionGuard, checked_local, digest, durable_json,
                              encoder_available, report_guard, sync_directory,
                              tree_bytes, validate_pdf)
from simple_session import encode_budgeted, _publish_and_clean, _validate_files

LOG = logging.getLogger(__name__)
STATES = '.live-retention'


def sync_file(path):
    with Path(path).open('rb') as handle:
        os.fsync(handle.fileno())


def job_path(root, cycle):
    if not isinstance(cycle, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', cycle):
        raise ValueError('Invalid completed-test identifier')
    return Path(root) / STATES / (cycle + '.json')


def compact_records(root, records):
    """Overlay durable publications so stale GUI/report jobs cannot restore raw paths."""
    result = []
    for record in records:
        cycle = record.get('cycle_id')
        path = job_path(root, cycle) if cycle else None
        state = json.loads(path.read_text()) if path and path.is_file() else {}
        if state.get('published') or state.get('status') in ('PUBLISHED', 'CLEANED'):
            record = state['record']
        result.append(copy.deepcopy(record))
    return result


def enqueue_test(root, record, jewel):
    path = job_path(root, record['cycle_id'])
    if path.is_file():
        return  # One immutable terminal export per cycle; retries use its snapshot.
    if any(str(note).startswith(('Recovery/save error:', 'Prediction archive write failed:'))
           for note in record.get('issues', [])):
        return  # The export failed; never turn it into a successful cleanup.
    durable_json(path, dict(status='QUEUED', attempts=0, next_retry=0,
                            record=copy.deepcopy(record), jewel=copy.deepcopy(jewel),
                            updated_at=time.time()))


def _state(path, state, status, **changes):
    state.update(status=status, updated_at=time.time(), **changes)
    durable_json(path, state)


def compact_test(root, path, policy, release=None):
    """Encode outside report locks; atomically publish all three files before cleanup."""
    if policy.require_server_ack or not policy.cleanup_raw_after_archive:
        raise ValueError('Automatic local cleanup is disabled by the storage policy')
    root, path = Path(root).resolve(), Path(path)
    state = json.loads(path.read_text())
    cycle = state['record']['cycle_id']
    job_path(root, cycle)  # Validate names read from a durable job.
    original = state.get('original_record', state['record'])
    jid = original.get('jewel_id')
    if not isinstance(jid, int) or isinstance(jid, bool) or jid < 0:
        raise ValueError('Unassigned jewel; raw evidence retained')
    final = root / f'J{jid}' / cycle
    stage = root / STATES / (cycle + '.staging')
    if state['status'] == 'CLEANED':
        if release:
            release(cycle)
        return state['record']
    if not state.get('files'):
        _state(path, state, 'ENCODING', attempts=state.get('attempts', 0) + 1,
               original_record=original, error=None, traceback=None)
        exe = encoder_available()
        if not exe:
            raise OSError('FFmpeg/libx264 unavailable')
        sources = []
        from compact_evidence import media_info
        for key, filename in (('rubbing_clip', 'rubbing.avi'), ('acid_clip', 'acid.avi')):
            clip = original.get(key)
            if not clip or not clip.get('path'):
                raise OSError('Missing ' + key + '; raw evidence retained')
            source = checked_local(root, clip['path'])
            sources.append((key, filename, source, media_info(source)))
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        record = copy.deepcopy(original)
        for key, filename, _, _ in sources:
            record[key] = dict(record[key], path=str(final / filename))
            (stage / filename).touch()
        record.pop('acid_candidate_clip', None)
        record.pop('compact_report', None)
        from audit_report import render_report
        from session_evidence import prediction_text

        def resolve(value):
            target = checked_local(root, value)
            if target.parent == final:
                return stage / target.name
            return target

        # Relative buttons in this PDF resolve from its final per-jewel folder.
        render_report(stage / 'report.pdf', final, [state['jewel']], [record], [],
                      prediction_text, asset_resolver=resolve, simple_output=True)
        validate_pdf(stage / 'report.pdf')
        # Reserve space for the session index and live bookkeeping. A jewel is
        # normally tested once; existing attempts count towards the same cap.
        used = tree_bytes(final.parent)
        available = policy.target_test_bytes - used - (stage / 'report.pdf').stat().st_size - 131072
        duration = sum(info['duration_seconds'] for _, _, _, info in sources)
        for key, filename, source, info in sources:
            _state(path, state, 'ENCODING', current_clip=filename)
            encode_budgeted(source, stage / filename,
                            int(available * info['duration_seconds'] / duration), policy, exe)
            # Preserve exact timing metadata but identify it as compressed media.
            record[key]['archive_media'] = media_info(stage / filename)
        files = {str((final / p.name).relative_to(root)): dict(bytes=p.stat().st_size, sha256=digest(p))
                 for p in stage.iterdir() if p.is_file()}
        if final.exists():
            raise OSError('Untracked jewel output already exists; raw evidence retained')
        for p in stage.iterdir():
            if p.suffix == '.avi':
                media_info(p)
            sync_file(p)
        sync_directory(stage)
        if tree_bytes(stage) + used + 131072 > policy.target_test_bytes:
            raise OSError('Completed jewel exceeds 3 MB budget; raw evidence retained')
        record['compact_report'] = str(final / 'report.pdf')
        _state(path, state, 'READY', files=files, record=record, current_clip=None)
    # A READY job can be resumed even if the process died immediately after rename.
    if not final.exists():
        for relative, item in state['files'].items():
            candidate = stage / Path(relative).name
            if not candidate.is_file() or digest(candidate) != item['sha256']:
                raise OSError('Staged jewel verification failed; raw evidence retained')
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, final)
        sync_directory(final.parent)
        sync_directory(stage.parent)
    _validate_files(root, dict(files=state['files']))
    with report_guard(root):
        _state(path, state, 'PUBLISHED', published=True)
        # Publishing a refreshed session PDF/JSON is mandatory before raw deletion.
        from session_evidence import _write_report
        session_path = root / 'session.json'
        if session_path.is_file():
            data = json.loads(session_path.read_text())
            tests = data['tests']
            if not any(r.get('cycle_id') == cycle for r in tests):
                tests.append(original)
            jewels = data['jewels']
            problems = data.get('recording_problems', [])
        else:
            tests, jewels, problems = [original], [state['jewel']], []
        _write_report(root, jewels, tests, problems)
        validate_pdf(root / 'report.pdf')
        sync_file(root / 'report.pdf')
        sync_file(root / 'session.json')
        sync_directory(root)
        # Delete only this cycle's scratch directory. Other cycles/readers use
        # the journal's pins/leases, so shared frames are never removed here.
        raw = checked_local(root, original['record_path']).parent
        if raw.parent != root or not raw.name.startswith('test_'):
            raise ValueError('Unexpected raw test folder; no deletion authorized')
        if raw.exists():
            from acid_review_queue import preserve_review_sample
            preserve_review_sample(raw / 'manifest.json')
            shutil.rmtree(raw)
        _state(path, state, 'CLEANED', error=None)
    if release:
        release(cycle)
    return state['record']


def finalize_live_session(root, policy):
    """Called only after all camera/recovery/report/maintenance workers have drained."""
    if policy.require_server_ack or not policy.cleanup_raw_after_archive:
        raise ValueError('Automatic local cleanup is disabled by the storage policy')
    root = Path(root).resolve()
    with report_guard(root):
        receipt_dir = root / '.simple-staging'
        receipt = receipt_dir / 'commit.json'
        if receipt.is_file():
            return _publish_and_clean(root, receipt_dir, json.loads(receipt.read_text()))
        data = json.loads((root / 'session.json').read_text())
        tests = compact_records(root, data['tests'])
        if any(not r.get('compact_report') for r in tests):
            raise OSError('Some jewels are still awaiting compression; saved jobs will retry on next launch')
        if any(p.get('provisional_clip') for p in data.get('recording_problems', [])):
            raise OSError('Unassigned video evidence present; originals retained')
        files = {'report.pdf': dict(bytes=(root / 'report.pdf').stat().st_size, sha256=digest(root / 'report.pdf'))}
        for record in tests:
            state = json.loads(job_path(root, record['cycle_id']).read_text())
            _validate_files(root, dict(files=state['files']))
            files.update(state['files'])
        groups = {Path(name).parts[0] for name in files if name != 'report.pdf'}
        common = (files['report.pdf']['bytes'] + max(1, len(groups)) - 1) // max(1, len(groups))
        sizes = [dict(jewel=group, bytes=common + sum(item['bytes'] for name, item in files.items()
                                                   if Path(name).parts[0] == group),
                      limit_bytes=policy.target_test_bytes) for group in sorted(groups)]
        if any(item['bytes'] > policy.target_test_bytes for item in sizes):
            raise OSError('Final session index pushes a jewel above the size limit; working data retained')
        if receipt_dir.exists():
            shutil.rmtree(receipt_dir)
        receipt_dir.mkdir()
        commit = dict(files=files, previous_bytes=tree_bytes(root), jewel_sizes=sizes)
        durable_json(receipt, commit)
        return _publish_and_clean(root, receipt_dir, commit)


class LiveRetention(threading.Thread):
    """One sequential low-priority encoder, with durable retry deadlines and restart recovery."""
    def __init__(self, root, policy, journal, emit):
        super().__init__(name='jewel-storage', daemon=False)
        self.root = Path(root).resolve()
        self.policy, self.journal, self.emit = policy, journal, emit
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.last_scan = 0
        self.roots = [self.root]
        self.finished_jobs = set()

    def submit(self, record, jewel):
        enqueue_test(self.root, record, jewel)
        self.wake.set()

    def close(self):
        self.stopping.set()
        self.wake.set()
        self.join()

    def process_pending(self, force=False):
        # Discover only sessions with saved maintenance jobs, never arbitrary
        # historical data. A SessionGuard prevents touching another live app.
        if time.monotonic() - self.last_scan >= 30:
            saved = {p.parent for pattern in ('*/' + STATES, '*/.simple-staging')
                     for p in self.root.parent.glob(pattern) if p.parent != self.root}
            self.roots = [self.root] + sorted(saved)
            self.last_scan = time.monotonic()
        for root in self.roots:
            try:
                guard = nullcontext() if root == self.root else SessionGuard(root)
                with guard:
                    for path in sorted((root / STATES).glob('*.json')):
                        if path in self.finished_jobs:
                            continue
                        state = json.loads(path.read_text())
                        if state.get('status') == 'CLEANED':
                            if root == self.root:
                                self.journal.release_interval(state['record']['cycle_id'])
                            self.finished_jobs.add(path)
                            continue
                        if not force and state.get('next_retry', 0) > time.time():
                            continue
                        cycle = state['record']['cycle_id']
                        try:
                            self.emit('storage', status='ENCODING', message=f'Compressing J{state["record"].get("jewel_id")}; camera continues.')
                            compact_test(root, path, self.policy,
                                         self.journal.release_interval if root == self.root else None)
                            self.emit('report', path=str(root / 'report.pdf')) if root == self.root else None
                            self.emit('storage', status='CLEANED', message=f'J{state["record"].get("jewel_id")}: report and both AVI clips saved; temporary data released.')
                            self.finished_jobs.add(path)
                        except Exception as exc:
                            # Reload to retain any READY/PUBLISHED commit written before failure.
                            state = json.loads(path.read_text())
                            tries = state.get('attempts', 0)
                            _state(path, state, 'FAILED', error=str(exc),
                                   traceback=traceback.format_exc(),
                                   next_retry=time.time() + min(300, 5 * 2 ** min(tries, 6)))
                            LOG.exception('Jewel storage failed: %s / %s', root.name, cycle)
                            self.emit('storage', status='FAILED', message=f'Storage cleanup will retry; originals retained: {exc}')
                    if (root != self.root and not (root / 'session_closed.json').exists()
                            and (root / STATES).is_dir() and (root / 'session.json').is_file()):
                        # The other process has exited (we hold its recording lock).
                        # A killed idle session can close automatically once every
                        # test is published. Any unfinished test/spool keeps its raw
                        # footage intact for recovery rather than guessing coverage.
                        pending_raw = any(root.glob('test_*'))
                        spool = root / 'deferred_recovery'
                        pending_spool = spool.is_dir() and any(spool.iterdir())
                        data = json.loads((root / 'session.json').read_text())
                        records = compact_records(root, data['tests'])
                        if (not pending_raw and not pending_spool and records
                                and all(r.get('compact_report') for r in records)
                                and not any(p.get('provisional_clip') for p in data.get('recording_problems', []))):
                            durable_json(root / 'session_closed.json',
                                         dict(closed_at=time.time(), recovered_after_process_exit=True))
                    receipt = root / '.simple-staging/commit.json'
                    if root != self.root and ((root / 'session_closed.json').exists() or receipt.is_file()):
                        try:
                            if (root / STATES).exists() or receipt.is_file():
                                finalize_live_session(root, self.policy)
                            elif (root / '.simple-staging').exists():
                                # Recover the old shutdown-only implementation too.
                                error_path = root / 'storage_error.json'
                                previous = json.loads(error_path.read_text()) if error_path.exists() else {}
                                if not force and previous.get('next_retry', 0) > time.time():
                                    continue
                                from simple_session import finalize_simple_session
                                finalize_simple_session(root, self.policy, session_guard=guard)
                        except Exception as exc:
                            # Per-test errors remain in their job files. This also
                            # covers interrupted legacy shutdown-only migrations.
                            durable_json(root / 'storage_error.json', dict(error=str(exc), updated_at=time.time(),
                                                                         traceback=traceback.format_exc(),
                                                                         next_retry=time.time() + 60))
            except BlockingIOError:
                continue

    def run(self):
        while not self.stopping.is_set():
            try:
                self.process_pending()
            except Exception:
                LOG.exception('Storage maintenance scan failed')
            self.wake.wait(1)
            self.wake.clear()
        # Drain queued work once on normal shutdown; failures stay on disk for
        # next launch instead of keeping the GUI alive with endless retries.
        self.process_pending(force=True)
