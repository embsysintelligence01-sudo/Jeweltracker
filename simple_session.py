"""Publish a portable PDF and two full-duration clips per test, then remove scratch data."""
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import shutil

from compact_evidence import media_info, run_encoder
from evidence_archive import (RetentionPolicy, SessionGuard, checked_local, digest,
                              durable_json, encoder_available, report_guard,
                              sync_directory, tree_bytes, validate_pdf)


def encode_budgeted(source, output, budget, policy, exe):
    """Two-pass H.264, full frame and duration; the file size is checked before use."""
    info = media_info(source)
    if budget < 50_000:
        raise OSError('PDF leaves insufficient space for both clips; originals retained')
    scale = min(1., 1280 / max(info['width'], info['height']))
    width = max(2, int(info['width'] * scale) // 2 * 2)
    height = max(2, int(info['height'] * scale) // 2 * 2)
    fps = min(10., info['fps'])
    filters = f'scale={width}:{height}:flags=lanczos,setsar=1,fps={fps:.9f}'
    bitrate = int(budget * .92 * 8 / info['duration_seconds'])
    log = str(output) + '.pass'
    try:
        for _ in range(3):
            for pass_number in (1, 2):
                run_encoder([exe, '-nostdin', '-v', 'error', '-y', '-threads', '1',
                    '-i', str(source), '-an', '-vf', filters, '-c:v', 'libx264',
                    '-preset', policy.h264_preset, '-b:v', str(bitrate),
                    '-threads', str(policy.encoder_threads), '-filter_threads', '1',
                    '-pix_fmt', 'yuv420p', '-pass', str(pass_number), '-passlogfile', log,
                    *(['-f', 'null', os.devnull] if pass_number == 1 else
                      ['-vtag', 'H264', '-f', 'avi', str(output)])])
            encoded = media_info(output)  # Decode the entire output before accepting it.
            if abs(encoded['duration_seconds'] - info['duration_seconds']) > max(.25, 2 / fps):
                raise OSError('Compressed clip duration changed; originals retained')
            if output.stat().st_size <= budget:
                return
            bitrate = int(bitrate * budget / output.stat().st_size * .9)
        raise OSError('Clip could not fit its size budget; originals retained')
    finally:
        for path in output.parent.glob(output.name + '.pass*'):
            path.unlink()


def _validate_files(root, commit, stage=None):
    for relative, expected in commit['files'].items():
        path = checked_local(root, relative)
        if stage is not None and (stage / relative).is_file():
            path = stage / relative
        if not path.is_file() or path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise OSError('Simple session verification failed: ' + relative)
        if path.suffix == '.pdf':
            validate_pdf(path)
        else:
            media_info(path)


def _publish_and_clean(root, stage, commit):
    # A durable staged receipt permits resuming after any rename or deletion.
    _validate_files(root, commit, stage)
    for relative in commit['files']:
        source = stage / relative
        if source.is_file():
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            sync_directory(destination.parent)
    _validate_files(root, commit)
    # Retain review frames before session scratch files are removed.
    from acid_review_queue import preserve_review_sample
    for manifest in root.glob('test_*/manifest.json'):
        preserve_review_sample(manifest)
    keep = {Path(name).parts[0] for name in commit['files']}
    keep.update({stage.name, '.recording.lock', '.evidence.lock', 'pending_review', 'packet_sealing'})
    for path in root.iterdir():
        if path.name not in keep:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    # Only canonical output files may remain inside the jewel directories.
    for name in keep - {stage.name, '.recording.lock', '.evidence.lock', 'report.pdf', 'pending_review', 'packet_sealing'}:
        folder = root / name
        for path in sorted(folder.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if path.is_file() and path.relative_to(root).as_posix() not in commit['files']:
                path.unlink()
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    shutil.rmtree(stage)
    sync_directory(root)
    return dict(status='cleaned', previous_bytes=commit['previous_bytes'],
                final_bytes=tree_bytes(root), jewel_sizes=commit['jewel_sizes'])


def finalize_simple_session(directory, policy=None, session_guard=None):
    """Caller must drain all writers first. Failed validation never deletes sources."""
    root = Path(directory).resolve()
    policy = policy or RetentionPolicy.load()
    if policy.require_server_ack:
        raise ValueError('Simple output requires the explicit local-cleanup policy')
    with (nullcontext(session_guard) if session_guard is not None else SessionGuard(root)):
        with report_guard(root):
            stage = root / '.simple-staging'
            receipt = stage / 'commit.json'
            if receipt.is_file():
                return _publish_and_clean(root, stage, json.loads(receipt.read_text()))
            if not (root / 'session.json').exists() and (root / 'report.pdf').is_file():
                validate_pdf(root / 'report.pdf')
                return dict(status='already_cleaned', final_bytes=tree_bytes(root))
            if not (root / 'session_closed.json').is_file():
                raise ValueError('Session is still open; originals retained')
            validate_pdf(root / 'report.pdf')
            data = copy.deepcopy(json.loads((root / 'session.json').read_text()))
            tests = data['tests']
            # Unassigned camera events need review, rather than silent deletion.
            if any(p.get('provisional_clip') for p in data.get('recording_problems', [])):
                raise ValueError('Unassigned video evidence present; originals retained')
            exe = encoder_available()
            if tests and not exe:
                raise OSError('FFmpeg/libx264 required; originals retained')
            previous_bytes = tree_bytes(root)
            if stage.exists():
                shutil.rmtree(stage)  # Incomplete pre-publication encoding, never originals.
            stage.mkdir()
            counts = Counter(t.get('jewel_id') for t in tests)
            seen = Counter()
            clips = defaultdict(list)
            for index, record in enumerate(tests, 1):
                jid = record.get('jewel_id')
                if jid is not None and (not isinstance(jid, int) or isinstance(jid, bool) or jid < 0):
                    raise ValueError('Invalid jewel ID')
                group = f'J{jid}' if jid is not None else f'Unassigned_{index}'
                seen[jid] += 1
                folder = Path(group)
                if counts[jid] > 1:
                    folder /= f'test_{seen[jid]}'
                (stage / folder).mkdir(parents=True)
                for key, filename in (('rubbing_clip', 'rubbing.avi'), ('acid_clip', 'acid.avi')):
                    metadata = record.get(key)
                    if not metadata or not metadata.get('path'):
                        raise ValueError(f'{group} has no {key}; originals retained')
                    source = checked_local(root, metadata['path'])
                    relative = folder / filename
                    duration = media_info(source)['duration_seconds']
                    clips[group].append((source, relative, duration))
                    record[key] = dict(metadata, path=str(root / relative))
                record.pop('acid_candidate_clip', None)

            from audit_report import render_report
            from session_evidence import prediction_text

            def resolve(path):
                target = checked_local(root, path)
                candidate = stage / target.relative_to(root)
                # Let PDF buttons target the final names while images still come
                # from originals. Videos are encoded next; placeholders are temporary.
                return candidate if candidate.exists() else target

            for items in clips.values():
                for _, relative, _ in items:
                    (stage / relative).touch()
            render_report(stage / 'report.pdf', root, data['jewels'], tests,
                          data.get('recording_problems', []), prediction_text,
                          asset_resolver=resolve, simple_output=True)
            validate_pdf(stage / 'report.pdf')
            common_share = ((stage / 'report.pdf').stat().st_size + max(1, len(clips)) - 1) // max(1, len(clips))
            sizes = []
            for group, items in clips.items():
                available = policy.target_test_bytes - common_share - 4096
                duration = sum(item[2] for item in items)
                for source, relative, seconds in items:
                    encode_budgeted(source, stage / relative, int(available * seconds / duration), policy, exe)
                size = common_share + sum((stage / relative).stat().st_size for _, relative, _ in items)
                if size > policy.target_test_bytes:
                    raise OSError(f'{group} exceeds the per-jewel budget; originals retained')
                sizes.append(dict(jewel=group, bytes=size, limit_bytes=policy.target_test_bytes))
            files = {path.relative_to(stage).as_posix(): dict(bytes=path.stat().st_size, sha256=digest(path))
                     for path in stage.rglob('*') if path.is_file()}
            commit = dict(files=files, previous_bytes=previous_bytes, jewel_sizes=sizes)
            durable_json(receipt, commit)
            return _publish_and_clean(root, stage, commit)


def _decoded_digest(path):
    """Check that changing containers has preserved every decoded pixel."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    result = hashlib.sha256()
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result.update(frame.tobytes())
            count += 1
    finally:
        capture.release()
    if not count:
        raise OSError('No decoded frames: ' + str(path))
    return count, result.hexdigest()


def convert_simple_session_to_avi(directory, policy=None):
    """Lossless container conversion of an already simplified MP4 session."""
    root = Path(directory).resolve()
    policy = policy or RetentionPolicy.load()
    with SessionGuard(root), report_guard(root):
        if (root / 'session.json').exists():
            raise ValueError('Use normal finalization for sessions with original evidence')
        stage = root / '.simple-staging'
        receipt = stage / 'commit.json'
        if receipt.is_file():
            return _publish_and_clean(root, stage, json.loads(receipt.read_text()))
        sources = sorted(root.glob('J*/**/*.mp4'))
        if not sources:
            return dict(status='already_avi', final_bytes=tree_bytes(root))
        exe = encoder_available()
        if not exe:
            raise OSError('FFmpeg required for lossless AVI conversion')
        validate_pdf(root / 'report.pdf')
        previous_bytes = tree_bytes(root)
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        pdf = (root / 'report.pdf').read_bytes()
        groups = defaultdict(int)
        for source in sources:
            source = checked_local(root, source)
            relative = source.relative_to(root)
            target = relative.with_suffix('.avi')
            destination = stage / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            info = media_info(source)
            run_encoder([exe, '-nostdin', '-v', 'error', '-y', '-i', str(source),
                         '-map', '0:v:0', '-c:v', 'copy', '-an',
                         '-bsf:v', 'h264_mp4toannexb', '-vtag', 'H264',
                         '-r', str(info['fps']), str(destination)])
            actual = media_info(destination)
            if abs(actual['duration_seconds'] - info['duration_seconds']) > .01:
                raise OSError('AVI playback duration changed; MP4 retained')
            if _decoded_digest(source) != _decoded_digest(destination):
                raise OSError('AVI decoded pixels changed; MP4 retained')
            # Extensions have equal byte length, so the original PDF cross-reference
            # offsets and embedded photos remain intact. Change only exact URI entries.
            old = b'/URI (' + relative.as_posix().encode('ascii') + b')'
            new = b'/URI (' + target.as_posix().encode('ascii') + b')'
            pdf = pdf.replace(old, new)
            groups[relative.parts[0]] += destination.stat().st_size
        (stage / 'report.pdf').write_bytes(pdf)
        validate_pdf(stage / 'report.pdf')
        common_share = (len(pdf) + len(groups) - 1) // len(groups)
        sizes = [dict(jewel=name, bytes=size + common_share, limit_bytes=policy.target_test_bytes)
                 for name, size in groups.items()]
        if any(item['bytes'] > policy.target_test_bytes for item in sizes):
            raise OSError('AVI overhead exceeds the per-jewel limit; MP4 retained')
        # Preserve any existing AVI clips when converting a mixed session.
        for source in root.glob('J*/**/*.avi'):
            relative = source.relative_to(root)
            if not (stage / relative).exists():
                raise ValueError('Mixed MP4/AVI session needs review; originals retained')
        files = {p.relative_to(stage).as_posix(): dict(bytes=p.stat().st_size, sha256=digest(p))
                 for p in stage.rglob('*') if p.is_file()}
        commit = dict(files=files, previous_bytes=previous_bytes, jewel_sizes=sizes)
        durable_json(receipt, commit)
        return _publish_and_clean(root, stage, commit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--convert-mp4-to-avi', action='store_true')
    args = parser.parse_args()
    action = convert_simple_session_to_avi if args.convert_mp4_to_avi else finalize_simple_session
    print(json.dumps(action(args.session), indent=2))
