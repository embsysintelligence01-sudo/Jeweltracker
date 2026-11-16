"""Keep lossless acid review samples independently of compressed session evidence."""
import json
import os
import shutil
import tempfile
from pathlib import Path


def preserve_review_sample(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    record = json.loads(manifest_path.read_text(encoding='utf-8'))
    if record.get('review_capture_path'):
        return record['review_capture_path']  # Already queued, possibly reviewed/moved.
    export = record.get('temporal_training_capture')
    if not export:
        return None
    source = manifest_path.parent
    if not source.name.startswith('test_'):
        raise ValueError('Expected an individual acid test directory')
    queue = source.parent / 'pending_review'
    queue.mkdir(exist_ok=True)
    final = queue / source.name
    if not final.exists():
        temporary = Path(tempfile.mkdtemp(prefix='.review_', dir=queue))
        try:
            for phase in ('baseline', 'transition', 'observation'):
                entries = export[phase + '_frames']
                if not entries:
                    raise ValueError('Review sequence is incomplete')
                for entry in entries:
                    relative = entry if isinstance(entry, str) else entry['path']
                    path = (source / relative).resolve()
                    if not path.is_relative_to(source):
                        raise ValueError('Review frame escapes sample directory')
                    destination = temporary / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
                    with destination.open('rb') as handle:
                        os.fsync(handle.fileno())
                    if destination.read_bytes() != path.read_bytes():
                        raise OSError('Review frame copy verification failed')
            from evidence_archive import durable_json, sync_directory
            durable_json(temporary / 'manifest.json', record)
            for folder in sorted((p for p in temporary.rglob('*') if p.is_dir()),
                                 key=lambda p: len(p.parts), reverse=True):
                sync_directory(folder)
            sync_directory(temporary)
            temporary.rename(final)
            sync_directory(queue)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    record['review_capture_path'] = str(final / 'manifest.json')
    from evidence_archive import durable_json
    durable_json(manifest_path, record)
    return record['review_capture_path']
