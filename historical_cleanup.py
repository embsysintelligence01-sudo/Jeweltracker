"""Scan every session, budget per test, and migrate only verified closed evidence."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import argparse,json,math,os,time
from evidence_archive import (ArchiveManager,RetentionPolicy,SessionGuard,checked_local,digest,
    durable_json,tree_bytes,encoder_available,validate_video,validate_pdf,valid_ack)


def per_test_sizes(data,files,final,target,manifest_bytes=0):
    """Charge own assets to each test; share common PDF/JSON once, not 2 MB/session."""
    owners={};tests=data['tests'];root=str(final)+os.sep
    def visit(v,index):
        if isinstance(v,dict):
            for key,item in v.items():
                if key!='original_source_frame':visit(item,index)
        elif isinstance(v,list):
            for item in v:visit(item,index)
        elif isinstance(v,str) and v.startswith(root):
            rel=str(Path(v).relative_to(final))
            if rel in files:owners.setdefault(rel,set()).add(index)
    for i,r in enumerate(tests):visit(r,i)
    # Canonical report/session/manifests are also retained publications.
    common=manifest_bytes+sum(item['bytes'] for rel,item in files.items() if rel not in owners)
    common+=sum(item['bytes'] for rel,item in files.items() if rel in ('report.pdf','session.json'))
    sizes=[common/len(tests)]*len(tests)
    for rel,ids in owners.items():
        for i in ids:sizes[i]+=files[rel]['bytes']/len(ids)
    result=[]
    for i,r in enumerate(tests):
        rel=str(Path(r['record_path']).relative_to(final))
        own_manifest=files.get(rel,{}).get('bytes',0)
        size=math.ceil(sizes[i]+own_manifest)
        result.append(dict(cycle_id=r.get('cycle_id'),jewel_id=r.get('jewel_id'),allocated_retained_bytes=size,
            target_bytes=target,size_exception=size>target,
            explanation='Quality floor and essential original evidence preserved; target is not a reason to truncate or discard evidence.' if size>target else 'Within per-test target; common PDF/JSON charged proportionally.'))
    return result


def load_session(root):
    p=root/'session.json'
    if not p.is_file():raise ValueError('Missing final session.json')
    data=json.loads(p.read_text())
    if not data.get('tests'):raise ValueError('No finalized tests')
    validate_pdf(root/'report.pdf')
    expected={p.resolve() for p in root.glob('test_*') if p.is_dir()}
    recorded=set()
    for r in data['tests']:
        if not r.get('capture_complete') or r.get('acid_onset') is None or not r.get('finished'):
            raise ValueError('Incomplete test: originals protected')
        if r.get('close_reason') not in (None,'REACTION_COMPLETE') or r.get('reaction_coverage_status')=='PARTIAL':
            raise ValueError('Interrupted/partial test: originals protected')
        if r['finished']<r['acid_onset']+4.9:raise ValueError('Five-second reaction not completed')
        if any(str(x).startswith(('Recovery/save error:','Prediction archive write failed:')) for x in r.get('issues',[])):
            raise ValueError('Test finalization error')
        p=checked_local(root,r['record_path']);recorded.add(root/p.parent.name)
        if not p.is_file() or json.loads(p.read_text())!=r:raise ValueError('Per-test manifest mismatch/missing')
        for key in ('rubbing_clip','acid_clip'):
            c=r.get(key) or {}
            if not c.get('path') or not c.get('frames'):raise ValueError('Required complete video missing')
            if not checked_local(root,c['path']).is_file():raise ValueError('Required video file missing')
        rubbing=r['rubbing_clip'];acid=r['acid_clip']
        if r.get('returned') is None:raise ValueError('Return timestamp unavailable')
        for c,lo,hi in ((rubbing,r['started'],r['returned']),(acid,r['acid_onset']-2,r['acid_onset']+5)):
            tol=max(.25,2/c.get('playback_fps',5))
            if c.get('actual_start',float('inf'))>lo+tol or c.get('actual_end',0)<hi-tol:
                raise ValueError('Video does not cover required pickup/return or acid window')
    if expected!=recorded:raise ValueError('Unfinished test directories present')
    if any((root/'deferred_recovery').glob('*.pkl')):raise ValueError('Pending background recovery work')
    return data


def active_legacy_session(root):
    """Old apps have no lock: conservatively refuse execution while ANY live app runs."""
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:argv=(proc/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if any(Path(os.fsdecode(arg)).name in ('jewel_tracker_5.py','jewel_tracker_6.py') for arg in argv if arg):return True
    return False


def scan_session(root,policy,check_video=False):
    root=Path(root).resolve()
    row=dict(session=str(root),previous_bytes=0,test_count=0,eligible=False,
        estimated_archive_bytes=None,estimated_savings_bytes=0,proposed_deletions=[])
    try:
        before=tree_bytes(root);row['previous_bytes']=before
        # Read-only lock probe; never create files in --dry-run.
        lock=root/'.recording.lock'
        if lock.exists():
            import fcntl
            with lock.open('r') as handle:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        elif active_legacy_session(root):raise ValueError('Legacy app may still be recording; stop it before migration')
        if (root/'archive_commit.json').exists():
            manager=ArchiveManager(root,policy);commit=manager.validate_commit(republish=False)
            final,data=manager._validated_generation(commit)
            if any(not r.get('capture_complete') or r.get('reaction_coverage_status')=='PARTIAL'
                   or r.get('close_reason') not in (None,'REACTION_COMPLETE') for r in data['tests']):raise ValueError('Incomplete archived session protected')
            m=json.loads((final/'archive_manifest.json').read_text());row.update(test_count=len(data['tests']),
                eligible=True,already_archived=True,acknowledged=valid_ack(root,commit,policy),test_sizes=m.get('test_sizes',[]))
            row['estimated_archive_bytes']=tree_bytes(final)
        else:
            data=load_session(root);row['test_count']=len(data['tests']);row['eligible']=True
            row['estimated_archive_bytes']=policy.target_test_bytes*len(data['tests'])
            if check_video:
                seen=set()
                for r in data['tests']:
                    for key in ('rubbing_clip','acid_clip','acid_candidate_clip'):
                        c=r.get(key) or {};path=c.get('path')
                        if path and path not in seen:validate_video(checked_local(root,path),c);seen.add(path)
        row['estimated_savings_bytes']=max(0,before-row['estimated_archive_bytes'])
        row['estimate_note']='Target-based estimate only; quality floors/essential evidence may exceed 2 MB per test. Verification and policy can prevent deletion.'
        candidates=[root/'frames',root/'frames.jsonl']
        for p in root.glob('test_*'):
            candidates.extend(p/name for name in ('baseline','application','reaction'))
            candidates.extend(p.glob('*.avi'))
        candidates.extend(root.glob('.publish-*'));candidates.extend(root.glob('frames.pruning.tmp'))
        row['proposed_deletions']=[str(p.relative_to(root)) for p in candidates if p.exists()]
        row['video_verification']='decoded' if check_video else 'structural_only; decode verification required before cleanup'
        row['cleanup_policy']='requires central acknowledgment' if policy.require_server_ack else 'offline verified-local cleanup'
    except Exception as exc:
        row.update(eligible=False,skip_reason=str(exc),estimated_savings_bytes=0)
    return row


def migrate(root,policy,resume=False):
    root=Path(root).resolve()
    with SessionGuard(root) as guard:
        manager=ArchiveManager(root,policy)
        if (root/'archive_commit.json').exists():
            commit=manager.validate_commit()
            final,data=manager._validated_generation(commit)
            manifest=json.loads((final/'archive_manifest.json').read_text())
            # Finish an authorized interrupted tomb removal before replacing its
            # owning generation. Otherwise its deletion ledger would be orphaned.
            if (root/'.cleanup'/commit['archive_id']).exists():
                pending=manager.finalize(True,guard)
                if pending.get('status')!='cleaned':return dict(pending,test_sizes=manifest.get('test_sizes',[]))
            fallback_videos=(policy.archive_codec=='h264' and encoder_available()
                             and any(v.get('codec')!='h264' for v in manifest.get('videos',[])))
            if manifest.get('storage_profile')!='compact_v2' or fallback_videos:
                result=manager.finalize(True,guard,force_repack=True)
                commit=json.loads((root/'archive_commit.json').read_text())
                final,data=manager._validated_generation(commit)
                manifest=json.loads((final/'archive_manifest.json').read_text())
            else:result=manager._cleanup(commit)
            return dict(result,already_archived=True,test_sizes=manifest.get('test_sizes',[]))
        state=root/'archive_state.json'
        old=json.loads(state.read_text()) if state.is_file() else {}
        pending=old if old.get('stage')=='PUBLISHING' else old.get('resume',{})
        if pending.get('stage')=='PUBLISHING' and not resume:raise RuntimeError('Interrupted archive: use --resume')
        if pending.get('stage')!='PUBLISHING':
            load_session(root)
            if active_legacy_session(root):raise RuntimeError('Live app running: historical migration postponed')
            # Structural checks + exclusive recording lock + no legacy app are
            # sufficient to mark older closed formats without modifying SOP data.
            if not (root/'session_closed.json').exists():
                durable_json(root/'session_closed.json',dict(closed_at=time.time(),historical_structural_verification=True))
        result=manager.finalize(quiescent=True,session_guard=guard)
        commit=json.loads((root/'archive_commit.json').read_text());final,_=manager._validated_generation(commit)
        return dict(result,test_sizes=json.loads((final/'archive_manifest.json').read_text()).get('test_sizes',[]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path,nargs='?',default=Path('captures/sessions'))
    mode=p.add_mutually_exclusive_group();mode.add_argument('--dry-run',action='store_true');mode.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--config',type=Path)
    p.add_argument('--verify-videos',action='store_true',help='Decode historical clips during read-only inspection')
    p.add_argument('--report',type=Path,help='Save final report outside all session folders')
    args=p.parse_args();policy=RetentionPolicy.load(args.config)
    if args.resume and not args.execute:p.error('--resume requires --execute')
    if args.execute and policy.archive_codec=='h264' and not encoder_available():p.error('FFmpeg/libx264 required for compact migration; install ffmpeg or set JEWELTRACKER_FFMPEG')
    single=(args.root/'session.json').is_file()
    roots=[args.root] if single else sorted(d for d in args.root.iterdir() if d.is_dir() and not d.name.startswith('.'))
    rows=[]
    for root in roots:
        row=scan_session(root,policy,args.verify_videos)
        state=root/'archive_state.json'
        if args.resume and args.execute and (root/'session_closed.json').exists() and state.exists():
            try:
                d=json.loads(state.read_text());pending=d if d.get('stage')=='PUBLISHING' else d.get('resume',{})
                if pending.get('stage')=='PUBLISHING':row['eligible']=True;row.pop('skip_reason',None)
            except (ValueError,TypeError,AttributeError) as exc:
                row.update(eligible=False,skip_reason='Corrupt recovery state: '+str(exc))
        if args.execute and row['eligible']:
            try:
                result=migrate(root,policy,args.resume);row.update(result)
            except Exception as exc:row.update(status='skipped',skip_reason=str(exc))
        try:row['new_bytes']=tree_bytes(root)
        except OSError as exc:row.update(new_bytes=row['previous_bytes'],skip_reason=str(exc))
        row['space_recovered_bytes']=max(0,row['previous_bytes']-row['new_bytes'])
        rows.append(row);print(json.dumps(row),flush=True)
    summary=dict(mode='execute' if args.execute else 'dry-run',sessions_scanned=len(rows),
        previous_bytes=sum(r['previous_bytes'] for r in rows),new_bytes=sum(r['new_bytes'] for r in rows),
        space_recovered_bytes=sum(r['space_recovered_bytes'] for r in rows),
        estimated_savings_bytes=sum(r['estimated_savings_bytes'] for r in rows),
        optimized_sessions=sum('test_sizes' in r and args.execute for r in rows),
        skipped_sessions=sum(not r['eligible'] or bool(r.get('skip_reason')) for r in rows),
        eligible_sessions=sum(r['eligible'] for r in rows),
        awaiting_ack=sum(r.get('status')=='awaiting_server_ack' for r in rows),
        size_exceptions=sum(t.get('size_exception',False) for r in rows for t in r.get('test_sizes',[])))
    print(json.dumps(dict(final_storage_report=summary),indent=2))
    if args.report:
        if any(args.report.resolve().is_relative_to(root.resolve()) for root in roots):p.error('Report must be outside session folders')
        durable_json(args.report,dict(summary=summary,sessions=rows))

if __name__=='__main__':main()
