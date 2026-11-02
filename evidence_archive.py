"""Crash-safe post-session archives. No model work and no live-path encoding."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass,asdict
from pathlib import Path
import argparse,copy,fcntl,hashlib,hmac,json,math,os,shutil,subprocess,time,uuid
import cv2

@dataclass
class RetentionPolicy:
    enabled: bool=True
    idle_buffer_seconds: float=20.
    pre_roll_seconds: float=5.
    disk_warning_free_gib: float=3.
    archive_codec: str='h264'
    h264_crf: int=18
    h264_preset: str='veryfast'
    encoder_threads: int=1
    cleanup_raw_after_archive: bool=True
    require_server_ack: bool=True
    ack_hmac_environment: str='JEWELTRACKER_ACK_HMAC_KEY'
    archive_retention_days: int=90

    def __post_init__(self):
        for name in ('enabled','cleanup_raw_after_archive','require_server_ack'):
            if type(getattr(self,name)) is not bool:raise ValueError('Retention flags must be JSON booleans')
        for name in ('idle_buffer_seconds','pre_roll_seconds','disk_warning_free_gib','h264_crf','encoder_threads','archive_retention_days'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Retention numbers must be finite')
        if self.disk_warning_free_gib<0 or self.archive_codec not in ('h264','mjpeg'):raise ValueError('Invalid storage alert/codec policy')
        if self.idle_buffer_seconds<self.pre_roll_seconds or self.pre_roll_seconds<4:
            raise ValueError('Idle buffer must cover at least four seconds of pre-roll')
        if not 0<=self.h264_crf<=28 or not 1<=self.encoder_threads<=2:
            raise ValueError('Invalid archive quality/thread settings')
        if self.archive_retention_days<1:raise ValueError('Retention must be at least one day')

    @classmethod
    def load(cls,path=None):
        path=Path(path or Path(__file__).with_name('evidence_retention.json'))
        return cls(**json.loads(path.read_text())) if path.is_file() else cls()


def sync_directory(path):
    fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def durable_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path);sync_directory(path.parent)
    finally:tmp.unlink(missing_ok=True)


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):result.update(data)
    return result.hexdigest()


class SessionGuard:
    def __init__(self,directory):
        self.directory=Path(directory).resolve()
        self.handle=(self.directory/'.recording.lock').open('a+')
        try:fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:self.handle.close();raise
    def __del__(self):
        if getattr(self,'handle',None) is not None:self.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):
        if not self.handle.closed:self.handle.close()


@contextmanager
def report_guard(directory):
    # Protect reports/manifests across threads, processes and archive publication.
    with (Path(directory)/'.evidence.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        yield


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file() and not p.is_symlink())


def checked_local(root,path):
    root=Path(root).resolve();p=Path(path)
    p=p if p.is_absolute() else root/p
    if p.is_symlink():raise ValueError('Symlink evidence is not accepted: '+str(p))
    resolved=p.resolve()
    if not resolved.is_relative_to(root):raise ValueError('Evidence escapes session: '+str(p))
    # Parent symlinks within the session are also disallowed.
    if any(q.is_symlink() for q in [p,*p.parents] if q!=root.parent):
        raise ValueError('Symlink parent in evidence path')
    return resolved


def validate_video(path,metadata):
    path=Path(path)
    if not path.is_file() or path.stat().st_size==0:raise OSError('Missing/empty video: '+str(path))
    if hasattr(cv2,'CAP_PROP_N_THREADS'):
        cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    else:cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():raise OSError('Unplayable video: '+str(path))
        fps=float(cap.get(cv2.CAP_PROP_FPS));count=0;shape=None
        if not math.isfinite(fps) or fps<=0:raise OSError('Invalid video FPS')
        while True:
            ok,frame=cap.read()
            if not ok:break
            if frame is None or not frame.size:raise OSError('Empty video frame')
            if shape is not None and frame.shape!=shape:raise OSError('Video frame resolution changed')
            shape=frame.shape;count+=1
        if count<1 or count!=metadata.get('frames'):raise OSError('Decoded video frame count does not match evidence metadata')
        timestamps=metadata.get('timestamps',[])
        if len(timestamps)!=count or any(b<=a for a,b in zip(timestamps,timestamps[1:])):
            raise OSError('Invalid video timestamp sidecar')
        duration=count/fps;expected=float(metadata['playback_seconds'])
        if abs(duration-expected)>max(.25,2/fps):raise OSError('Video duration does not match evidence metadata')
        if abs(float(metadata['actual_start'])-timestamps[0])>.001 or abs(float(metadata['actual_end'])-timestamps[-1])>.001:
            raise OSError('Video source timestamps disagree')
        requested_start,requested_end=metadata['requested_start'],metadata['requested_end']
        if timestamps[0]<requested_start-.001 or timestamps[-1]>requested_end+.001:
            raise OSError('Video outside declared interval')
        if expected+1/fps<timestamps[-1]-timestamps[0]:raise OSError('Video plays faster than declared source interval')
        return dict(frames=count,fps=fps,duration_seconds=duration,width=shape[1],height=shape[0])
    finally:cap.release()


def encoder_available():
    exe=shutil.which('ffmpeg')
    if not exe:return None
    result=subprocess.run([exe,'-hide_banner','-encoders'],capture_output=True,text=True,timeout=10)
    return exe if result.returncode==0 and 'libx264' in result.stdout else None


def archive_video(source,output,policy,ffmpeg):
    if ffmpeg and policy.archive_codec=='h264':
        output=output.with_suffix('.mp4')
        command=[ffmpeg,'-nostdin','-v','error','-y','-i',str(source),'-an','-c:v','libx264',
            '-crf',str(policy.h264_crf),'-preset',policy.h264_preset,'-threads',str(policy.encoder_threads),
            '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
        # Low-priority bounded encoder, exclusively after the inference session.
        if shutil.which('nice'):command=['nice','-n','10',*command]
        result=subprocess.run(command,capture_output=True,text=True,timeout=3600)
        if result.returncode:raise OSError('H.264 encoding failed: '+result.stderr[-1000:])
        return output,'h264'
    output=output.with_suffix(source.suffix.lower())
    shutil.copyfile(source,output)
    return output,'original_mjpeg_fallback'


def ack_payload_valid(data,session_id,manifest_sha256,policy):
    if not policy.require_server_ack:return True
    key=os.environ.get(policy.ack_hmac_environment)
    if not key:return False
    try:
        data=dict(data);signature=data.pop('signature')
        expected=hmac.new(key.encode(),json.dumps(data,sort_keys=True,separators=(',',':')).encode(),hashlib.sha256).hexdigest()
        return (hmac.compare_digest(signature,expected) and data['session_id']==session_id
            and data['archive_manifest_sha256']==manifest_sha256 and bool(data['server_receipt_id'])
            and bool(data['acknowledged_at']))
    except (ValueError,KeyError,TypeError):return False


def valid_ack(root,commit,policy):
    if not policy.require_server_ack:return True
    try:data=json.loads((Path(root)/'server_ack.json').read_text())
    except (OSError,ValueError):return False
    return ack_payload_valid(data,Path(root).name,commit['manifest_sha256'],policy)


def resume_expirations(parent,policy,session_id=None):
    """A durable deletion ledger survives a crash midway through rmtree."""
    parent=Path(parent)
    for ledger in sorted((parent/'.retention-deletions').glob('*.json')):
        data=json.loads(ledger.read_text())
        if data.get('status')=='deleted' or (session_id is not None and data['session_id']!=session_id):continue
        name=data['tomb_name']
        if Path(name).name!=name or not name.startswith('.expired-'):raise ValueError('Unsafe expiration tomb name')
        tomb=parent/name
        if tomb.is_symlink():raise ValueError('Symlink expiration tomb')
        if time.time()-data['committed_at']<=policy.archive_retention_days*86400:continue
        if not ack_payload_valid(data.get('receipt',{}),data['session_id'],data['manifest_sha256'],policy):continue
        if tomb.exists():shutil.rmtree(tomb);sync_directory(parent)
        elif (parent/data['session_id']).exists():continue  # crash before atomic rename: revalidate normally
        data['status']='deleted';durable_json(ledger,data)



class ArchiveManager:
    def __init__(self,directory,policy=None):
        self.root=Path(directory).resolve();self.policy=policy or RetentionPolicy.load()

    def _state(self,stage,**kw):
        durable_json(self.root/'archive_state.json',dict(stage=stage,updated_at=time.time(),**kw))

    def finalize(self,quiescent=False,session_guard=None):
        if not quiescent:raise RuntimeError('Archive requires all writer/recovery/report jobs to be drained')
        if session_guard is None:
            with SessionGuard(self.root) as guard:return self.finalize(True,guard)
        if session_guard.handle.closed or session_guard.directory!=self.root:
            raise RuntimeError('Invalid session ownership guard')
        with report_guard(self.root):
            try:
                if (self.root/'archive_commit.json').is_file():
                    commit=self.validate_commit();return self._cleanup(commit)
                state_path=self.root/'archive_state.json'
                state=json.loads(state_path.read_text()) if state_path.is_file() else {}
                resume=state if state.get('stage')=='PUBLISHING' else state.get('resume',{})
                if resume.get('stage')=='PUBLISHING':
                    commit=dict(archive_id=resume['archive_id'],manifest_sha256=resume['manifest_sha256'],committed_at=time.time())
                    final,data=self._validated_generation(commit)
                    self._publish(final,data)
                    durable_json(self.root/'archive_commit.json',commit)
                    return self._cleanup(commit)
                return self._build()
            except Exception as exc:
                state_path=self.root/'archive_state.json'
                previous=json.loads(state_path.read_text()) if state_path.is_file() else {}
                resume=previous if previous.get('stage')=='PUBLISHING' else previous.get('resume',{})
                self._state('FAILED_RAW_RETAINED',error=str(exc),resume=resume)
                raise

    def _build(self):
        if not (self.root/'session_closed.json').is_file():raise RuntimeError('Session has no quiescent shutdown marker')
        data=json.loads((self.root/'session.json').read_text())
        if not data.get('tests'):raise RuntimeError('No per-jewel evidence was finalized; raw retained')
        expected={p.resolve() for p in self.root.glob('test_*') if p.is_dir()}
        recorded={checked_local(self.root,r['record_path']).parent for r in data['tests']}
        if expected!=recorded:raise OSError('Some test directories lack finalized manifests; raw retained')
        for r in data['tests']:
            if r.get('classifier_input_archive_error'):raise OSError('Exact classifier-input archive failed; raw retained')
            if any(str(note).startswith(('Recovery/save error:','Prediction archive write failed:')) for note in r.get('issues',[])):
                raise RuntimeError('A test finalizer failed; raw retained')
            p=checked_local(self.root,r['record_path'])
            if not p.is_file():raise OSError('Missing per-test manifest')
            disk=json.loads(p.read_text())
            if disk!=r:raise OSError('Session and per-test manifests disagree')
        identifier=uuid.uuid4().hex
        stage=self.root/'archives'/('.staging-'+identifier);final=stage.parent/identifier
        stage.mkdir(parents=True);self._state('BUILDING',archive_id=identifier)
        mapping={};videos={};validation=[];video_sources=set();source_cache={};ffmpeg=encoder_available()
        def logical(p):return str(final/p.relative_to(stage))
        def copy_asset(source):
            p=checked_local(self.root,source)
            if not p.is_file():raise OSError('Missing referenced evidence: '+str(p))
            rel=p.relative_to(self.root)
            dest=stage/('original_frames' if rel.parts[0]=='frames' else 'images')/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copyfile(p,dest)
            mapping[str(p)]=logical(dest);return logical(dest)
        for r in data['tests']:
            for name in ('rubbing_clip','acid_clip'):
                clip=r.get(name)
                if not clip or not clip.get('path') or not clip.get('frames'):
                    raise OSError('Required '+name+' is unavailable; raw retained')
        # Validate and deduplicate clips before changing any canonical reference.
        for r in data['tests']:
            for name in ('rubbing_clip','acid_clip','acid_candidate_clip'):
                clip=r.get(name)
                if not clip or not clip.get('path'):continue
                source=checked_local(self.root,clip['path'])
                signature=json.dumps(clip,sort_keys=True)
                cached=source_cache.get((str(source),signature))
                info,sha=cached if cached else (validate_video(source,clip),digest(source))
                source_cache[(str(source),signature)]=(info,sha)
                video_sources.add(str(source))
                if sha not in videos:
                    dest=stage/'videos'/sha;dest.parent.mkdir(exist_ok=True)
                    dest,codec=archive_video(source,dest,self.policy,ffmpeg)
                    valid=validate_video(dest,clip)
                    if valid['width']<info['width'] or valid['height']<info['height']:
                        raise OSError('Archive video resolution shrank')
                    videos[sha]=(logical(dest),codec)
                    validation.append(dict(source=str(source),archived=logical(dest),codec=codec,**valid))
                mapping[str(source)]=videos[sha][0]
        # Keep exact classifier summaries for future sessions; older JPEG exports
        # can only produce explicitly labelled reconstructions.
        from acid_test.model_training.session_dataset import prepare_temporal_triplet
        import numpy as np
        for r in data['tests']:
            folder=checked_local(self.root,r['record_path']).parent
            complete_sequences=all(any((folder/name).glob('*.jpg')) for name in ('baseline','application','reaction'))
            if not r.get('classifier_input_images') and (r.get('prediction') or complete_sequences):
                sequences=[]
                for name,tail in (('baseline',None),('application',3),('reaction',5)):
                    paths=sorted((folder/name).glob('*.jpg'));paths=paths[-tail:] if tail else paths
                    ims=[cv2.imread(str(p)) for p in paths]
                    if not ims or any(im is None for im in ims):raise OSError('Missing classifier sequence; raw retained')
                    sequences.append(ims)
                images=prepare_temporal_triplet(*sequences)
                r['classifier_input_images']={}
                for name,im in zip(('baseline','application','reaction'),images):
                    dest=stage/'classifier_inputs'/folder.name/(name+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
                    if not cv2.imwrite(str(dest),cv2.cvtColor(im,cv2.COLOR_RGB2BGR)):raise OSError('Classifier summary save failed')
                    r['classifier_input_images'][name]=logical(dest)
                r['classifier_input_origin']='reconstructed_from_saved_jpeg_not_original_live_arrays'
        # Rewrite only real local media references. Model paths outside this
        # session and descriptive text are not followed or modified.
        media={'.jpg','.jpeg','.png','.avi','.mp4'}
        def rewrite(value):
            if isinstance(value,dict):return {k:rewrite(v) for k,v in value.items()}
            if isinstance(value,list):return [rewrite(v) for v in value]
            if not isinstance(value,str):return value
            p=Path(value)
            if p.is_absolute() and p.resolve().is_relative_to(final):return value
            if p.suffix.lower() not in media:return value
            candidate=p if p.is_absolute() else self.root/p
            if p.is_absolute() and not candidate.resolve().is_relative_to(self.root):return value
            p=checked_local(self.root,candidate)
            if str(p) in mapping:return mapping[str(p)]
            if p.suffix.lower() in {'.avi','.mp4'}:raise OSError('Uncatalogued video reference')
            return copy_asset(p)
        data=rewrite(data)
        for r in data['tests']:
            old=checked_local(self.root,r['record_path'])
            target=stage/'tests'/old.parent.name/'manifest.json'
            r['record_path']=logical(target)
            # Metadata sidecars remain per clip even if video bytes deduplicate.
            for name in ('rubbing_clip','acid_clip','acid_candidate_clip'):
                clip=r.get(name)
                if clip and clip.get('path'):
                    target_sidecar=target.parent/(name+'.json')
                    clip['sidecar']=logical(target_sidecar);durable_json(target_sidecar,clip)
            durable_json(target,r)
        # Retain the frame catalogue without leaving dangling raw file links.
        index=stage/'frames.jsonl';original=self.root/'frames.jsonl'
        video_frames={}
        for record in data['tests']:
            for name in ('rubbing_clip','acid_clip','acid_candidate_clip'):
                clip=record.get(name) or {}
                for i,ts in enumerate(clip.get('timestamps',[])):
                    video_frames.setdefault(ts,(clip['path'],i))
        with index.open('w') as out:
            if original.is_file():
                for line in original.read_text().splitlines():
                    entry=json.loads(line);old=str(self.root/entry['file']);timestamp=entry['timestamp']
                    entry['original_file']=entry['file'];entry['file']=mapping.get(old)
                    entry['storage']='selected_original_image' if entry['file'] else 'not_selected_for_archive'
                    if not entry['file'] and timestamp in video_frames:
                        video,index_in_video=video_frames[timestamp]
                        entry.update(file=video,frame_index=index_in_video,storage='archived_video')
                    out.write(json.dumps(entry)+'\n')
            out.flush();os.fsync(out.fileno())
        durable_json(stage/'session.json',data)
        from audit_report import render_report
        from session_evidence import prediction_text
        def resolve(path):
            p=Path(path)
            return stage/p.relative_to(final) if p.is_relative_to(final) else p
        render_report(stage/'report.pdf',self.root,data['jewels'],data['tests'],data.get('recording_problems',[]),prediction_text,asset_resolver=resolve)
        if not (stage/'report.pdf').read_bytes().startswith(b'%PDF'):raise OSError('Archive PDF invalid')
        # Check rewritten media references and all expected source images now.
        def verify_refs(v):
            if isinstance(v,dict):
                for item in v.values():verify_refs(item)
            elif isinstance(v,list):
                for item in v:verify_refs(item)
            elif isinstance(v,str) and v.startswith(str(final)+os.sep):
                if not resolve(v).is_file():raise OSError('Missing archive reference: '+v)
        verify_refs(data)
        files={str(p.relative_to(stage)):dict(sha256=digest(p),bytes=p.stat().st_size)
               for p in stage.rglob('*') if p.is_file()}
        manifest=dict(schema='jewel_evidence_archive_v1',session_id=self.root.name,archive_id=identifier,
            created_at=time.time(),files=files,videos=validation,video_sources=sorted(video_sources),policy=asdict(self.policy))
        durable_json(stage/'archive_manifest.json',manifest)
        for p in stage.rglob('*'):
            if p.is_file():
                with p.open('rb') as f:os.fsync(f.fileno())
        sync_directory(stage);os.replace(stage,final);sync_directory(final.parent)
        self._state('PUBLISHING',archive_id=identifier,manifest_sha256=digest(final/'archive_manifest.json'))
        # Publication is resumable: immutable archive first, canonical pointers
        # second, commit last. No source is removed on a partial publication.
        self._publish(final,data)
        commit=dict(archive_id=identifier,manifest_sha256=digest(final/'archive_manifest.json'),committed_at=time.time())
        durable_json(self.root/'archive_commit.json',commit)
        self._state('COMMITTED',archive_id=identifier)
        return self._cleanup(commit)

    def _publish(self,final,data):
        for name in ('report.pdf','session.json','frames.jsonl'):
            temp=self.root/('.publish-'+uuid.uuid4().hex)
            shutil.copyfile(final/name,temp)
            with temp.open('rb') as f:os.fsync(f.fileno())
            os.replace(temp,self.root/name)
        for r in data['tests']:
            source=Path(r['record_path']);folder=self.root/source.parent.name
            durable_json(folder/'manifest.json',r)
        sync_directory(self.root)

    def _validated_generation(self,commit):
        final=checked_local(self.root,self.root/'archives'/commit['archive_id'])
        manifest=final/'archive_manifest.json'
        if digest(manifest)!=commit['manifest_sha256']:raise OSError('Archive manifest checksum mismatch')
        content=json.loads(manifest.read_text())
        for rel,item in content['files'].items():
            p=checked_local(final,rel)
            if not p.is_file() or p.stat().st_size!=item['bytes'] or digest(p)!=item['sha256']:
                raise OSError('Archived evidence checksum mismatch: '+rel)
        return final,json.loads((final/'session.json').read_text())

    def validate_commit(self,republish=True):
        commit=json.loads((self.root/'archive_commit.json').read_text())
        final,data=self._validated_generation(commit)
        if republish:self._publish(final,data)
        return commit

    def _cleanup(self,commit):
        if not self.policy.cleanup_raw_after_archive:
            self._state('COMMITTED_RAW_RETAINED');return dict(status='raw_retained')
        if not valid_ack(self.root,commit,self.policy):
            self._state('AWAITING_SERVER_ACK',archive_id=commit['archive_id'])
            return dict(status='awaiting_server_ack',manifest_sha256=commit['manifest_sha256'])
        self.validate_commit(republish=False)
        final=self.root/'archives'/commit['archive_id'];data=json.loads((final/'session.json').read_text())
        candidates=[self.root/'frames']
        for r in data['tests']:
            folder=self.root/Path(r['record_path']).parent.name
            candidates.extend(folder/name for name in ('baseline','application','reaction')
                              if name in r.get('classifier_input_images',{}))
            # Only remove original video exports catalogued in the archive.
        manifest=json.loads((final/'archive_manifest.json').read_text())
        candidates.extend(checked_local(self.root,source) for source in manifest['video_sources'])
        trash=self.root/'.cleanup'/commit['archive_id'];trash.mkdir(parents=True,exist_ok=True)
        self._state('CLEANING',archive_id=commit['archive_id'])
        deleted=0
        for source in candidates:
            source=checked_local(self.root,source)
            if not source.exists():continue
            target=trash/source.relative_to(self.root);target.parent.mkdir(parents=True,exist_ok=True)
            size=tree_bytes(source) if source.is_dir() else source.stat().st_size
            os.replace(source,target);sync_directory(source.parent);deleted+=size
        # A crash after a rename leaves recoverable trash, never a half-deleted
        # active evidence directory. Revalidate archive/ack before every retry.
        shutil.rmtree(trash)
        self._state('CLEANED',archive_id=commit['archive_id'],removed_bytes=deleted)
        return dict(status='cleaned',removed_bytes=deleted)


def main():
    parser=argparse.ArgumentParser(description='Evidence archives. Cleanup defaults to dry-run; no legacy session is assumed closed.')
    parser.add_argument('root',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--archive-session',action='store_true')
    parser.add_argument('--confirm-closed',action='store_true',help='Explicitly certify an old pre-retention session has no active jobs')
    parser.add_argument('--apply',action='store_true',help='Apply eligible committed raw cleanup/expired archive removal; otherwise list only')
    parser.add_argument('--ack-file',type=Path,help='Import an authenticated central-server receipt for this session')
    args=parser.parse_args();policy=RetentionPolicy.load(args.config)
    if args.ack_file and not args.apply:parser.error('--ack-file requires --apply; dry-run is read-only')
    if args.archive_session:
        with SessionGuard(args.root) as guard:
            if args.confirm_closed:durable_json(args.root/'session_closed.json',dict(closed_at=time.time(),legacy_operator_confirmation=True))
            print(json.dumps(ArchiveManager(args.root,policy).finalize(quiescent=True,session_guard=guard),indent=2))
        return
    single=(args.root/'session.json').is_file() or (args.root/'archive_commit.json').is_file() or not args.root.exists()
    if args.apply:
        parent=args.root.parent if single else args.root
        resume_expirations(parent,policy,args.root.name if single else None)
    if not args.root.exists():
        print(json.dumps(dict(session=str(args.root),status='not_present')));return
    roots=[args.root] if single else sorted(p for p in args.root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    for root in roots:
        if not (root/'archive_commit.json').is_file():
            state_path=root/'archive_state.json'
            state=json.loads(state_path.read_text()) if state_path.is_file() else {}
            resume=state if state.get('stage')=='PUBLISHING' else state.get('resume',{})
            if args.apply and (root/'session_closed.json').is_file() and resume.get('stage')=='PUBLISHING':
                try:
                    result=ArchiveManager(root,policy).finalize(quiescent=True)
                    print(json.dumps(dict(session=str(root),publication_resumed=True,**result)))
                except (OSError,ValueError,RuntimeError) as exc:
                    print(json.dumps(dict(session=str(root),eligible=False,error=str(exc))))
            else:
                print(json.dumps(dict(session=str(root),eligible=False,reason='No validated archive commit; raw will be retained')))
            continue
        try:
            with SessionGuard(root),report_guard(root):
                manager=ArchiveManager(root,policy)
                # Dry-run is strictly read-only: validate without republishing.
                commit=manager.validate_commit(republish=False)
                if args.ack_file:
                    receipt=json.loads(args.ack_file.read_text());durable_json(root/'server_ack.json',receipt)
                acknowledged=valid_ack(root,commit,policy)
                expired=time.time()-commit['committed_at']>policy.archive_retention_days*86400
                print(json.dumps(dict(session=str(root),acknowledged=acknowledged,expired=expired,apply=args.apply)))
                if args.apply:
                    manager.validate_commit()
                    if acknowledged and expired:
                        # Retention expiry deletes whole closed sessions only with
                        # policy-authorized receipt and validated checksums.
                        identifier=uuid.uuid4().hex
                        tomb=root.with_name('.expired-'+root.name+'-'+identifier)
                        receipt=json.loads((root/'server_ack.json').read_text()) if (root/'server_ack.json').is_file() else {}
                        ledger=root.parent/'.retention-deletions'/(identifier+'.json')
                        durable_json(ledger,dict(status='deleting',session_id=root.name,tomb_name=tomb.name,
                            manifest_sha256=commit['manifest_sha256'],committed_at=commit['committed_at'],receipt=receipt))
                        os.replace(root,tomb);sync_directory(root.parent)
                        resume_expirations(root.parent,policy,root.name)
                    else:manager._cleanup(commit)
        except (OSError,ValueError,RuntimeError) as exc:
            print(json.dumps(dict(session=str(root),eligible=False,error=str(exc))))

if __name__=='__main__':main()
