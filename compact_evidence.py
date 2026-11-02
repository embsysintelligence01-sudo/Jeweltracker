"""Quality-bounded archival encoding; runs only after evidence processing."""
from pathlib import Path
from fractions import Fraction
import math,re,shutil,subprocess,uuid
import cv2


def media_info(path):
    params=[cv2.CAP_PROP_N_THREADS,1] if hasattr(cv2,'CAP_PROP_N_THREADS') else []
    cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,params)
    try:
        if not cap.isOpened():raise OSError('Cannot inspect archive video')
        fps=float(cap.get(cv2.CAP_PROP_FPS));count=0;shape=None
        while True:
            ok,frame=cap.read()
            if not ok:break
            if shape is not None and frame.shape!=shape:raise OSError('Archive resolution changed')
            count+=1;shape=frame.shape
        if not count or not math.isfinite(fps) or fps<=0:raise OSError('Archive has no decodable frames')
        return dict(frames=count,fps=fps,duration_seconds=count/fps,width=shape[1],height=shape[0])
    finally:cap.release()


def run_encoder(command,timeout=3600):
    if shutil.which('nice'):command=['nice','-n','10',*command]
    result=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
    if result.returncode:raise OSError('Archive encoding/quality check failed: '+result.stderr[-1200:])
    return result


def quality_ssim(exe,source,destination,info,threads=1):
    selection=info.get('source_selection_filter','setpts=PTS-STARTPTS')
    filters=(f"[0:v]{selection},scale={info['width']}:{info['height']}:flags=lanczos,setsar=1[a];"
             "[1:v]setsar=1,setpts=PTS-STARTPTS[b];[a][b]ssim=shortest=1")
    result=run_encoder([exe,'-nostdin','-hide_banner','-v','info','-threads',str(threads),'-i',str(source),
        '-threads',str(threads),'-i',str(destination),'-filter_complex_threads','1','-lavfi',filters,
        '-threads',str(threads),'-an','-f','null','-'])
    scores=re.findall(r'All:([0-9.]+)',result.stderr)
    if not scores:raise OSError('SSIM verification unavailable')
    return float(scores[-1])


def encode_compact(source,output,metadata,policy,exe,budget):
    """Retain the smallest acceptable profile, never truncate to force a budget."""
    src=media_info(source)
    floor=min(policy.archive_min_long_edge,max(src['width'],src['height']))
    profiles=[(max(floor,min(1600,max(src['width'],src['height']))),min(15.,src['fps']),22),
              (floor,min(10.,src['fps']),25),(floor,min(10.,src['fps']),28),
              (max(src['width'],src['height']),src['fps'],18)]
    results=[];kept=None
    for edge,fps,crf in dict.fromkeys(profiles):
        if crf==18 and kept is not None:break
        scale=min(1.,edge/max(src['width'],src['height']))
        width=max(2,int(src['width']*scale)//2*2);height=max(2,int(src['height']*scale)//2*2)
        # No cropping: resize the full source frame and preserve its aspect ratio.
        mapping=list(range(src['frames']))
        selection='setpts=PTS-STARTPTS'
        if abs(fps-src['fps'])>.001:
            rate=(Fraction(str(fps))/Fraction(str(src['fps']))).limit_denominator(1_000_000)
            a,b=rate.numerator,rate.denominator
            # Explicit deterministic selection permits exact original timestamp
            # mapping, including the first frame, without guessing fps-filter phase.
            mapping=[i for i in mapping if i*a//b>(i-1)*a//b]
            selection=f"select='gt(floor(n*{a}/{b}),floor((n-1)*{a}/{b}))',settb=AVTB,setpts=N/({fps:.9f}*TB)"
        filters=selection+f',scale={width}:{height}:flags=lanczos,setsar=1'
        candidate=output.parent/(output.name+'.'+uuid.uuid4().hex+'.mp4')
        result=dict(width=width,height=height,fps=fps,crf=crf)
        try:
            run_encoder([exe,'-nostdin','-v','error','-y','-threads','1','-i',str(source),'-an','-vf',filters,
                '-c:v','libx264','-crf',str(crf),'-preset',policy.h264_preset,'-threads',str(policy.encoder_threads),
                '-filter_threads','1','-r',f'{fps:.9f}','-fps_mode','cfr','-pix_fmt','yuv420p','-movflags','+faststart',str(candidate)])
            info=media_info(candidate)
            if abs(info['duration_seconds']-metadata['playback_seconds'])>max(.25,2/fps):
                raise OSError('Adaptive encode changed the complete clip duration')
            if info['frames']!=len(mapping):raise OSError('Encoded frame selection does not match timestamp mapping')
            info['source_selection_filter']=selection
            info['source_frame_indices']=mapping
            ssim=quality_ssim(exe,source,candidate,info,policy.encoder_threads)
            result.update(bytes=candidate.stat().st_size,ssim=ssim,accepted=ssim>=policy.archive_min_ssim)
            results.append(result)
            if not result['accepted']:continue
            if kept is None or candidate.stat().st_size<kept[0]:
                destination=output.with_suffix('.mp4')
                candidate.replace(destination);kept=(destination.stat().st_size,destination,info,ssim)
            if kept[0]<=budget:break
        finally:candidate.unlink(missing_ok=True)
    if kept is None:raise OSError('No archive profile passed the configured visible-quality floor; originals retained')
    size,destination,info,ssim=kept
    info.update(source_width=src['width'],source_height=src['height'],
                source_frames=src['frames'],source_fps=src['fps'],ssim=ssim,bytes=size,
                bitrate_bps=round(size*8/info['duration_seconds']),budget_bytes=budget,
                budget_met=size<=budget,profiles=results)
    return destination,'h264',info
