"""Visual audit PDF. All evidence and outcomes come from the saved manifest."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import textwrap
from xml.sax.saxutils import escape
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4

NAVY=HexColor('#0c1c2b');TEAL=HexColor('#008b8b');INK=HexColor('#243546')
PALE=HexColor('#f3f6f9');BORDER=HexColor('#d9e2ea');AMBER=HexColor('#a7600a')
GREEN=HexColor('#16803c')
W,H=A4
REPORT_TIMEZONE=ZoneInfo('Asia/Kolkata')


def audit_outcome(r):
    if r.get('sop_review_status')=='PENDING':return 'PENDING'
    if r.get('sop_review_status')=='REVIEW_REQUIRED':return 'REVIEW REQUIRED'
    if not r.get('capture_complete') or not r.get('prediction'):return 'INCOMPLETE'
    mask_verified=r.get('mask_evidence_status')=='VERIFIED'
    if (r.get('rubbing_verified') and mask_verified and r.get('acid_target_status')=='ACID_OVERLAPS_NEW_RUB'
            and not r.get('rub_previous_acid_overlap',0) and not r.get('issues')
            and r['prediction']['prediction']!='invalid'):
        return 'EVIDENCE VERIFIED'
    return 'REVIEW REQUIRED'


def when(value):
    return datetime.fromtimestamp(value,REPORT_TIMEZONE).strftime('%H:%M:%S.%f')[:-3] if value is not None else 'Not captured'


def session_period(tests):
    starts=[r['started'] for r in tests if r.get('started') is not None]
    ends=[r['finished'] for r in tests if r.get('finished') is not None]
    def stamp(value):
        return datetime.fromtimestamp(value,REPORT_TIMEZONE).strftime('%d %b %Y %H:%M:%S')
    return ('All jewels | Start: '+(stamp(min(starts)) if starts else 'Not captured')
            +' | End: '+(stamp(max(ends)) if ends else 'Not captured')+' | IST')


def event_rows(r):
    """Per-cycle audit events; hand visibility is never a missing interval."""
    rows=[]
    for v in r.get('visibility',[]):
        if v.get('jewel_id')==r.get('jewel_id') and v.get('status')=='MISSED':
            rows.append(['Jewel genuinely missed',when(v['start']),when(v['end']),
                         f"{max(0.,v['end']-v['start']):.2f} s",'Tracking lost; no hand association'])
    if not rows:
        rows.append(['Jewel tracking','-','-','0.00 s missed','No genuine missing interval recorded'])
    baseline=r.get('classification_baseline_window')
    start=r.get('baseline_at')
    stop=(baseline[1] if baseline else r.get('application_candidate_at') or r.get('acid_onset'))
    if start is not None:
        stop=min(start+2,stop) if stop is not None else min(start+2,r.get('finished') or start)
        duration=max(0.,stop-start)
        status=('Complete' if duration>=1.9 else 'Incomplete; early application' if r.get('acid_onset') is not None else 'Incomplete; application not confirmed / session ended')
        if r.get('frame_counts',{}).get('baseline')==0:
            status='No usable baseline frames captured'
        rows.append(['Baseline',when(start),when(stop),f'{duration:.2f} / 2.00 s',status])
    onset=r.get('acid_onset')
    if onset is not None:
        end=r.get('reaction_observation_end')
        if end is None:end=min(onset+5,r.get('finished') or onset,r.get('reaction_end') if r.get('reaction_end') is not None else float('inf'))
        usable=r.get('usable_reaction_seconds',max(0.,end-onset))
        rows.append(['Reaction observation',when(onset),when(end),f'{usable:.2f} / 5.00 s usable',
                     r.get('reaction_coverage_status') or ('Complete' if end>=onset+4.9 else 'Incomplete')])
        coverage=r.get('reaction_interrupted_at') or r.get('coverage_since')
        if r.get('reaction_end') is not None:
            rows.append(['Possible cleaning / coverage',when(coverage if coverage is not None else r['reaction_end']),
                         when(min(onset+5,r.get('finished') or onset+5)),'Interrupted',
                         'Hand/cloth/glare observed; cleaning not confirmed'])
        for event in r.get('reaction_rejected_frames',[]):
            rows.append(['Reaction frame excluded',when(event.get('timestamp')),'-','-',str(event.get('reason','Visibility uncertain'))])
    else:
        rows.append(['Acid application','-','-','Not confirmed','No verified application onset; reaction not started'])
    for event in r.get('acid_candidate_events',[]):
        rows.append(['Application check',when(event.get('candidate_at')),when(event.get('timestamp')),'-',
                     str(event.get('status',''))+(' | '+str(event['reason']) if event.get('reason') else '')])
    for note in r.get('issues',[]):rows.append(['Evidence note','-','-','-',note])
    for error in r.get('rubbing_recovery_errors',[]):rows.append(['Recovery error','-','-','-',str(error)])
    return rows


def render_report(path,directory,jewels,tests,recording_problems,prediction_text,asset_resolver=None,simple_output=False):
    directory=Path(directory).resolve()
    resolve=asset_resolver or (lambda p:Path(p) if Path(p).is_absolute() else directory/Path(p))
    c=canvas.Canvas(str(path),pagesize=A4)
    c.setTitle('JewelTracker | Continuous SOP Audit')
    c.setAuthor('EMBSYS / JEWELTRACKER')
    page=0
    def text(x,y,value,size=9,color=INK,bold=False):
        c.setFillColor(color);c.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
        c.drawString(x,H-y,str(value))
    def lines(x,y,value,width=80,size=8,limit=None):
        rows=[]
        for paragraph in str(value).splitlines() or ['']:
            rows.extend(textwrap.wrap(paragraph,width=width) or [''])
        for row in rows[:limit] if limit else rows:
            text(x,y,row,size);y+=size+3
        return y
    def header(title,subtitle):
        nonlocal page
        page+=1
        c.setFillColor(NAVY);c.rect(0,H-103,W,103,fill=1,stroke=0)
        text(28,25,'EMBSYS / JEWELTRACKER',9,white,True)
        text(28,57,title,21,white,True)
        text(28,80,subtitle,9,white)
        text(28,95,session_period(tests),7,white)
        c.setFillColor(TEAL);c.rect(0,H-106,W,3,fill=1,stroke=0)
    def footer():
        text(28,H-20,'Session '+directory.name,7)
        text(W-120,H-20,f'VISUAL AUDIT | {page:02d}',7)
        c.showPage()
    def box(x,y,w,h,fill=white):
        c.setFillColor(fill);c.setStrokeColor(BORDER)
        c.roundRect(x,H-y-h,w,h,8,fill=1,stroke=1)
    def audit_table(headings, rows, widths, y, title):
        style=ParagraphStyle('audit_cell',fontName='Helvetica',fontSize=7,leading=10,textColor=INK,
                             splitLongWords=True)
        cells=[[Paragraph(escape(str(value)).replace('\n','<br/>'),style) for value in row]
               for row in [headings]+rows]
        table=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),BORDER),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,PALE]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.4,BORDER),
            ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        while table is not None:
            available=H-65-y
            _,height=table.wrap(W-56,available)
            if height<=available:
                table.drawOn(c,28,H-y-height)
                return y+height+18
            pieces=table.split(W-56,available)
            if pieces:
                first=pieces[0];_,height=first.wrap(W-56,available)
                first.drawOn(c,28,H-y-height)
                table=pieces[1] if len(pieces)>1 else None
            elif y==130:
                raise ValueError('Audit table row exceeds a report page')
            footer();header(title+' | Continued','Per-jewel audit tables');y=130
        return y
    def picture(p,x,y,w,h,missing='Evidence not captured'):
        if p and resolve(p).is_file():
            reader=ImageReader(str(resolve(p)));iw,ih=reader.getSize();scale=min(w/iw,h/ih)
            rw,rh=iw*scale,ih*scale
            c.drawImage(reader,x+(w-rw)/2,H-y-h+(h-rh)/2,width=rw,height=rh,mask='auto')
        else:
            c.setFillColor(PALE);c.rect(x,H-y-h,w,h,fill=1,stroke=0)
            lines(x+9,y+h/2,missing,width=max(20,int(w/4.5)),size=9,limit=4)
    def panel(x,y,w,h,title,p,caption,second=None,status=None,verified=False):
        box(x,y,w,h)
        text(x+11,y+20,title,11,INK,True)
        image_top=34
        if status is not None:
            text(x+11,y+39,status,10,GREEN if verified else AMBER,True)
            image_top=50
        if second:
            picture(p,x+10,y+image_top,(w-25)/2,h-image_top-52)
            picture(second,x+15+(w-25)/2,y+image_top,(w-25)/2,h-image_top-52)
        else:picture(p,x+10,y+image_top,w-20,h-image_top-52)
        lines(x+11,y+h-40,caption,width=int((w-20)/4.1),size=7,limit=3)
    def link(label,path,x,y,width=170):
        text(x,y,label,8,TEAL)
        if path and resolve(path).is_file():
            target=Path(path) if Path(path).is_absolute() else directory/Path(path)
            if simple_output:
                from urllib.parse import quote
                url=quote(target.relative_to(directory).as_posix())
            else:url=target.absolute().as_uri()
            c.linkURL(url,(x,H-y-3,x+width,H-y+10),relative=0,thickness=0)
    def clip_button(label,clip,x,y):
        path=(clip or {}).get('path')
        available=bool(path and resolve(path).is_file())
        box(x,y,120,25,white)
        if available:
            link('Open '+label+' clip',path,x+8,y+16,width=104)
        else:
            text(x+8,y+16,label+' unavailable',7,AMBER)
    header('Gold Loan SOP | Session Audit','Actual session evidence | Material prediction and process verification are separate')
    completed=sum(bool(r.get('capture_complete')) for r in tests)
    summary=[('JEWELS',len(jewels)),('TEST CYCLES',len(tests)),('ACID CAPTURES',completed)]
    card_width=(W-56-20)/3
    for i,(label,value) in enumerate(summary):
        x=28+i*(card_width+10);box(x,124,card_width,61,PALE);text(x+10,143,label,8);text(x+10,171,value,22,TEAL,True)
    by_id={j['id']:j for j in jewels}
    text(28,212,'TEST-LEVEL OUTCOMES',10,INK,True)
    y=232
    for r in tests:
        if y>H-150:
            footer();header('Session Audit | Continued','Test-level outcomes');y=132
        jid=f"J{r['jewel_id']}" if r.get('jewel_id') is not None else 'Unassigned'
        box(28,y,W-56,100,PALE)
        text(40,y+19,jid,12,TEAL,True)
        picture(by_id.get(r.get('jewel_id'),{}).get('image'),76,y+7,48,47,'No photo')
        text(135,y+19,audit_outcome(r),10,INK,True)
        lines(135,y+36,'Acid model: '+prediction_text(r.get('prediction')),width=53,size=8,limit=2)
        lines(135,y+62,'Contact: '+r.get('rubbing_contact_status','Not observed')+' | Mask: '+r.get('mask_evidence_status','Not established'),width=60,size=7,limit=2)
        text(40,y+90,'Start: '+when(r.get('started'))+' | End: '+when(r.get('finished'))+' IST',8)
        clip_button('rubbing',r.get('rubbing_clip'),W-160,y+13)
        clip_button('acid',r.get('acid_clip'),W-160,y+48)
        if r.get('compact_report'):
            link('Open full jewel report',r['compact_report'],135,y+79,width=145)
        y+=110
    if not tests:lines(28,y,'No jewellery test cycle was established.' if simple_output else 'No jewellery test cycle was established. Session frames and tracked jewel visibility are preserved.',90)
    y=max(y+15,580)
    if y>H-100:
        footer();header('Session Audit | Evidence legend','Read predictions and evidence quality independently');y=138
    lines(28,y,'Gold: jewellery contact mask. Magenta: rubbed-mark detection rectangle (read verification status). Reaction images contain original camera pixels.',100,size=9)
    lines(28,y+48,'Unverified, rejected and missing evidence never becomes a confirmed SOP pass. Model output is a prediction; it is not physical ground truth.',100,size=9)
    if not simple_output:link('Open session manifest',directory/'session.json',28,H-45)
    footer()
    for r in tests:
        if r.get('compact_report'):continue  # Full evidence is embedded in its immutable jewel PDF.
        jid=f"J{r['jewel_id']}" if r.get('jewel_id') is not None else 'Unassigned jewel'
        header(jid+' | Rubbing and Acid Evidence','Start: '+when(r.get('started'))+' | End: '+when(r.get('finished'))+' IST | '+audit_outcome(r))
        picture(by_id.get(r.get('jewel_id'),{}).get('image'),28,118,65,66,'No original image')
        text(105,133,'ORIGINAL JEWELLERY / CYCLE',8)
        text(105,151,jid+' | '+str(r.get('cycle_id','legacy record')),10,INK,True)
        text(105,169,'Acid model: '+prediction_text(r.get('prediction')),11,TEAL,True)
        verified_contact=r.get('rubbing_verified',False)
        target=r.get('acid_target_status')
        acid=('OVERLAPPING' if target=='ACID_OVERLAPS_NEW_RUB' else 'PARTIAL OVERLAP' if target=='ACID_PARTIALLY_OVERLAPS_NEW_RUB' else 'DIFFERENT LOCATION' if target=='ACID_AT_DIFFERENT_LOCATION' else 'COMPARISON UNAVAILABLE' if r.get('acid_onset') is not None else 'NOT CONFIRMED')
        contact_caption=when(r.get('rub_timestamp'))+' | '+r.get('rubbing_contact_reason','Contact unavailable')
        mask_caption=when(r.get('mask_timestamp',r.get('candidate_mask_timestamp',r.get('mask_completed_at'))))+' | '+r.get('mask_reason','New mask unavailable')
        panel(28,196,264,235,'01 / Rubbing contact',r.get('rub_image'),contact_caption,
              status='VERIFIED' if verified_contact else 'UNVERIFIED',verified=verified_contact)
        verified_mask=r.get('mask_evidence_status')=='VERIFIED'
        panel(304,196,264,235,'02 / New rubbed-mark detection',r.get('mask_image') or r.get('candidate_mask_image'),mask_caption,
              status='VERIFIED' if verified_mask else 'NOT VERIFIED',verified=verified_mask)
        mask_caption+=' | '+str(r.get('rub_evidence_source',r.get('evidence_source','LIVE / RECORDED MODEL')))
        acid_caption=when(r.get('acid_placement_timestamp',r.get('acid_onset')))+' | '+(target or 'Fresh application not established')
        acid_caption+=' | '+str(r.get('acid_evidence_source',r.get('evidence_source','LIVE / RECORDED MODEL')))
        verified_acid=target=='ACID_OVERLAPS_NEW_RUB'
        panel(28,444,264,238,'03 / Acid placement',r.get('acid_rub_comparison_image') or r.get('acid_detection_image'),acid_caption,
              status='VERIFIED - OVERLAPPING' if verified_acid else acid,verified=verified_acid)
        reaction_caption=when(r.get('reaction_first_timestamp'))+' / '+when(r.get('reaction_last_timestamp'))+' | '+str(round(r.get('usable_reaction_seconds',0),2))+' s usable of 5.0 s requested'
        panel(304,444,264,238,'04 / Reaction observation',r.get('reaction_first_image') or r.get('acid_image'),reaction_caption,second=r.get('acid_image'))
        missed=sum(max(0,v['end']-v['start']) for v in r.get('visibility',[]) if v.get('jewel_id')==r.get('jewel_id') and v['status']=='MISSED')
        box(28,696,540,103,PALE)
        text(39,713,'AUDIT OUTCOME: '+audit_outcome(r),10,INK,True)
        text(39,730,'Pickup '+when(r.get('started'))+' | Baseline '+when(r.get('baseline_at'))+' | End '+when(r.get('finished')),8)
        text(39,745,f'Tracking missed: {missed:.2f} s | Close reason: '+str(r.get('close_reason','Not recorded')),8)
        reason=(r.get('prediction') or {}).get('invalid_reason') or '; '.join(r.get('issues',[])) or r.get('rubbing_contact_reason','')
        lines(39,760,reason,width=113,size=7,limit=2)
        if not simple_output:link('Original clips / exact evidence in appendix',r.get('record_path'),39,790)
        footer()
    # Full reasons/timestamps are never silently truncated to fit a visual card.
    for r in tests:
        if r.get('compact_report'):continue
        jid=f"J{r['jewel_id']}" if r.get('jewel_id') is not None else 'Unassigned'
        header(jid+' | Audit Details','Exact timestamps, visibility intervals, confidence and failure reasons')
        y=130
        y=audit_table(['Event','Start','End','Duration','Status / reason'],event_rows(r),[105,65,65,80,W-371],y,jid+' | Event Log')
        if y>H-150:
            footer();header(jid+' | Evidence Details','Recorded evidence and verification');y=130
        details=[]
        if r.get('backup'):
            details.append('Backup verification: '+str(r['backup'].get('status'))+' | '+str(r['backup'].get('reason')))
        if r.get('performance_timings'):details.append('Processing seconds and counts: '+str(r['performance_timings']))
        if r.get('sop_review_status'):details.append('SOP evidence status: '+r['sop_review_status'])
        for key in ('cycle_id','revision','alignment_mode','rubbed_model','rubbed_region_kind','before_status','ph_status','close_reason','contact_close_reason','reaction_coverage_status','rubbing_status','rubbing_contact_status','rubbing_contact_reason','mask_evidence_status','mask_reason','acid_status','fresh_acid_status','acid_target_status','application_verification','application_timing_status','clear_patch_frames'):
            if r.get(key) is not None:details.append(key.replace('_',' ').capitalize()+': '+str(r[key]))
        for key in ('before_timestamp','started','rubbing_started_at','rubbing_trigger_confirmed_at','rub_timestamp','returned','contact_closed_at','mask_timestamp','candidate_mask_timestamp','mask_completed_at','mask_decision_at','baseline_at','first_acid_candidate_at','application_candidate_at','application_verified_at','acid_onset','acid_placement_timestamp','finished'):
            details.append(key.replace('_',' ').capitalize()+': '+when(r.get(key)))
        details.append('Rub source frame: '+str(r.get('rub_source_frame','Not captured')))
        details.append('Acid source frame: '+str(r.get('acid_source_frame','Not captured')))
        if r.get('acid_area_status'):details.append('Acid area: '+r['acid_area_status']+'; model-supported visual estimate, not a chemical boundary.')
        if r.get('acid_detector_model'):
            details.append('Acid detector: '+r['acid_detector_model']+' | '+str(r.get('acid_detector_backend')))
        if r.get('rub_display_kind')=='EDGESAM_SEGMENTATION':
            details.append('Rubbed-mark display: EdgeSAM backup pixel segmentation with recorded persistence and contact support.')
        elif r.get('rub_display_kind'):
            details.append('Rubbed-mark display: detection rectangle, not pixel segmentation. Before/after and persistence checks are recorded separately.')
        if r.get('previous_acid_state'):
            state=r['previous_acid_state']
            details.append('Clear stone before pickup: '+state['status']+' | Checked '+when(state.get('checked_at')))
        if r.get('classification_roi_source'):
            details.append('Classifier input: '+r['classification_roi_source']+'; independent of the displayed acid area.')
            details.append('Classifier region verification: '+str(r.get('classification_roi_status','Not recorded')))
            if r.get('classification_advisory_status'):
                details.append('Additional detector checks (advisory): '+r['classification_advisory_status'])
            if r.get('confirmed_classification_region'):
                details.append('Confirmed classifier source: '+when(r['confirmed_classification_region']['timestamp'])+' | Box '+str(r['confirmed_classification_region']['bbox'])+'; original source pose retained.')
            details.append('Classifier detector input: '+str(r.get('classification_detector_input','Not recorded')))
            details.append('Classifier baseline window: '+str(r.get('classification_baseline_window'))+' | Pouring window: '+str(r.get('classification_transition_window')))
        if r.get('acid_search_stopped_at') is not None:details.append('Acid search stopped at next pickup: '+when(r['acid_search_stopped_at']))
        details.append('Acid area describes the visible patch after confirmed application; any prior-residue warning remains separate. It does not identify the chemical boundary of a fresh dose.')
        for reason,count in r.get('acid_area_rejections',{}).items():details.append(f'Acid-area check: {reason} ({count} checks).')
        if r.get('acid_new_mask_overlap') is not None:details.append(f"Estimated acid/new-rub overlap: {r['acid_new_mask_overlap']:.1%} of smaller region.")
        if r.get('acid_estimated_rub_coverage') is not None:details.append(f"Rubbed mask covered by estimated acid area: {r['acid_estimated_rub_coverage']:.1%} (80% for overlapping; any lower positive coverage is partial).")
        if r.get('rub_previous_acid_overlap',0)>0:details.append('New rubbing overlaps previous acid; contamination possible.')
        if (r.get('prediction') or {}).get('invalid_reason'):details.append('INVALID reason: '+r['prediction']['invalid_reason'])
        rows=[list(item.split(': ',1)) if ': ' in item else ['Evidence',item] for item in details]
        y=audit_table(['Evidence / field','Recorded value'],rows,[160,W-216],y,jid+' | Evidence Details')
        for key in (('rubbing_clip','acid_clip') if simple_output else ('rubbing_clip','acid_clip','acid_candidate_clip')):
            clip=r.get(key) or {}
            if y>H-75:footer();header(jid+' | Evidence Links','Original saved video');y=130
            link(key.replace('_',' ').title(),clip.get('path'),28,y);y+=20
        if not simple_output:link('Per-cycle JSON manifest',r.get('record_path'),28,min(y,H-55))
        footer()
    if recording_problems:
        header('Recording Exceptions','Capture failures remain separate from model predictions')
        y=135
        for p in recording_problems:
            if y>H-90:footer();header('Recording Exceptions | Continued','Capture audit');y=135
            y=lines(28,y,f"{p['reason']}: {p['end']-p['start']:.2f} s, {p['count']} observations.",104)+10
            if p.get('provisional_clip',{}):
                link('Unverified pickup video (not a confirmed test)',p['provisional_clip'].get('path'),28,y)
                y+=20
        footer()
    c.save()
