"""Visual audit PDF. All evidence and outcomes come from the saved manifest."""
from datetime import datetime
from pathlib import Path
import textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4

NAVY=HexColor('#0c1c2b');TEAL=HexColor('#008b8b');INK=HexColor('#243546')
PALE=HexColor('#f3f6f9');BORDER=HexColor('#d9e2ea');AMBER=HexColor('#a7600a')
W,H=A4


def audit_outcome(r):
    if not r.get('capture_complete') or not r.get('prediction'):return 'INCOMPLETE'
    mask_verified=r.get('mask_evidence_status')=='VERIFIED'
    if (r.get('rubbing_verified') and mask_verified and r.get('acid_target_status')=='ACID_OVERLAPS_NEW_RUB'
            and not r.get('rub_previous_acid_overlap',0) and not r.get('issues')
            and r['prediction']['prediction']!='invalid'):
        return 'EVIDENCE VERIFIED'
    return 'REVIEW REQUIRED'


def when(value):
    return datetime.fromtimestamp(value).strftime('%H:%M:%S.%f')[:-3] if value is not None else 'Not captured'


def render_report(path,directory,jewels,tests,recording_problems,prediction_text):
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
        c.setFillColor(TEAL);c.rect(0,H-106,W,3,fill=1,stroke=0)
    def footer():
        text(28,H-20,'Session '+directory.name,7)
        text(W-120,H-20,f'VISUAL AUDIT | {page:02d}',7)
        c.showPage()
    def box(x,y,w,h,fill=white):
        c.setFillColor(fill);c.setStrokeColor(BORDER)
        c.roundRect(x,H-y-h,w,h,8,fill=1,stroke=1)
    def picture(p,x,y,w,h,missing='Evidence not captured'):
        if p and Path(p).is_file():
            reader=ImageReader(str(p));iw,ih=reader.getSize();scale=min(w/iw,h/ih)
            rw,rh=iw*scale,ih*scale
            c.drawImage(reader,x+(w-rw)/2,H-y-h+(h-rh)/2,width=rw,height=rh,mask='auto')
        else:
            c.setFillColor(PALE);c.rect(x,H-y-h,w,h,fill=1,stroke=0)
            lines(x+9,y+h/2,missing,width=max(20,int(w/4.5)),size=9,limit=4)
    def card(x,y,w,title,value,detail):
        box(x,y,w,68)
        text(x+10,y+16,title.upper(),8)
        text(x+10,y+35,value,10,TEAL if value in ('VERIFIED','FRESH / ON MARK') else AMBER,True)
        lines(x+10,y+49,detail,width=int(w/4.1),size=7,limit=2)
    def panel(x,y,w,h,title,p,caption,second=None):
        box(x,y,w,h)
        text(x+11,y+20,title,11,INK,True)
        if second:
            picture(p,x+10,y+34,(w-25)/2,h-86)
            picture(second,x+15+(w-25)/2,y+34,(w-25)/2,h-86)
        else:picture(p,x+10,y+34,w-20,h-86)
        lines(x+11,y+h-40,caption,width=int((w-20)/4.1),size=7,limit=3)
    def link(label,path,x,y):
        text(x,y,label,8,TEAL)
        if path and Path(path).is_file():
            c.linkURL(Path(path).absolute().as_uri(),(x,H-y-3,x+170,H-y+10),relative=0,thickness=0)
    header('Gold Loan SOP | Session Audit','Actual session evidence | Material prediction and process verification are separate')
    completed=sum(bool(r.get('capture_complete')) for r in tests)
    verified=sum(audit_outcome(r)=='EVIDENCE VERIFIED' for r in tests)
    for i,(label,value) in enumerate([('JEWELS',len(jewels)),('TEST CYCLES',len(tests)),('ACID CAPTURES',completed),('VERIFIED SOP',verified)]):
        x=28+i*137;box(x,124,128,61,PALE);text(x+10,143,label,8);text(x+10,171,value,22,TEAL,True)
    text(28,212,'TEST-LEVEL OUTCOMES',10,INK,True)
    y=232
    for r in tests:
        if y>H-110:
            footer();header('Session Audit | Continued','Test-level outcomes');y=132
        jid=f"J{r['jewel_id']}" if r.get('jewel_id') is not None else 'Unassigned'
        box(28,y,W-56,61,PALE)
        text(40,y+19,jid,12,TEAL,True)
        text(135,y+19,audit_outcome(r),10,INK,True)
        text(135,y+36,'Acid model: '+prediction_text(r.get('prediction')),9)
        lines(135,y+49,'Contact: '+r.get('rubbing_contact_status','Not observed')+' | Mask: '+r.get('mask_evidence_status','Not established'),width=83,size=7,limit=1)
        y+=71
    if not tests:lines(28,y,'No jewellery test cycle was established. Session frames and tracked jewel visibility are preserved.',90)
    y=max(y+15,580)
    if y>H-100:
        footer();header('Session Audit | Evidence legend','Read predictions and evidence quality independently');y=138
    lines(28,y,'Cyan / gold: jewellery contact. Magenta: candidate or verified new mark (read status). Cyan acid box: detector footprint, not chemical segmentation.',100,size=9)
    lines(28,y+48,'Unverified, rejected and missing evidence never becomes a confirmed SOP pass. Model output is a prediction; it is not physical ground truth.',100,size=9)
    link('Open session manifest',directory/'session.json',28,H-45)
    footer()
    by_id={j['id']:j for j in jewels}
    for r in tests:
        jid=f"J{r['jewel_id']}" if r.get('jewel_id') is not None else 'Unassigned jewel'
        header(jid+' | Rubbing and Acid Evidence','Timestamped source frames | '+audit_outcome(r))
        picture(by_id.get(r.get('jewel_id'),{}).get('image'),28,118,65,66,'No original image')
        text(105,133,'ORIGINAL JEWELLERY / CYCLE',8)
        text(105,151,jid+' | '+str(r.get('cycle_id','legacy record')),10,INK,True)
        text(105,169,'Acid model: '+prediction_text(r.get('prediction')),11,TEAL,True)
        verified_contact=r.get('rubbing_verified',False)
        target=r.get('acid_target_status')
        acid=('OVERLAPPING' if target=='ACID_OVERLAPS_NEW_RUB' else 'PARTIAL OVERLAP' if target=='ACID_PARTIALLY_OVERLAPS_NEW_RUB' else 'DIFFERENT LOCATION' if target=='ACID_AT_DIFFERENT_LOCATION' else 'COMPARISON UNAVAILABLE' if r.get('acid_onset') is not None else 'NOT CONFIRMED')
        card(28,196,174,'Rubbing contact','VERIFIED' if verified_contact else 'UNVERIFIED',r.get('rubbing_contact_status','Not observed'))
        card(211,196,174,'New rubbed mask','VERIFIED' if r.get('mask_evidence_status')=='VERIFIED' else 'NOT VERIFIED',r.get('mask_evidence_status','Not established'))
        card(394,196,174,'Acid placement',acid,r.get('acid_status','Not detected'))
        contact_caption=when(r.get('rub_timestamp'))+' | '+r.get('rubbing_contact_reason','Contact unavailable')
        mask_caption=when(r.get('mask_timestamp',r.get('candidate_mask_timestamp',r.get('mask_completed_at'))))+' | '+r.get('mask_reason','New mask unavailable')
        panel(28,279,264,195,'01 / Rubbing contact',r.get('rub_image'),contact_caption)
        panel(304,279,264,195,'02 / New stone mark',r.get('mask_image') or r.get('candidate_mask_image'),mask_caption)
        acid_caption=when(r.get('acid_placement_timestamp',r.get('acid_onset')))+' | '+(target or 'Fresh application not established')
        panel(28,487,264,195,'03 / Acid placement',r.get('acid_rub_comparison_image') or r.get('acid_detection_image'),acid_caption)
        reaction_caption=when(r.get('reaction_first_timestamp'))+' / '+when(r.get('reaction_last_timestamp'))+' | '+str(round(r.get('usable_reaction_seconds',0),2))+' s usable of 5.0 s requested'
        panel(304,487,264,195,'04 / Reaction observation',r.get('reaction_first_image') or r.get('acid_image'),reaction_caption,second=r.get('acid_image'))
        missed=sum(max(0,v['end']-v['start']) for v in r.get('visibility',[]) if v.get('jewel_id')==r.get('jewel_id') and v['status']=='MISSED')
        box(28,696,540,103,PALE)
        text(39,713,'AUDIT OUTCOME: '+audit_outcome(r),10,INK,True)
        text(39,730,'Pickup '+when(r.get('started'))+' | Baseline '+when(r.get('baseline_at'))+' | End '+when(r.get('finished')),8)
        text(39,745,f'Tracking missed: {missed:.2f} s | Close reason: '+str(r.get('close_reason','Not recorded')),8)
        reason=(r.get('prediction') or {}).get('invalid_reason') or '; '.join(r.get('issues',[])) or r.get('rubbing_contact_reason','')
        lines(39,760,reason,width=113,size=7,limit=2)
        link('Original clips / exact evidence in appendix',r.get('record_path'),39,790)
        footer()
    # Full reasons/timestamps are never silently truncated to fit a visual card.
    for r in tests:
        jid=f"J{r['jewel_id']}" if r.get('jewel_id') is not None else 'Unassigned'
        header(jid+' | Audit Details','Exact timestamps, visibility intervals, confidence and failure reasons')
        y=130
        details=[]
        for key in ('cycle_id','revision','alignment_mode','rubbed_model','rubbed_region_kind','before_status','ph_status','close_reason','contact_close_reason','reaction_coverage_status','rubbing_status','rubbing_contact_status','rubbing_contact_reason','mask_evidence_status','mask_reason','acid_status','fresh_acid_status','acid_target_status','application_verification','application_timing_status','clear_patch_frames'):
            if r.get(key) is not None:details.append(key.replace('_',' ').capitalize()+': '+str(r[key]))
        for key in ('before_timestamp','started','rub_timestamp','returned','contact_closed_at','mask_timestamp','candidate_mask_timestamp','mask_completed_at','mask_decision_at','baseline_at','first_acid_candidate_at','application_candidate_at','application_verified_at','acid_onset','acid_placement_timestamp','finished'):
            details.append(key.replace('_',' ').capitalize()+': '+when(r.get(key)))
        details.append('Rub source frame: '+str(r.get('rub_source_frame','Not captured')))
        details.append('Acid source frame: '+str(r.get('acid_source_frame','Not captured')))
        if r.get('acid_area_status'):details.append('Acid area: '+r['acid_area_status']+'; model-supported visual estimate, not a chemical boundary.')
        if r.get('previous_acid_state'):
            state=r['previous_acid_state']
            details.append('Clear stone before pickup: '+state['status']+' | Checked '+when(state.get('checked_at')))
        details.append('Acid area describes the visible patch after confirmed application; any prior-residue warning remains separate. It does not identify the chemical boundary of a fresh dose.')
        for reason,count in r.get('acid_area_rejections',{}).items():details.append(f'Acid-area check: {reason} ({count} checks).')
        if r.get('acid_new_mask_overlap') is not None:details.append(f"Estimated acid/new-rub overlap: {r['acid_new_mask_overlap']:.1%} of smaller region.")
        if r.get('acid_estimated_rub_coverage') is not None:details.append(f"Rubbed mask covered by estimated acid area: {r['acid_estimated_rub_coverage']:.1%} (80% for overlapping; any lower positive coverage is partial).")
        if r.get('rub_previous_acid_overlap',0)>0:details.append('New rubbing overlaps previous acid; contamination possible.')
        if (r.get('prediction') or {}).get('invalid_reason'):details.append('INVALID reason: '+r['prediction']['invalid_reason'])
        details.extend('Evidence note: '+note for note in r.get('issues',[]))
        details.extend('Recovery model error: '+error for error in r.get('rubbing_recovery_errors',[]))
        for event in r.get('acid_candidate_events',[]):
            details.append('Acid candidate '+when(event.get('candidate_at'))+': '+str(event.get('status'))+
                           ' at '+when(event.get('timestamp'))+(' | '+str(event['reason']) if event.get('reason') else ''))
        for v in r.get('visibility',[]):
            details.append(f"J{v['jewel_id']} {v['status']}: {v['end']-v['start']:.2f} seconds ({when(v['start'])} to {when(v['end'])})")
        for item in details:
            needed=(len(textwrap.wrap(item,104)) or 1)*12+6
            if y+needed>H-65:
                footer();header(jid+' | Audit Details Continued','Complete recorded evidence');y=130
            y=lines(28,y,item,width=104,size=8)+6
        for key in ('rubbing_clip','acid_clip','acid_candidate_clip'):
            clip=r.get(key) or {}
            if y>H-75:footer();header(jid+' | Evidence Links','Original saved video');y=130
            link(key.replace('_',' ').title(),clip.get('path'),28,y);y+=20
        link('Per-cycle JSON manifest',r.get('record_path'),28,min(y,H-55))
        footer()
    if recording_problems:
        header('Recording Exceptions','Capture failures remain separate from model predictions')
        y=135
        for p in recording_problems:
            if y>H-90:footer();header('Recording Exceptions | Continued','Capture audit');y=135
            y=lines(28,y,f"{p['reason']}: {p['end']-p['start']:.2f} s, {p['count']} observations.",104)+10
        footer()
    c.save()
