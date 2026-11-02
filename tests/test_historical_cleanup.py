import copy,io,json,tempfile,unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import test_evidence_retention as fixtures
from evidence_archive import ArchiveManager,RetentionPolicy,digest,durable_json,validate_video,encoder_available,SessionGuard
from session_evidence import atomic_json,write_report
from historical_cleanup import scan_session,migrate,per_test_sizes,main
from compact_evidence import encode_compact

class HistoricalCleanupTests(unittest.TestCase):
    def fixture(self,root,**kw):return fixtures.EvidenceRetentionTests().fixture(root,**kw)

    def test_dry_run_scans_all_legacy_directories_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);root=parent/'sessions';root.mkdir()
            good=root/'good';good.mkdir();self.fixture(good)
            (root/'unfinalized').mkdir();(root/'missing-json').mkdir()
            before={str(p):digest(p) for p in root.rglob('*') if p.is_file()}
            report=parent/'report.json'
            with patch('historical_cleanup.active_legacy_session',return_value=False),patch('sys.argv',['cleanup',str(root),'--dry-run','--report',str(report)]),redirect_stdout(io.StringIO()):main()
            d=json.loads(report.read_text());self.assertEqual(3,d['summary']['sessions_scanned'])
            self.assertEqual(1,d['summary']['eligible_sessions']);self.assertEqual(2,d['summary']['skipped_sessions'])
            self.assertEqual(before,{str(p):digest(p) for p in root.rglob('*') if p.is_file()})
            self.assertTrue(d['sessions'][0]['proposed_deletions'])

    def test_incomplete_or_active_sessions_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,r=self.fixture(root)
            with SessionGuard(root):self.assertFalse(scan_session(root,RetentionPolicy())['eligible'])
            r['capture_complete']=False;r['close_reason']='SESSION_STOP'
            atomic_json(root/'test_1/manifest.json',r);write_report(root,[dict(id=1,image=str(root/'J1.jpg'))],[r])
            self.assertFalse(scan_session(root,RetentionPolicy())['eligible'])
            with self.assertRaises(RuntimeError):ArchiveManager(root,RetentionPolicy(require_server_ack=False)).finalize(True)
            self.assertTrue((root/'frames').exists())

    def test_two_megabyte_budget_is_per_test_not_session_average(self):
        final=Path('/archive')
        data=dict(tests=[dict(cycle_id='a',jewel_id=1,record_path='/archive/a.json',video='/archive/a.mp4'),
                         dict(cycle_id='b',jewel_id=2,record_path='/archive/b.json',video='/archive/b.mp4')])
        files={'a.mp4':dict(bytes=2_400_000),'b.mp4':dict(bytes=100_000),'a.json':dict(bytes=1000),
               'b.json':dict(bytes=1000),'report.pdf':dict(bytes=10_000),'session.json':dict(bytes=1000)}
        sizes=per_test_sizes(data,files,final,2_000_000)
        self.assertTrue(sizes[0]['size_exception']);self.assertFalse(sizes[1]['size_exception'])

    def test_complete_legacy_session_migrates_and_preserves_sop_and_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,original=self.fixture(root,prediction=True)
            (root/'session_closed.json').unlink()
            with patch('historical_cleanup.active_legacy_session',return_value=False),patch('evidence_archive.encoder_available',return_value=None):
                result=migrate(root,RetentionPolicy(require_server_ack=False,archive_codec='mjpeg'))
            self.assertEqual('cleaned',result['status'])
            self.assertFalse((root/'frames').exists());self.assertFalse((root/'frames.jsonl').exists())
            r=json.loads((root/'session.json').read_text())['tests'][0]
            for key in ('prediction','issues','capture_complete','acid_onset','returned','started','finished'):
                self.assertEqual(original[key],r[key])
            self.assertEqual(original['acid_clip']['timestamps'],r['acid_clip']['timestamps'])
            for key in ('rubbing_clip','acid_clip'):validate_video(r[key]['path'],r[key])
            self.assertTrue((root/'report.pdf').read_bytes().startswith(b'%PDF'))
            self.assertTrue(result['test_sizes'])

    def test_required_ack_and_bank_original_retention_never_delete_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            with patch('evidence_archive.encoder_available',return_value=None):
                result=ArchiveManager(root).finalize(True)
            self.assertEqual('awaiting_server_ack',result['status']);self.assertTrue((root/'frames').exists())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            policy=RetentionPolicy(require_server_ack=False,original_min_retention_days=7)
            with patch('evidence_archive.encoder_available',return_value=None),patch('evidence_archive.time.time',return_value=120):
                result=ArchiveManager(root,policy).finalize(True)
            self.assertEqual('bank_original_retention_hold',result['status']);self.assertTrue((root/'frames').exists())

    def test_size_exception_keeps_exact_essential_original_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,r=self.fixture(root)
            image=np.random.default_rng(5).integers(0,256,(1000,1000,3),dtype=np.uint8)
            path=root/'test_1/essential_original.png';cv2.imwrite(str(path),image);sha=digest(path)
            r['essential_original']=str(path)
            atomic_json(root/'test_1/manifest.json',r);write_report(root,[dict(id=1,image=str(root/'J1.jpg'))],[r])
            with patch('evidence_archive.encoder_available',return_value=None):
                result=ArchiveManager(root,RetentionPolicy(require_server_ack=False)).finalize(True)
            self.assertTrue(result['status']=='cleaned')
            commit=json.loads((root/'archive_commit.json').read_text())
            manifest=json.loads((root/'archives'/commit['archive_id']/'archive_manifest.json').read_text())
            self.assertTrue(manifest['test_sizes'][0]['size_exception'])
            kept=json.loads((root/'session.json').read_text())['tests'][0]['essential_original']
            self.assertEqual(sha,digest(kept))

    def test_missing_or_corrupted_source_prevents_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,r=self.fixture(root)
            Path(r['acid_clip']['path']).write_bytes(b'broken')
            self.assertFalse(scan_session(root,RetentionPolicy(),check_video=True)['eligible'])
            with patch('evidence_archive.encoder_available',return_value=None),self.assertRaises(OSError):migrate(root,RetentionPolicy(require_server_ack=False))
            self.assertTrue((root/'frames').exists())

    def test_resume_required_after_interrupted_publish_and_safe_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);policy=RetentionPolicy(require_server_ack=False)
            manager=ArchiveManager(root,policy)
            with patch('evidence_archive.encoder_available',return_value=None),patch.object(manager,'_publish',side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):manager.finalize(True)
            with self.assertRaises(RuntimeError):migrate(root,policy,resume=False)
            self.assertEqual('cleaned',migrate(root,policy,resume=True)['status'])
            self.assertEqual(1,len(list((root/'archives').iterdir())))

    def test_actual_h264_preserves_duration_original_timestamps_and_quality_floor(self):
        exe=encoder_available()
        if not exe:self.skipTest('FFmpeg/libx264 not installed')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.avi'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),10,(1920,1080))
            for i in range(12):
                image=np.full((1080,1920,3),80,np.uint8)
                cv2.rectangle(image,(700+i*10,400),(760+i*10,650),(20,150,200),-1);writer.write(image)
            writer.release()
            meta=dict(frames=12,timestamps=[100+i*.1 for i in range(12)],playback_seconds=1.2,
                      playback_fps=10,actual_start=100,actual_end=101.1,requested_start=100,requested_end=101.2)
            target,codec,media=encode_compact(source,root/'optimized',meta,RetentionPolicy(),exe,300_000)
            self.assertEqual('h264',codec);self.assertLessEqual(media['width'],1600)
            self.assertGreaterEqual(max(media['width'],media['height']),1280)
            self.assertGreaterEqual(media['ssim'],.95)
            self.assertLess(abs(media['duration_seconds']-1.2),.25)
            validate_video(target,dict(meta,archive_media=media))
            self.assertEqual(12,len(meta['timestamps']))

    def test_adaptive_frame_mapping_matches_actual_encoded_pixels(self):
        exe=encoder_available()
        if not exe:self.skipTest('FFmpeg/libx264 not installed')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.avi'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),20,(1280,720))
            for i in range(20):writer.write(np.full((720,1280,3),30+i*5,np.uint8))
            writer.release()
            target,_,media=encode_compact(source,root/'out',dict(playback_seconds=1.),RetentionPolicy(),exe,300_000)
            self.assertEqual(15,media['frames']);self.assertEqual(15,len(media['source_frame_indices']))
            cap=cv2.VideoCapture(str(target))
            try:
                for i in media['source_frame_indices']:
                    ok,frame=cap.read();self.assertTrue(ok)
                    self.assertAlmostEqual(30+i*5,float(np.median(frame)),delta=3)
            finally:cap.release()

    def test_corrupt_pdf_and_corrupt_resume_state_do_not_abort_other_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);root=parent/'sessions';root.mkdir()
            bad=root/'bad';bad.mkdir();self.fixture(bad)
            (bad/'report.pdf').write_bytes(b'%PDF-truncated')
            (bad/'archive_state.json').write_text('{bad-json')
            good=root/'good';good.mkdir();self.fixture(good)
            config=parent/'policy.json';durable_json(config,dict(require_server_ack=False,archive_codec='mjpeg'))
            report=parent/'result.json'
            with patch('historical_cleanup.active_legacy_session',return_value=False),patch('evidence_archive.encoder_available',return_value=None),patch('sys.argv',['cleanup',str(root),'--execute','--resume','--config',str(config),'--report',str(report)]),redirect_stdout(io.StringIO()):main()
            result=json.loads(report.read_text())
            self.assertEqual(1,result['summary']['optimized_sessions']);self.assertEqual(1,result['summary']['skipped_sessions'])
            self.assertTrue((bad/'frames').exists());self.assertFalse((good/'frames').exists())

    def test_malformed_session_types_and_timestamps_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            for content in ('[]','null','{"tests":[null]}'):
                (root/'session.json').write_text(content)
                self.assertFalse(scan_session(root,RetentionPolicy())['eligible'])

    def test_corrupt_selected_image_or_copy_preserves_raw(self):
        import shutil
        for corrupt_source in (True,False):
            with self.subTest(corrupt_source=corrupt_source),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);self.fixture(root);original_copy=shutil.copyfile
                if corrupt_source:(root/'J1.jpg').write_bytes(b'broken image')
                def copy_image(source,dest,*args,**kwargs):
                    result=original_copy(source,dest,*args,**kwargs)
                    if not corrupt_source and Path(dest).name=='J1.jpg':Path(dest).write_bytes(b'bad copy')
                    return result
                with patch('evidence_archive.encoder_available',return_value=None),patch('evidence_archive.shutil.copyfile',side_effect=copy_image):
                    with self.assertRaises(OSError):migrate(root,RetentionPolicy(require_server_ack=False))
                self.assertTrue((root/'frames').exists());self.assertFalse((root/'archive_commit.json').exists())

    def legacy_archive(self,root):
        self.fixture(root,prediction=True)
        policy=RetentionPolicy(require_server_ack=False)
        with patch('evidence_archive.encoder_available',return_value=None):ArchiveManager(root,policy).finalize(True)
        commit=json.loads((root/'archive_commit.json').read_text())
        path=root/'archives'/commit['archive_id']/'archive_manifest.json'
        manifest=json.loads(path.read_text());manifest['storage_profile']='legacy_mjpeg'
        durable_json(path,manifest);commit['manifest_sha256']=digest(path)
        durable_json(root/'archive_commit.json',commit)
        return commit

    def test_previously_committed_archives_are_repacked_and_old_generation_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=self.legacy_archive(root)
            with patch('evidence_archive.encoder_available',return_value=None):
                result=migrate(root,RetentionPolicy(require_server_ack=False))
            new=json.loads((root/'archive_commit.json').read_text())
            self.assertEqual('cleaned',result['status']);self.assertNotEqual(old['archive_id'],new['archive_id'])
            self.assertFalse((root/'archives'/old['archive_id']).exists())
            ArchiveManager(root).validate_commit(republish=False)
            self.assertEqual(1,len(list((root/'archives').iterdir())))

    def test_current_mjpeg_fallback_archive_is_upgraded_when_encoder_available(self):
        if not encoder_available():self.skipTest('FFmpeg/libx264 not installed')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            with patch('evidence_archive.encoder_available',return_value=None):
                ArchiveManager(root,RetentionPolicy(require_server_ack=False)).finalize(True)
            old=json.loads((root/'archive_commit.json').read_text())
            result=migrate(root,RetentionPolicy(require_server_ack=False))
            new=json.loads((root/'archive_commit.json').read_text())
            manifest=json.loads((root/'archives'/new['archive_id']/'archive_manifest.json').read_text())
            self.assertNotEqual(old['archive_id'],new['archive_id'])
            self.assertTrue(all(v['codec']=='h264' for v in manifest['videos']))
            self.assertEqual('cleaned',result['status'])

    def test_repacked_archive_requires_new_ack_and_keeps_old_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=self.legacy_archive(root)
            with patch('evidence_archive.encoder_available',return_value=None):result=migrate(root,RetentionPolicy())
            self.assertEqual('awaiting_server_ack',result['status'])
            self.assertTrue((root/'archives'/old['archive_id']).exists())

    def test_repack_cleanup_resumes_after_partial_tomb_deletion(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=self.legacy_archive(root);original_rmtree=shutil.rmtree
            def interrupted(path,*args,**kw):
                path=Path(path)
                if path.parent.name=='.cleanup':
                    old_path=path/'archives'/old['archive_id']
                    if old_path.exists():original_rmtree(old_path)
                    raise OSError('power loss after removing obsolete archive tomb')
                return original_rmtree(path,*args,**kw)
            with patch('evidence_archive.encoder_available',return_value=None),patch('evidence_archive.shutil.rmtree',side_effect=interrupted):
                with self.assertRaises(OSError):migrate(root,RetentionPolicy(require_server_ack=False))
            self.assertEqual('cleaned',migrate(root,RetentionPolicy(require_server_ack=False),resume=True)['status'])
            ArchiveManager(root).validate_commit(republish=False)

    def test_failed_quality_check_preserves_raw_and_canonical_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);before=(root/'session.json').read_bytes()
            with patch('evidence_archive.encoder_available',return_value='ffmpeg'),patch('compact_evidence.encode_compact',side_effect=OSError('quality floor failed')):
                with self.assertRaises(OSError):migrate(root,RetentionPolicy(require_server_ack=False))
            self.assertTrue((root/'frames').exists());self.assertEqual(before,(root/'session.json').read_bytes())

    def test_truncated_interval_and_nonfinite_metadata_are_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,r=self.fixture(root)
            r['returned']=109
            atomic_json(root/'test_1/manifest.json',r);write_report(root,[dict(id=1,image=str(root/'J1.jpg'))],[r])
            with patch('evidence_archive.encoder_available',return_value=None),self.assertRaises(OSError):
                ArchiveManager(root,RetentionPolicy(require_server_ack=False)).finalize(True)
            self.assertTrue((root/'frames').exists())
            bad=dict(r['acid_clip']);bad['timestamps'][0]=float('nan')
            with self.assertRaises(OSError):validate_video(bad['path'],bad)

if __name__=='__main__':unittest.main()
