import json,os,shutil,tempfile,threading,time,unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import hmac,hashlib
import cv2
import numpy as np
from evidence_archive import (ArchiveManager,RetentionPolicy,SessionGuard,durable_json,digest,tree_bytes,
                              validate_video,valid_ack,main)
from session_evidence import FrameJournal,write_report,atomic_json

class EvidenceRetentionTests(unittest.TestCase):
    def fixture(self,root,prediction=False):
        folder=root/'test_1';folder.mkdir()
        journal=FrameJournal(root,fps=10)
        image=np.random.default_rng(10).integers(20,130,(64,96,3),dtype=np.uint8)
        for i in range(201):
            journal.offer(image,100+i*.1);journal.queue.join()
        journal.close()
        rubbing=journal.clip(103,106,folder/'pickup_rubbing_return.avi')
        acid=journal.clip(108,115,folder/'acid_2s_before_5s_after.avi')
        cv2.imwrite(str(root/'J1.jpg'),image)
        r=dict(jewel_id=1,cycle_id='test_1',started=104,returned=106,baseline_at=108,
               acid_onset=110,finished=115,close_reason='REACTION_COMPLETE',capture_complete=True,
               reaction_coverage_status='COMPLETE',usable_reaction_seconds=5,
               issues=[],visibility=[],rubbing_status='UNCONFIRMED',
               rubbing_clip=rubbing,acid_clip=acid,acid_candidate_clip=dict(acid),
               rub_source_frame='frames/103.000000.jpg',rub_image=str(root/'J1.jpg'),
               prediction=None,record_path=str(folder/'manifest.json'))
        if prediction:
            r['prediction']=dict(prediction='non_gold',confidence=.9)
            for name in ('baseline','application','reaction'):
                (folder/name).mkdir()
                for i in range(5):cv2.imwrite(str(folder/name/f'{i:04d}.jpg'),image)
        atomic_json(folder/'manifest.json',r)
        write_report(root,[dict(id=1,image=str(root/'J1.jpg'))],[r])
        durable_json(root/'session_closed.json',dict(jobs_drained=True,closed_at=time.time()))
        return journal,r

    def test_idle_buffer_is_bounded_but_pins_and_reader_leases_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=RetentionPolicy(idle_buffer_seconds=5,pre_roll_seconds=4)
            j=FrameJournal(root,fps=1,policy=p);image=np.zeros((32,32,3),np.uint8)
            j.pin_interval('live',2,5)
            with j.lease(6,7):
                for i in range(21):
                    j.processed_through(i);j.offer(image,i);j.queue.join()
                times={e['timestamp'] for e in j.entries}
                self.assertTrue({2,3,4,5,6,7}.issubset(times))
                self.assertNotIn(10,times)
                self.assertTrue((root/'frames/2.000000.jpg').is_file())
            j.offer(image,22);j.queue.join();j.close()
            self.assertNotIn(6,{e['timestamp'] for e in j.entries})
            self.assertLess(len(j.entries),12)

    def test_monitor_watermark_prevents_pruning_delayed_pickup_pre_roll(self):
        with tempfile.TemporaryDirectory() as tmp:
            j=FrameJournal(tmp,fps=1,policy=RetentionPolicy(idle_buffer_seconds=5,pre_roll_seconds=4))
            j.processed_through(5)
            for i in range(20):j.offer(np.zeros((16,16,3),np.uint8),i);j.queue.join()
            j.pin_interval('late',2)
            j.processed_through(20);j.offer(np.zeros((16,16,3),np.uint8),20);j.queue.join();j.close()
            self.assertTrue((Path(tmp)/'frames/2.000000.jpg').exists())

    def test_archive_preserves_links_summaries_partial_video_and_deduplicates_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root,prediction=True)
            before=tree_bytes(root)
            policy=RetentionPolicy(require_server_ack=False)
            with patch('evidence_archive.encoder_available',return_value=None):
                result=ArchiveManager(root,policy).finalize(quiescent=True)
            self.assertEqual('cleaned',result['status']);self.assertFalse((root/'frames').exists())
            data=json.loads((root/'session.json').read_text());r=data['tests'][0]
            self.assertEqual(r['acid_clip']['path'],r['acid_candidate_clip']['path'])
            self.assertEqual(2,len(list((root/'archives').glob('*/videos/*'))))
            self.assertTrue(Path(r['rub_source_frame']).is_file())
            self.assertTrue(Path(r['record_path']).is_file())
            self.assertEqual(r,json.loads((root/'test_1/manifest.json').read_text()))
            for value in r['classifier_input_images'].values():self.assertTrue(Path(value).is_file())
            self.assertFalse((root/'test_1/baseline').exists())
            self.assertLess(tree_bytes(root),before)
            validate_video(r['acid_clip']['path'],r['acid_clip'])
            # Stale concurrent report input cannot restore raw references.
            write_report(root,[],[])
            self.assertEqual(data,json.loads((root/'session.json').read_text()))

    def test_required_server_ack_is_authenticated_and_bound_to_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            policy=RetentionPolicy()
            with patch('evidence_archive.encoder_available',return_value=None):
                result=ArchiveManager(root,policy).finalize(quiescent=True)
            self.assertEqual('awaiting_server_ack',result['status']);self.assertTrue((root/'frames').exists())
            commit=json.loads((root/'archive_commit.json').read_text())
            receipt=dict(session_id=root.name,archive_manifest_sha256=commit['manifest_sha256'],server_receipt_id='server-123',acknowledged_at='2026-10-31T12:00:00Z')
            receipt['signature']='bad';durable_json(root/'server_ack.json',receipt)
            with patch.dict(os.environ,JEWELTRACKER_ACK_HMAC_KEY='test-key'):
                self.assertFalse(valid_ack(root,commit,policy))
                receipt.pop('signature')
                receipt['signature']=hmac.new(b'test-key',json.dumps(receipt,sort_keys=True,separators=(',',':')).encode(),hashlib.sha256).hexdigest()
                durable_json(root/'server_ack.json',receipt)
                self.assertEqual('cleaned',ArchiveManager(root,policy).finalize(quiescent=True)['status'])

    def test_encoder_failure_preserves_raw_and_original_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);original=(root/'session.json').read_bytes()
            with patch('evidence_archive.encoder_available',return_value='ffmpeg'),patch('evidence_archive.archive_video',side_effect=OSError('encoder failed')):
                with self.assertRaises(OSError):ArchiveManager(root,RetentionPolicy(require_server_ack=False)).finalize(quiescent=True)
            self.assertTrue((root/'frames').exists());self.assertEqual(original,(root/'session.json').read_bytes())
            self.assertFalse((root/'archive_commit.json').exists())

    def test_missing_or_truncated_video_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,r=self.fixture(root)
            Path(r['acid_clip']['path']).write_bytes(b'broken')
            with self.assertRaises(OSError):ArchiveManager(root,RetentionPolicy(require_server_ack=False)).finalize(quiescent=True)
            self.assertTrue((root/'frames').exists());self.assertFalse((root/'archive_commit.json').exists())

    def test_missing_raw_clip_frame_and_resolution_changes_are_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            j=FrameJournal(tmp);a=np.zeros((16,16,3),np.uint8)
            j.offer(a,1);j.queue.join();j.offer(a,1.2);j.queue.join();j.close()
            (Path(tmp)/'frames/1.200000.jpg').unlink()
            with self.assertRaises(OSError):j.clip(1,1.3,Path(tmp)/'bad.avi')
            cv2.imwrite(str(Path(tmp)/'frames/1.200000.jpg'),np.zeros((24,24,3),np.uint8))
            with self.assertRaises(OSError):j.clip(1,1.3,Path(tmp)/'shape.avi')

    def test_crash_after_archive_rename_resumes_publication_before_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);manager=ArchiveManager(root,RetentionPolicy(require_server_ack=False))
            def interrupted_publish(final,data):
                shutil.copyfile(final/'session.json',root/'session.json')
                raise OSError('simulated crash after canonical session update')
            with patch('evidence_archive.encoder_available',return_value=None),patch.object(manager,'_publish',side_effect=interrupted_publish):
                with self.assertRaises(OSError):manager.finalize(quiescent=True)
            self.assertTrue((root/'frames').exists());self.assertFalse((root/'archive_commit.json').exists())
            self.assertEqual('cleaned',manager.finalize(quiescent=True)['status'])
            self.assertEqual(1,len(list((root/'archives').iterdir())))

    def test_crash_during_cleanup_can_retry_only_with_valid_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);manager=ArchiveManager(root,RetentionPolicy(require_server_ack=False))
            with patch('evidence_archive.encoder_available',return_value=None),patch('evidence_archive.shutil.rmtree',side_effect=OSError('simulated crash')):
                with self.assertRaises(OSError):manager.finalize(quiescent=True)
            self.assertTrue((root/'archive_commit.json').exists());self.assertTrue(list((root/'.cleanup').rglob('*.jpg')))
            self.assertEqual('cleaned',manager.finalize(quiescent=True)['status'])

    def test_checksum_mismatch_after_ack_never_cleans_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            with patch('evidence_archive.encoder_available',return_value=None):ArchiveManager(root).finalize(quiescent=True)
            target=next((root/'archives').glob('*/videos/*'));target.write_bytes(b'corrupt')
            with self.assertRaises(OSError):ArchiveManager(root,RetentionPolicy(require_server_ack=False)).finalize(quiescent=True)
            self.assertTrue((root/'frames').exists())

    def test_session_guard_and_concurrent_report_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            with SessionGuard(root):
                with self.assertRaises(BlockingIOError):SessionGuard(root)
            manager=ArchiveManager(root,RetentionPolicy(require_server_ack=False))
            data=json.loads((root/'session.json').read_text())
            with patch('evidence_archive.encoder_available',return_value=None),ThreadPoolExecutor(2) as pool:
                a=pool.submit(manager.finalize,True)
                b=pool.submit(write_report,root,data['jewels'],data['tests'])
                a.result();b.result()
            r=json.loads((root/'session.json').read_text())['tests'][0]
            self.assertIn('/archives/',r['acid_clip']['path']);self.assertTrue(Path(r['acid_clip']['path']).is_file())

    def test_no_quiescent_marker_or_empty_session_never_deletes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);(root/'session_closed.json').unlink()
            with self.assertRaises(RuntimeError):ArchiveManager(root).finalize(quiescent=True)
            self.assertTrue((root/'frames').exists())
            with self.assertRaises(RuntimeError):ArchiveManager(root).finalize()

    def test_running_session_blocks_archive_even_if_caller_claims_quiescence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            with SessionGuard(root):
                with self.assertRaises(BlockingIOError):ArchiveManager(root).finalize(quiescent=True)
            self.assertTrue((root/'frames').exists())

    def test_bad_duration_or_missing_frames_fail_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,r=self.fixture(Path(tmp));clip=dict(r['acid_clip'])
            clip['playback_seconds']+=3
            with self.assertRaises(OSError):validate_video(clip['path'],clip)
            clip=dict(r['acid_clip']);clip['frames']+=1
            with self.assertRaises(OSError):validate_video(clip['path'],clip)

    def test_pdf_failure_never_commits_or_removes_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            with patch('evidence_archive.encoder_available',return_value=None),patch('audit_report.render_report',side_effect=OSError('pdf failed')):
                with self.assertRaises(OSError):ArchiveManager(root,RetentionPolicy(require_server_ack=False)).finalize(quiescent=True)
            self.assertTrue((root/'frames').exists());self.assertFalse((root/'archive_commit.json').exists())

    def test_dry_run_does_not_change_files_and_expired_retention_requires_apply(self):
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'session';root.mkdir();self.fixture(root)
            policy=RetentionPolicy(require_server_ack=False,cleanup_raw_after_archive=False)
            with patch('evidence_archive.encoder_available',return_value=None):ArchiveManager(root,policy).finalize(quiescent=True)
            config=root/'policy.json';durable_json(config,dict(require_server_ack=False,archive_retention_days=1))
            commit=json.loads((root/'archive_commit.json').read_text());commit['committed_at']=time.time()-172800
            durable_json(root/'archive_commit.json',commit)
            before={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}
            with patch('sys.argv',['archive',str(root),'--config',str(config)]),redirect_stdout(io.StringIO()):main()
            after={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before,after);self.assertTrue((root/'frames').exists())
            with patch('sys.argv',['archive',str(root),'--config',str(config),'--apply']),redirect_stdout(io.StringIO()):main()
            self.assertFalse(root.exists())

    def test_h264_command_is_quality_controlled_and_thread_bounded(self):
        from evidence_archive import archive_video
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.avi';source.write_bytes(b'source')
            with patch('evidence_archive.subprocess.run',return_value=SimpleNamespace(returncode=0,stderr='')) as run:
                output,codec=archive_video(source,root/'video',RetentionPolicy(),'ffmpeg')
            command=run.call_args.args[0]
            self.assertEqual('.mp4',output.suffix);self.assertEqual('h264',codec)
            self.assertEqual('18',command[command.index('-crf')+1])
            self.assertEqual('1',command[command.index('-threads')+1])
            self.assertIn('libx264',command);self.assertIn('+faststart',command)

    def test_unfinished_test_directory_blocks_cleanup_and_low_disk_is_reported(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);(root/'test_unfinished').mkdir()
            with self.assertRaises(OSError):ArchiveManager(root).finalize(quiescent=True)
            self.assertTrue((root/'frames').exists())
        with tempfile.TemporaryDirectory() as tmp:
            j=FrameJournal(tmp,policy=RetentionPolicy())
            with patch('session_evidence.shutil.disk_usage',return_value=SimpleNamespace(free=1)):
                j.offer(np.zeros((16,16,3),np.uint8),1);j.queue.join()
            j.close();self.assertTrue(j.storage_status['low_disk'])

    def test_expiration_crash_resumes_from_durable_deletion_ledger(self):
        import io
        from contextlib import redirect_stdout
        from evidence_archive import resume_expirations
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);root=parent/'session';root.mkdir();self.fixture(root)
            policy=RetentionPolicy(require_server_ack=False)
            with patch('evidence_archive.encoder_available',return_value=None):ArchiveManager(root,policy).finalize(quiescent=True)
            commit=json.loads((root/'archive_commit.json').read_text());commit['committed_at']=time.time()-172800
            durable_json(root/'archive_commit.json',commit)
            config=parent/'policy.json';durable_json(config,dict(require_server_ack=False,archive_retention_days=1))
            with patch('sys.argv',['archive',str(parent),'--config',str(config),'--apply']),redirect_stdout(io.StringIO()),patch('evidence_archive.shutil.rmtree',side_effect=OSError('interrupted deletion')):
                main()
            self.assertFalse(root.exists());self.assertTrue(list(parent.glob('.expired-*')))
            resume_expirations(parent,RetentionPolicy(require_server_ack=False,archive_retention_days=1))
            self.assertFalse(list(parent.glob('.expired-*')))
            self.assertTrue(all(json.loads(p.read_text())['status']=='deleted' for p in (parent/'.retention-deletions').glob('*.json')))

    def test_classifier_input_archive_failure_does_not_change_material_prediction(self):
        from test_confirmed_acid_crop import ConfirmedAcidCropTests,PREDICTION
        from test_sop_revision import monitor_case
        with monitor_case() as (m,events):
            a,_=ConfirmedAcidCropTests().prepare(m)
            original=cv2.imwrite
            def write(path,*args,**kwargs):
                if 'classifier_input_' in str(path):return False
                return original(path,*args,**kwargs)
            with patch('continuous_testing.cv2.imwrite',side_effect=write):
                m._finish(106.1);m.drain_finalization()
            self.assertEqual(PREDICTION,m.tests[0]['prediction'],str({k:m.tests[0].get(k) for k in ('issues','frame_counts','reaction_coverage_status')}))
            self.assertTrue(m.tests[0]['classifier_input_archive_error'])
            m.classifier.predict.assert_called_once()

    def test_invalid_or_nonfinite_policy_cannot_disable_safety_checks(self):
        for kwargs in (dict(idle_buffer_seconds=float('nan')),dict(pre_roll_seconds=2),
                       dict(require_server_ack='false'),dict(archive_codec='unknown')):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):RetentionPolicy(**kwargs)

    def test_cleanup_command_resumes_closed_atomic_publication(self):
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            manager=ArchiveManager(root,RetentionPolicy(require_server_ack=False))
            with patch('evidence_archive.encoder_available',return_value=None),patch.object(manager,'_publish',side_effect=OSError('interrupted publication')):
                with self.assertRaises(OSError):manager.finalize(quiescent=True)
            config=root/'policy.json';durable_json(config,dict(require_server_ack=False))
            with patch('sys.argv',['archive',str(root),'--config',str(config),'--apply']),redirect_stdout(io.StringIO()):main()
            self.assertTrue((root/'archive_commit.json').is_file());self.assertFalse((root/'frames').exists())
            self.assertEqual(1,len(list((root/'archives').iterdir())))

if __name__=='__main__':unittest.main()
