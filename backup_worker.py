"""Persistent low-priority backup process; JSON requests/responses on stdio."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys


def limits(config):
    try:
        os.nice(max(0,min(19,int(config['nice']))))
    except OSError:
        pass
    try:
        cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0,set(cpus[-max(1,int(config['max_cpus'])):]))
    except (AttributeError,OSError):
        pass
    import cv2
    cv2.setNumThreads(config['threads'])


def serve():
    engine = None
    for line in sys.stdin:
        try:
            message = json.loads(line)
            config = message['config']
            with redirect_stdout(sys.stderr):
                if engine is None:
                    limits(config)
                    from backup_analysis import AnalysisEngine
                    engine = AnalysisEngine(config)
                job = json.loads(Path(message['request']).read_text())
                response = engine.analyze(job)
                response['worker']=dict(pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),cpus=sorted(os.sched_getaffinity(0)))
            print(json.dumps(response),flush=True)
        except Exception as exc:
            print(json.dumps(dict(error=str(exc))),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,help='Replay a backup_request.json without changing the original session')
    parser.add_argument('--output',type=Path,help='New evidence directory for a replay')
    args = parser.parse_args()
    if args.request:
        if args.output is None:
            parser.error('--output is required for replay')
        if args.output.exists() and any(args.output.iterdir()):
            parser.error('--output must be an empty/new directory')
        from backup_recovery import configuration, BackupWorker
        settings_path = Path(__file__).with_name('sop_config.json')
        config = configuration(json.loads(settings_path.read_text()).get('edgesam_backup'))
        job = json.loads(args.request.read_text())
        args.output.mkdir(parents=True,exist_ok=True)
        job['directory'] = str(args.output.resolve())
        worker = BackupWorker(config)
        try:
            result = worker.analyze(job,args.output)
            (args.output/'backup_result.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result,indent=2))
        finally:
            worker.close()
    else:
        serve()


if __name__=='__main__':
    main()
