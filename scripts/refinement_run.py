#!/usr/bin/env python3
"""Record every experimental invocation, stdout/stderr, timestamps and exit code."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/student_hall_refinement_20261026'

def now():return datetime.datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def event(row):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps({'time':now(),**row},ensure_ascii=False)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:raise SystemExit('missing command')
    folder=OUT/'runs';folder.mkdir(parents=True,exist_ok=True);logdir=OUT/'logs';logdir.mkdir(exist_ok=True)
    path=folder/f'{a.name}.json'
    if path.exists():raise SystemExit(f'Run already recorded: {path}; use a new run name')
    record={'name':a.name,'command':command,'cwd':str(ROOT),'start':now(),
            'thread_settings':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']},'status':'running'}
    path.write_text(json.dumps(record,ensure_ascii=False,indent=2));event({'event':'run_started',**record})
    start=time.time()
    with (logdir/f'{a.name}.log').open('w',buffering=1) as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        record['pid']=proc.pid;path.write_text(json.dumps(record,ensure_ascii=False,indent=2))
        print(json.dumps({'run':a.name,'pid':proc.pid,'log':str(logdir/f'{a.name}.log')}),flush=True)
        result=proc.wait()
    record.update(end=now(),elapsed_s=time.time()-start,exit_code=result,status='succeeded' if result==0 else 'failed')
    path.write_text(json.dumps(record,ensure_ascii=False,indent=2));event({'event':'run_completed',**record})
    print(json.dumps(record,ensure_ascii=False),flush=True);sys.exit(result)

if __name__=='__main__':main()
