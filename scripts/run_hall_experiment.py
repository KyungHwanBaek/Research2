#!/usr/bin/env python3
"""Immutable named run records for the 2026-10-27 follow-up experiments."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys,time
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/student_hall_refinement_20261027'
def now():return datetime.datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def event(row):
    OUT.mkdir(exist_ok=True,parents=True)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(dict(time=now(),**row),ensure_ascii=False)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    assert cmd
    for name in ['runs','logs']:(OUT/name).mkdir(parents=True,exist_ok=True)
    path=OUT/'runs'/f'{a.name}.json';assert not path.exists(),f'Choose a new run name: {path}'
    row=dict(name=a.name,command=cmd,cwd=str(ROOT),start=now(),status='running',threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']})
    path.write_text(json.dumps(row,indent=2));event(dict(event='started',**row));start=time.time()
    with (OUT/'logs'/f'{a.name}.log').open('w',buffering=1) as f:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);row['pid']=child.pid;path.write_text(json.dumps(row,indent=2));print(json.dumps(dict(run=a.name,pid=child.pid)),flush=True);result=child.wait()
    row.update(end=now(),seconds=time.time()-start,exit_code=result,status='succeeded' if result==0 else 'failed');path.write_text(json.dumps(row,indent=2));event(dict(event='completed',**row));print(json.dumps(row),flush=True);sys.exit(result)
if __name__=='__main__':main()
