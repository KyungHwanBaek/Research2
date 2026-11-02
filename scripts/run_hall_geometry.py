#!/usr/bin/env python3
"""Record commands for the 2026-10-31 geometry experiments."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse, json, os, subprocess, time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/student_hall_geometry_20261031'
def now(): return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def execute(name, command, cwd=None):
    for folder in ['runs','logs']: (OUT/folder).mkdir(parents=True,exist_ok=True)
    path=OUT/'runs'/f'{name}.json'
    if path.exists():
        original=name;attempt=2
        while path.exists():
            name=f'{original}_attempt{attempt}';attempt+=1;path=OUT/'runs'/f'{name}.json'
    row=dict(name=name,command=list(map(str,command)),cwd=str(cwd or ROOT),start=now(),status='running')
    path.write_text(json.dumps(row,indent=2)); start=time.time()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with (OUT/'logs'/f'{name}.log').open('w',buffering=1) as log:
        p=subprocess.Popen(row['command'],cwd=row['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
        row['pid']=p.pid;path.write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
        result=p.wait()
    row.update(end=now(),elapsed_s=time.time()-start,exit_code=result,status='succeeded' if result==0 else 'failed')
    path.write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
    if result: raise RuntimeError(f'{name} failed, see {OUT}/logs/{name}.log')
    return row
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    execute(a.name,a.command[1:] if a.command[:1]==['--'] else a.command)
