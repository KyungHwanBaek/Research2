#!/usr/bin/env python3
"""Hash raw inputs, preserve source, freeze dependencies, verify baseline."""
from pathlib import Path
import json,hashlib,subprocess,platform,shutil,datetime

R=Path(__file__).resolve().parents[1];O=R/'experiments/student_hall_refinement_20261026'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()
def main():
    inputs=R/'data/extracted/01-26-15-41-24';files=sorted(inputs.glob('*/*/*distorted.png'))+[inputs/n for n in ['calibration.npz','cam2lidar.npy','storage.hdf5']]
    rows=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    (O/'raw_input_manifest.json').write_text(json.dumps(rows,indent=2))
    before=json.loads((O/'baseline_manifest.json').read_text());checks=[dict(path=v['path'],unchanged=sha(R/v['path'])==v['sha256']) for v in before]
    assert all(v['unchanged'] for v in checks),'Baseline changed unexpectedly'
    (O/'baseline_preservation_check.json').write_text(json.dumps(checks,indent=2))
    src=O/'source_snapshot';src.mkdir(exist_ok=True)
    scripts=list((R/'scripts').glob('refine_student_hall_*.py'))+[R/'scripts'/n for n in ['refinement_run.py','record_refinement_provenance.py','prepare_refinement_site.py','render_refinement_evidence.py','verify_student_hall_refinement.py','reconstruct_student_hall_lidar.py','mesh_student_hall_lidar.py','poisson_student_hall.py','render_student_hall_geometry.py','improve_stereo_reconstruction.py','train_msplat_render_flythrough.py']]
    sources=[]
    for p in scripts:
        if p.exists():shutil.copyfile(p,src/p.name);sources.append(dict(path=str(p.relative_to(R)),sha256=sha(p)))
    (O/'source_manifest.json').write_text(json.dumps(sources,indent=2))
    environment=dict(platform=platform.platform(),machine=platform.machine(),captured=datetime.datetime.now().astimezone().isoformat(),gpu='Apple M4 / Metal MPS',ram_gb=24)
    for env in ['.venv','.venv-mesh312','.venv-gs-refine']:
        result=subprocess.run([str(R/env/'bin/python'),'-m','pip','freeze'],capture_output=True,text=True,check=True)
        (O/f'dependencies_{env.strip(".")}.txt').write_text(result.stdout)
        environment[env]=subprocess.check_output([str(R/env/'bin/python'),'--version'],text=True).strip()
    (O/'environment.json').write_text(json.dumps(environment,indent=2))
    print(json.dumps(dict(raw_inputs=len(rows),raw_input_bytes=sum(v['bytes'] for v in rows),baseline_checks=len(checks),baseline_preserved=True,snapshotted_scripts=len(sources))))
if __name__=='__main__':main()
