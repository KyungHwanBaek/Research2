#!/usr/bin/env python3
"""Artifact invariants, finite geometry, original preservation, static links."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
import hashlib,json,subprocess,gc
import numpy as np
import open3d as o
R=Path(__file__).resolve().parents[1];O=R/'experiments/student_hall_refinement_20261026';SITE=R/'sites/reconstruction-lab-share/dist';A=SITE/'assets'
class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['href','src'] and v:self.refs.append(v)
def main():
    checks={}
    for suffix in ['vertices','normals','faces']:
        assert (A/f's10-mesh-{suffix}.bin').read_bytes()==(A/f's16-mesh-{suffix}.bin').read_bytes()
    v=np.fromfile(A/'s16-mesh-vertices.bin',dtype='<f4').reshape(-1,3);c=np.fromfile(A/'s16-mesh-colors.bin',dtype='u1').reshape(-1,3);f=np.fromfile(A/'s16-mesh-faces.bin',dtype='<u4').reshape(-1,3)
    assert np.isfinite(v).all() and len(c)==len(v) and f.max()<len(v)
    checks['display_geometry_identical']=True;checks['display_vertices']=len(v);checks['display_triangles']=len(f)
    meshes=[]
    for p in (O/'surfaces').glob('*.ply'):
        m=o.io.read_triangle_mesh(str(p));v=np.asarray(m.vertices);f=np.asarray(m.triangles)
        assert len(v)>0 and len(f)>0 and np.isfinite(v).all() and f.max()<len(v),p
        if m.has_vertex_colors():assert np.isfinite(np.asarray(m.vertex_colors)).all()
        meshes.append(dict(file=p.name,vertices=len(v),triangles=len(f)));del m,v,f;gc.collect()
    checks['finite_meshes']=meshes
    for x in json.loads((O/'baseline_manifest.json').read_text()):assert hashlib.sha256((R/x['path']).read_bytes()).hexdigest()==x['sha256']
    checks['baseline_preserved']=True
    for name in ['student-hall-next.html','student-hall.html']:
        parser=Links();parser.feed((SITE/name).read_text())
        for ref in parser.refs:
            u=urlparse(ref)
            if u.scheme or u.netloc or not u.path:continue
            path=SITE/unquote(u.path)
            assert path.is_file(),f'{name}: missing {ref}'
    checks['html_links_exist']=True
    for name in ['refinement.js','student-viewer.js']:subprocess.run(['node','--check',str(SITE/name)],check=True)
    checks['js_syntax']=True
    large=[str(p.relative_to(SITE)) for p in SITE.rglob('*') if p.is_file() and p.stat().st_size>24*1024*1024];assert not large,large
    checks['all_public_assets_under_24mib']=True
    video=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames','-show_entries','format=duration','-of','json',str(A/'refine-flythrough.mp4')],text=True));checks['video']=video
    assert video['streams'][0]['nb_frames']=='72'
    checks['browser_interactions']='Not run: browser runtime reported no available browser.'
    (O/'qa.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks),flush=True)
if __name__=='__main__':main()
