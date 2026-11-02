#!/usr/bin/env python3
"""Validate selected artifacts, geometry orientation, archive and page links."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
import json,hashlib,zipfile,subprocess,gc
import numpy as np
import open3d as o
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'experiments/student_hall_refinement_20261027';B=ROOT/'experiments/student_hall_lidar';SITE=ROOT/'sites/reconstruction-lab-share/dist';A=SITE/'assets'
class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['href','src'] and v:self.refs.append(v)
def orientation(before,after):
    f=np.asarray(before.triangles);assert np.array_equal(f,np.asarray(after.triangles));v=np.asarray(before.vertices);w=np.asarray(after.vertices);count=0
    for start in range(0,len(f),100000):
        tri=f[start:start+100000];x=v[tri];y=w[tri];n=np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]);m=np.cross(y[:,1]-y[:,0],y[:,2]-y[:,0]);count+=int(np.sum(np.sum(n*m,axis=1)<0))
    return count
def main():
    r=json.loads((O/'results.json').read_text());q=dict(selected=r['selected']['name'],psnr=r['selected']['psnr'],ssim=r['selected']['ssim']);before=o.io.read_triangle_mesh(str(B/'fused/student_hall_poisson.ply'));after=o.io.read_triangle_mesh(str(O/'surfaces/mls_r0.35_l0.08_safe_final.ply'));flips=orientation(before,after);assert flips==0;q['safe_final_reversed_faces']=flips;del before,after;gc.collect()
    before=o.io.read_triangle_mesh(str(B/'fused/validation_poisson_11.ply'));after=o.io.read_triangle_mesh(str(O/'surfaces/mls_r0.35_l0.08_safe_validation.ply'));assert orientation(before,after)==0;q['safe_validation_reversed_faces']=0;del before,after;gc.collect()
    meshes=[]
    for p in sorted((O/'surfaces').glob('*.ply')):
        mesh=o.io.read_triangle_mesh(str(p));v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);assert len(v)>0 and len(f)>0 and np.isfinite(v).all() and f.max()<len(v)
        if mesh.has_vertex_colors():assert np.isfinite(np.asarray(mesh.vertex_colors)).all()
        meshes.append(dict(file=p.name,vertices=len(v),triangles=len(f)));del mesh,v,f;gc.collect()
    q['finite_meshes']=meshes
    v=np.fromfile(A/'s18-mesh-vertices.bin',dtype='<f4').reshape(-1,3);n=np.fromfile(A/'s18-mesh-normals.bin',dtype='<f4').reshape(-1,3);c=np.fromfile(A/'s18-mesh-colors.bin',dtype='u1').reshape(-1,3);f=np.fromfile(A/'s18-mesh-faces.bin',dtype='<u4').reshape(-1,3);assert len(v)==len(n)==len(c) and np.isfinite(v).all() and np.isfinite(n).all() and f.max()<len(v)
    q['display']=dict(vertices=len(v),triangles=len(f))
    for name in ['hall27.js','student-viewer.js']:subprocess.run(['node','--check',str(SITE/name)],check=True)
    q['js_syntax']=True
    parser=Links();parser.feed((SITE/'student-hall-20261027.html').read_text())
    for link in parser.refs:
        u=urlparse(link)
        if u.scheme or u.netloc or not u.path:continue
        assert (SITE/unquote(u.path)).is_file(),link
    q['page_links_exist']=True
    for p in SITE.rglob('*'):
        if p.is_file():assert p.stat().st_size<24*1024*1024,p
    q['public_asset_limit']=True
    archive=SITE/'downloads/student_hall_followup_20261027_records.zip'
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None;q['archive_entries']=len(z.namelist())
    video=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames','-show_entries','format=duration','-of','json',str(A/'hall27-flythrough.mp4')],text=True));assert video['streams'][0]['nb_frames']=='72';q['video']=video
    q['preservation']=json.loads((O/'preservation_checks.json').read_text());q['browser_interactions']='Not available: browser runtime selected no available browser; discovery list was empty.'
    q['preflight_corrections']=['Quoted the numeric-leading JavaScript label key after initial node syntax check found it. Final node checks pass.','Initial unconstrained MLS created microscopic reversed triangles; the separately saved safe candidate reverts implicated vertices and passes independent orientation checks.']
    (O/'qa.json').write_text(json.dumps(q,indent=2));print(json.dumps(q),flush=True)
if __name__=='__main__':main()
