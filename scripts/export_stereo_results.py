#!/usr/bin/env python3
"""Generate offline WebGL viewers, overview plots and a reconstruction report."""
from pathlib import Path
import base64
import json
import shutil
import numpy as np


def source_reference_sheet(source_root,output,segment,start,end):
    from PIL import Image,ImageDraw,ImageFont
    frames=sorted(p for p in Path(source_root).iterdir() if (p/'rgb/left.png').exists())
    picks=np.linspace(start,end,6,dtype=int)
    sheet=Image.new('RGB',(1080,608),'#10151d')
    draw=ImageDraw.Draw(sheet)
    try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',19)
    except OSError:font=ImageFont.load_default()
    for j,i in enumerate(picks):
        x=(j%3)*360;y=(j//3)*304
        image=Image.open(frames[i]/'rgb/left.png').convert('RGB').resize((360,270))
        sheet.paste(image,(x,y));time=frames[i].name.replace('_',':',2).replace('_','.')
        draw.text((x+8,y+276),f'Frame {i} | {time}',fill='white',font=font)
    name=f'segment{segment}_source_frames.jpg';sheet.save(Path(output)/name,quality=94)
    first,last=frames[start].name,frames[end].name
    return name,first.replace('_',':',2).replace('_','.'),last.replace('_',':',2).replace('_','.')


VIEWER='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>*{box-sizing:border-box}body{margin:0;background:#10151d;color:#eaf0f8;font:15px system-ui}header{padding:18px 24px;display:flex;gap:18px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #334154}h1{font-size:20px;margin:0}button,a{background:#243347;color:#eaf0f8;padding:9px 13px;border:1px solid #41516a;border-radius:7px;text-decoration:none;cursor:pointer}label{display:flex;align-items:center;gap:8px}canvas{display:block;width:100vw;height:calc(100vh - 142px)}footer{padding:12px 24px;color:#b7c7db}input{accent-color:#7dd3fc}#error{color:#fca5a5}</style>
<header><h1>__TITLE__</h1><button id="perspective">사선</button><button id="top">위</button><button id="front">정면</button><button id="reset">초기화</button><label>점 크기 <input id="size" type="range" min="1" max="5" step=".1" value="1.5"></label><a href="index.html">결과 목록</a></header>
<canvas id="scene"></canvas><footer>드래그: 회전 · Shift+드래그: 이동 · 스크롤: 확대/축소 · __COUNTS__ <span id="error"></span></footer>
<script>
const canvas=document.querySelector('canvas'),gl=canvas.getContext('webgl2');
if(!gl)document.getElementById('error').textContent='WebGL2를 지원하는 브라우저가 필요합니다.';
const data=__DATA__;
function unpack(s,T){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return new T(a.buffer)}
const vertex=`#version 300 es
in vec3 aPos;in vec3 aColor;in vec3 aNormal;uniform float yaw;uniform float pitch;uniform float zoom;uniform float aspect;uniform float size;uniform vec2 pan;out vec3 color;out float shade;
vec3 rotate(vec3 p){float c=cos(yaw),s=sin(yaw),cp=cos(pitch),sp=sin(pitch);vec3 q=vec3(c*p.x+s*p.z,-p.y,-s*p.x+c*p.z);return vec3(q.x,cp*q.y-sp*q.z,sp*q.y+cp*q.z);}
void main(){vec3 p=rotate(aPos);gl_Position=vec4(p.x*zoom/aspect+pan.x,p.y*zoom+pan.y,-p.z*.18,1.);gl_PointSize=size;color=aColor;vec3 n=normalize(rotate(aNormal));shade=.5+.5*abs(dot(n,normalize(vec3(.3,.7,1.))));}`;
const fragment=`#version 300 es
precision highp float;in vec3 color;in float shade;uniform bool mesh;out vec4 frag;void main(){if(!mesh&&length(gl_PointCoord-.5)>.5)discard;frag=vec4(color*(mesh?shade:1.),1.);}`;
function shader(type,src){let s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
const prg=gl.createProgram();gl.attachShader(prg,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(prg,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(prg);if(!gl.getProgramParameter(prg,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(prg));gl.useProgram(prg);
function attr(name,array,size,type=gl.FLOAT,normalized=false){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,array,gl.STATIC_DRAW);const p=gl.getAttribLocation(prg,name);gl.enableVertexAttribArray(p);gl.vertexAttribPointer(p,size,type,normalized,0,0)}
const pos=unpack(data.positions,Float32Array),colors=unpack(data.colors,Uint8Array);attr('aPos',pos,3);attr('aColor',colors,3,gl.UNSIGNED_BYTE,true);
attr('aNormal',data.normals?unpack(data.normals,Float32Array):new Float32Array(pos.length).fill(.577),3);
let faces=null;if(data.faces){faces=unpack(data.faces,Uint32Array);let b=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,b);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,faces,gl.STATIC_DRAW)}
const u={};for(const k of ['yaw','pitch','zoom','aspect','size','pan','mesh'])u[k]=gl.getUniformLocation(prg,k);
let yaw=-.55,pitch=-.35,zoom=1.15,pan=[0,0],last=null;
function draw(){const d=Math.min(devicePixelRatio,2);canvas.width=canvas.clientWidth*d;canvas.height=canvas.clientHeight*d;gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(.063,.082,.114,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.uniform1f(u.yaw,yaw);gl.uniform1f(u.pitch,pitch);gl.uniform1f(u.zoom,zoom);gl.uniform1f(u.aspect,canvas.width/canvas.height);gl.uniform1f(u.size,Number(document.getElementById('size').value)*d);gl.uniform2fv(u.pan,pan);gl.uniform1i(u.mesh,!!faces);if(faces)gl.drawElements(gl.TRIANGLES,faces.length,gl.UNSIGNED_INT,0);else gl.drawArrays(gl.POINTS,0,pos.length/3)}
canvas.onpointerdown=e=>{last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>last=null;canvas.onpointermove=e=>{if(!last)return;let dx=e.clientX-last[0],dy=e.clientY-last[1];if(e.shiftKey){pan[0]+=dx/canvas.clientWidth*2;pan[1]-=dy/canvas.clientHeight*2}else{yaw+=dx*.006;pitch+=dy*.006}last=[e.clientX,e.clientY];draw()};canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.min(30,Math.max(.2,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
document.getElementById('top').onclick=()=>{yaw=0;pitch=Math.PI/2;draw()};document.getElementById('front').onclick=()=>{yaw=0;pitch=0;draw()};document.getElementById('perspective').onclick=()=>{yaw=-.55;pitch=-.35;draw()};document.getElementById('reset').onclick=()=>{yaw=-.55;pitch=-.35;zoom=1.15;pan=[0,0];draw()};document.getElementById('size').oninput=draw;window.onresize=draw;draw();window.viewerReady={vertices:pos.length/3,triangles:faces?faces.length/3:0,error:gl.getError()};
</script></html>'''


def encoded(a):return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def viewer(path,xyz,rgb,title,normals=None,faces=None):
    center=(np.percentile(xyz,1,axis=0)+np.percentile(xyz,99,axis=0))/2
    radius=np.max(np.percentile(np.abs(xyz-center),98,axis=0))
    xyz=((xyz-center)/max(radius,1e-6)).astype('<f4')
    data={'positions':encoded(xyz),'colors':encoded(rgb.astype('u1'))}
    if normals is not None:data['normals']=encoded(normals.astype('<f4'))
    if faces is not None:data['faces']=encoded(faces.astype('<u4'))
    counts=f'{len(xyz):,} vertices'+(f' · {len(faces):,} triangles' if faces is not None else '')
    Path(path).write_text(VIEWER.replace('__TITLE__',title).replace('__COUNTS__',counts).replace('__DATA__',json.dumps(data)))


def main():
    import open3d as o3d
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(Path('logs/mplconfig').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path('results/stereo_improved_0905');output.mkdir(exist_ok=True)
    source=Path('experiments/stereo_improved_0905/fused')
    stats=json.loads((source/'summary.json').read_text());links=[]
    for r in stats:
        seg=r['segment'];stem=source/f'stereo_rgb_nir_segment{seg}'
        z=np.load(stem.with_suffix('.npz'));xyz,rgb=z['xyz'],z['rgb']
        rng=np.random.default_rng(42);ids=rng.choice(len(xyz),min(500000,len(xyz)),replace=False)
        viewer(output/f'segment{seg}_points.html',xyz[ids],rgb[ids],f'RGB·NIR 스테레오 복원 — 구간 {seg+1}')
        mesh=o3d.io.read_triangle_mesh(str(stem)+'_tsdf_preview.ply')
        viewer(output/f'segment{seg}_mesh.html',np.asarray(mesh.vertices),np.uint8(np.asarray(mesh.vertex_colors)*255),
                f'스테레오 표면 메시 — 구간 {seg+1}',np.asarray(mesh.vertex_normals),np.asarray(mesh.triangles))
        shutil.copy2(stem.with_suffix('.ply'),output/f'segment{seg}_points.ply')
        shutil.copy2(str(stem)+'_tsdf_mesh.ply',output/f'segment{seg}_mesh.ply')
        fig,axes=plt.subplots(1,2,figsize=(15,6),facecolor='#10151d')
        sample=ids[:min(110000,len(ids))]
        for ax,dims,title in zip(axes,[(0,2),(0,1)],['Top / XZ','Side / XY']):
            ax.set_facecolor('#10151d');ax.scatter(xyz[sample,dims[0]],xyz[sample,dims[1]],c=rgb[sample]/255,s=.12,rasterized=True)
            ax.set_aspect('equal');ax.set_title(title,color='white');ax.tick_params(colors='white');ax.grid(alpha=.1)
        fig.suptitle(f'LiDAR-free stereo reconstruction / segment {seg+1} / {len(xyz):,} points',color='white')
        fig.tight_layout();fig.savefig(output/f'segment{seg}_overview.jpg',dpi=150);plt.close(fig)
        reference,first,last=source_reference_sheet('data/extracted/sampled_540p/09-05-17-34-36',output,seg,*r['source_frame_range'])
        description={0:'도로와 기둥이 있는 복도를 포함하는 긴 이동 구간',1:'건물 입구에서 중정 쪽을 바라보며 회전하는 짧은 구간',2:'중정 주변을 관찰하며 조형물이 있는 쪽까지 이동하는 구간'}[seg]
        links.append(f'<article><h2>구간 {seg+1} — {description}</h2><p>촬영 시각 {first}–{last} · 원본 프레임 {r["source_frame_range"]} · 점 {r["final_points"]:,}개 · 삼각형 {r.get("mesh_triangles",0):,}개</p><p>아래 원본 사진 6장은 이 구간을 시간순으로 보여줍니다. 구간 번호는 추적이 이어진 시간 범위를 뜻합니다. 건물이나 장소 이름으로 나눈 번호가 아닙니다.</p><img src="{reference}" alt="구간 {seg+1} 원본 사진 시간순 비교"><p>이 사진들에서 관찰한 공간의 복원 결과:</p><img src="segment{seg}_overview.jpg"><p><a href="segment{seg}_points.html">3D 점군 회전해서 보기</a> <a href="segment{seg}_mesh.html">표면 메시 보기</a> <a href="segment{seg}_points.ply">점군 PLY</a> <a href="segment{seg}_mesh.ply">메시 PLY</a></p></article>')
    intro='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LiDAR 없이 개선한 3D 복원</title><style>body{max-width:1200px;margin:40px auto;padding:0 22px;font:16px/1.7 system-ui;background:#10151d;color:#eaf0f8}h1{font-size:34px}h2{font-size:24px}a{color:#7dd3fc;margin-right:18px}article{border-top:1px solid #334154;margin:32px 0;padding-top:16px}img{width:100%;border-radius:10px}p{color:#bfcede}table{border-collapse:collapse}td,th{padding:10px 18px;border:1px solid #334154}</style><h1>LiDAR 없이 개선한 3D 복원</h1><p>09-05-17-34-36 · 기존 좌우 RGB/NIR 이미지와 카메라 보정값 사용</p><p>카메라 사이의 회전·이동 방향 재보정 → 좌우 일관성 검사 → RGB/NIR 깊이 교차 검사 → 스테레오 카메라 추적 → 검증된 COLMAP 제약으로 경로 보정 → 다른 시점과 일치하는 깊이 융합 → TSDF 표면 복원.</p><p>촬영 공백이 있는 두 회전 구간의 연결은 확인되지 않아 세 구간을 각각 제공합니다. 거리 단위는 보정 파일의 이동값이 mm라는 가정에 따르며, 절대 치수와 전체 경로 오차는 독립 측정으로 검증되지 않았습니다.</p>'''
    intro=intro.replace('깊이 융합 → TSDF','깊이 융합 → TSDF')
    intro+='<p>긴 경로의 높이 누적 오차를 줄이기 위해, 이미지에서 측정한 바닥 평면을 사용해 각 구간의 바닥이 대체로 평탄하다는 제약을 추가했습니다. 경사로의 실제 높이 변화까지 검증한 결과는 아닙니다. 제약 전 결과는 experiments/stereo_improved_0905/fused_unconstrained에 보존했습니다.</p>'
    selected=Path('results/renders/3dgs_0905_sparse_30000')
    base=Path('results/renders/3dgs_0905_sparse_12000')
    alternate=Path('results/renders/3dgs_0905_stereo_12000')
    evaluations=[]
    for label,root in [('기존 점 초기화 · 12,000회',base),('스테레오 점 초기화 · 12,000회',alternate),('선택 모델 · 30,000회',selected)]:
        j=json.loads((root/'metrics.json').read_text());m=j['heldout_metrics']
        evaluations.append(f'<tr><td>{label}</td><td>{m["psnr"]:.2f}</td><td>{m["ssim"]:.3f}</td></tr>')
    shutil.copy2(selected/'flythrough.gif',output/'3dgs_flythrough.gif')
    shutil.copy2(selected/'heldout_contact_sheet.jpg',output/'3dgs_heldout_views.jpg')
    rendering='<article><h2>3DGS 렌더링</h2><p>원본 프레임 140–440에서 61개 시점 사용 · 학습 53개 / 제외 시점 검증 8개 · 720×540 · LiDAR 미사용. 긴 경로 점군과는 별도의 지역 렌더링 모델입니다.</p><table><tr><th>실험</th><th>PSNR ↑</th><th>SSIM ↑</th></tr>'+''.join(evaluations)+'</table><p>스테레오 초기화는 이번 검증에서 화질이 낮아 선택하지 않았습니다. 기존 점 초기화 모델을 추가 학습한 결과를 제공합니다.</p><img src="3dgs_flythrough.gif" alt="3DGS novel view flythrough"><p><a href="3dgs_flythrough.gif">이동 시점 GIF</a><a href="heldout_comparison.jpg">입력/12,000회/30,000회 비교</a><a href="../renders/3dgs_0905_sparse_30000/scene.ply">3DGS Gaussian PLY</a></p></article>'
    (output/'index.html').write_text(intro+rendering+'\n'.join(links)+'</html>')
    (output/'geometry_metrics.json').write_text(json.dumps(stats,indent=2))
    shutil.copy2('experiments/stereo_improved_0905/optimized_poses.npz',output/'camera_poses.npz')
    print('results',output/'index.html',flush=True)


if __name__=='__main__':main()
