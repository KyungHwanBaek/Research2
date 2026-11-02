#!/usr/bin/env python3
"""Optional photometric surface refinement and textured GLB export."""
import argparse
from run_hall_geometry import ROOT,OUT,execute
def main(stage,mode='validation',refined=False):
    if stage=='refine':
        p=OUT/f'mvs_stereo_{mode}';execute(f'mvs_stereo_photometric_refine_{mode}',[ROOT/'third_party/openmvs/RefineMesh','-i',p/'scene_mesh.mvs','-o',p/'scene_refined.mvs','-w',p,'--resolution-level','1','--min-resolution','640','--max-views','6','--decimate','1','--close-holes','0','--ensure-edge-size','0','--max-face-area','64','--scales','2','--gradient-step','20.04','--max-threads','6','--archive-type','2'])
    else:
        p=OUT/'mvs_stereo_final';execute('texture_photo_refined_final' if refined else 'texture_photo_final',[ROOT/'third_party/openmvs/TextureMesh','-i',p/('scene_refined.mvs' if refined else 'scene_mesh.mvs'),'-o',p/('photo_refined_textured.mvs' if refined else 'photo_textured.mvs'),'-w',p,'--export-type','glb','--resolution-level','1','--min-resolution','640','--close-holes','0','--max-texture-size','4096','--max-threads','4','--archive-type','2'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['refine','texture']);p.add_argument('--mode',choices=['validation','final'],default='validation');p.add_argument('--refined',action='store_true');a=p.parse_args();main(a.stage,a.mode,a.refined)
