#!/usr/bin/env python3
"""Validate conservative blends of the prior trajectory and joint stereo BA."""
import json
from pathlib import Path
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from improve_stereo_reconstruction import Stereo
from joint_stereo_bundle_adjustment import dense_score


def main():
    cv2.setNumThreads(4);root=Path('experiments/stereo_joint_ba_0905')
    old=np.load('experiments/stereo_improved_0905/optimized_poses.npz');result=old['poses'].copy()
    reports=json.loads((root/'comparison.json').read_text())
    s=Stereo('data/extracted/sampled_540p/09-05-17-34-36','experiments/stereo_improved_0905/depth')
    for r in reports:
        ids=np.flatnonzero(old['components']==r['segment']);base=old['poses'][ids]
        new=np.load(root/f'segment{r["segment"]}/poses.npz')['poses']
        delta=Rotation.from_matrix(np.einsum('nji,njk->nik',base[:,:3,:3],new[:,:3,:3])).as_rotvec()
        candidates=[{'alpha':0.,**r['dense_before']}];best=base;score=r['dense_before']['mean_cross_view_agreement_fraction'];alpha_best=0.
        for alpha in [.25,.5,.75,1.]:
            blended=base.copy();blended[:,:3,3]=base[:,:3,3]*(1-alpha)+new[:,:3,3]*alpha
            blended[:,:3,:3]=np.einsum('nij,njk->nik',base[:,:3,:3],Rotation.from_rotvec(delta*alpha).as_matrix())
            val=dense_score(s,old['indices'][ids],blended);candidates.append({'alpha':alpha,**val})
            if val['median_cross_view_relative_depth_error']<=r['dense_before']['median_cross_view_relative_depth_error'] and val['mean_cross_view_agreement_fraction']>score:
                best=blended;score=val['mean_cross_view_agreement_fraction'];alpha_best=alpha
            print('blend',r['segment'],alpha,val,flush=True)
        result[ids]=best;r['selection_candidates']=candidates;r['selected_alpha']=alpha_best;r['selected']=alpha_best>0
        r['selected_dense_score']=next(c for c in candidates if c['alpha']==alpha_best)
    np.savez(root/'selected_poses.npz',indices=old['indices'],poses=result,components=old['components'])
    (root/'comparison.json').write_text(json.dumps(reports,indent=2))


if __name__=='__main__':main()
