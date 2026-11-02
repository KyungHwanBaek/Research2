#!/usr/bin/env python3
"""Build an isolated msplat experiment; preserve installed native library."""
from pathlib import Path
import difflib,json,shutil

O=Path('experiments/student_hall_research_20261027')
SRC=Path('third_party/msplat'); DST=O/'native_source'

def main():
    assert not DST.exists()
    shutil.copytree(SRC,DST,ignore=shutil.ignore_patterns('.git','datasets','build','__pycache__','*.so','*.pyc'))
    changes={}
    for name in ['core/metal/msplat_metal.mm','core/metal/msplat_metal.metal']:
        p=DST/name; old=p.read_text();new=old.replace('#define MAX_TILE_ELEMS 2048','#define MAX_TILE_ELEMS 4096').replace('>2048 gaussians','>4096 gaussians').replace('max 2048/tile','max 4096/tile')
        if name.endswith('.mm'):
            new=new.replace('static FusedTensorCache g_tcache;','''static FusedTensorCache g_tcache;

int msplat_tile_overflow_status() {
    msplat_gpu_sync();
    return g_tcache.overflow_flag.defined() ? *g_tcache.overflow_flag.data<int32_t>() : 0;
}
''')
        p.write_text(new);changes[name]=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original/'+name,tofile='experiment/'+name))
    p=DST/'python/bindings.cpp';old=p.read_text();new=old.replace('namespace nb = nanobind;','int msplat_tile_overflow_status();\nnamespace nb = nanobind;')
    new=new.replace('m.doc() = "msplat: Metal-accelerated 3D Gaussian Splatting";','''m.doc() = "Isolated student hall msplat experiment: 4096 tile slots";
    m.def("tile_overflow_status", &msplat_tile_overflow_status);
    m.def("tile_capacity", [](){return 4096;});''')
    marker='        .def("load_checkpoint", &GaussianTrainer::load_checkpoint'
    assert marker in new
    new=new.replace(marker,'''        .def("scale_learning_rates", [](GaussianTrainer &t, float factor) {
            if (!(factor > 0.f && factor <= 1.f)) throw std::runtime_error("Invalid LR factor");
            for (int g=0; g<Model::N_ADAM_GROUPS; ++g) t.model->adam_lr[g] *= factor;
            t.model->means_lr_init *= factor; t.model->means_lr_final *= factor;
        }, "factor"_a)
'''+marker)
    p.write_text(new);changes['python/bindings.cpp']=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original/python/bindings.cpp',tofile='experiment/python/bindings.cpp'))
    (O/'native_changes.patch').write_text('\n'.join(changes.values()))
    (O/'native_manifest.json').write_text(json.dumps(dict(original=str(SRC),copy=str(DST),tile_capacity=4096,original_tile_capacity=2048,changes=list(changes),scope='Capacity only, diagnostics, LR-scaling API. No Mip filter/AbsGS/depth-loss implementation claimed.'),indent=2))
    print('NATIVE_COPY',DST,flush=True)
if __name__=='__main__':main()
