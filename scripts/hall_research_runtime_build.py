#!/usr/bin/env python3
"""Use Apple's supported runtime Metal compiler when xcrun metal is absent."""
from pathlib import Path
import difflib,json
O=Path('experiments/student_hall_research_20261027');D=O/'native_source'
def main():
    p=D/'CMakeLists.txt';old=p.read_text();start=old.index('add_custom_command(');end=old.index('# --- Core static library ---',start)
    new=old[:start]+'''# Runtime MSL compilation: no offline xcrun metal executable required.
add_custom_target(metallib ALL)

'''+old[end:]
    new=new.replace('install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib DESTINATION msplat)','install(FILES ${METAL_SOURCE} DESTINATION msplat)')
    p.write_text(new)
    changes=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='experiment/CMakeLists.txt',tofile='runtime/CMakeLists.txt'))
    p=D/'core/metal/msplat_metal.mm';old=p.read_text();start=old.index('    if (g_metallib_path) {',old.index('MetalContext* init_msplat_metal_context'));end=old.index('    if (!metal_library) {\n        fprintf',start)
    new=old[:start]+'''    // Isolated experiment: compile the bundled MSL with Metal's runtime API.
    Dl_info dl_info;
    NSString *sourcePath = nil;
    if (dladdr((void*)init_msplat_metal_context, &dl_info) && dl_info.dli_fname) {
        NSString *dir = [[NSString stringWithUTF8String:dl_info.dli_fname] stringByDeletingLastPathComponent];
        sourcePath = [dir stringByAppendingPathComponent:@"msplat_metal.metal"];
    }
    NSString *source = sourcePath ? [NSString stringWithContentsOfFile:sourcePath encoding:NSUTF8StringEncoding error:&error] : nil;
    if (source) {
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;
        metal_library = [device newLibraryWithSource:source options:options error:&error];
        fprintf(stderr, "Experimental runtime MSL compilation: %s\\n", metal_library ? "succeeded" : "failed");
    }

'''+old[end:];p.write_text(new)
    changes+=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='experiment/core/metal/msplat_metal.mm',tofile='runtime/core/metal/msplat_metal.mm'))
    with (O/'native_changes.patch').open('a') as f:f.write(changes)
    m=json.loads((O/'native_manifest.json').read_text());m['build_adjustment']='Offline xcrun metal missing. Use MTLDevice newLibraryWithSource runtime compilation; ship bundled MSL source. Runtime/compiler change is an additional confound beyond tile cap; add 2048-cap runtime control if needed.';(O/'native_manifest.json').write_text(json.dumps(m,indent=2))
    print('RUNTIME_COMPILER_PATCHED',flush=True)
if __name__=='__main__':main()
