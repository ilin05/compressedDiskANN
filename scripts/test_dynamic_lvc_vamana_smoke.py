#!/usr/bin/env python3
"""Small native-graph smoke test; requires numpy and the two newly built tools."""
import argparse
import importlib.util
from pathlib import Path
import struct
import subprocess
import tempfile
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--prepare-binary', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    root = a.output or Path(tempfile.mkdtemp(prefix='dynamic-vamana-smoke-'))
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(37)
    x = rng.integers(0,256,(1024,128)).astype('<f4')
    queries = (x[:64] + rng.integers(-3,4,(64,128))).astype('<f4')
    def save(path, values):
        with path.open('wb') as f:
            f.write(struct.pack('<II',*values.shape)); f.write(values.tobytes())
    save(root/'base.fbin',x); save(root/'query.fbin',queries)
    exact = np.square(queries[:,None,:]-x[None,:,:]).sum(axis=2).argmin(axis=1).astype('<u4')[:,None]
    save(root/'gt.bin',exact)
    subprocess.run([str(a.prepare_binary),str(root/'base.fbin'),str(root/'raw_index'),str(root/'alp_index'),'16','32','1'],
                   check=True,stdout=(root/'prepare.log').open('w'),stderr=subprocess.STDOUT)
    # Valid native DiskANN PQ file layout, using deterministic sampled codewords.
    table = x[:256].copy(); chunks = 32
    parts = [struct.pack('<II',256,128)+table.tobytes(),
             struct.pack('<II',128,1)+np.zeros(128,dtype='<f4').tobytes(),
             struct.pack('<II',chunks+1,1)+np.arange(0,129,4,dtype='<u4').tobytes()]
    offsets = [4096]
    for part in parts: offsets.append(offsets[-1]+len(part))
    with (root/'base.fbinpq32_pivots.bin').open('wb') as f:
        f.write(struct.pack('<II4Q',4,1,*offsets)); f.write(bytes(4096-f.tell()))
        for part in parts: f.write(part)
    codes = np.empty((len(x),chunks),dtype='u1')
    for c in range(chunks):
        codes[:,c] = np.square(x[:,None,4*c:4*c+4]-table[None,:,4*c:4*c+4]).sum(axis=2).argmin(axis=1)
    save(root/'base.fbinpq32_compressed.bin',codes)
    variants = [('raw','raw',1,'1',1,False),('alp_off','alp',1,'1',1,False),
                ('alp_alpha1','alp',1,'1',1,True),('alp_nocache','alp',1,'1',0,False),
                ('alp_a01','alp',0.1,'1,0.2',1,True)]
    import json
    for name,backend,alpha,query_alpha,cache,pq in variants:
        d = root/name; d.mkdir()
        (d/'config.json').write_text(json.dumps(dict(insert_alpha=alpha)))
        command = [str(a.binary),'--backend',backend,'--graph-prefix',str(root/'raw_index'),'--payload-prefix',str(root/('raw_index' if backend=='raw' else 'alp_index')),
                   '--data',str(root/'base.fbin'),'--query',str(root/'query.fbin'),'--gt',str(root/'gt.bin'),'--output',str(d),
                   '--batches','2','--ratio','0.05','--rounds','2','--search-every','1','--L','1,5,10',
                   '--insert-alpha',str(alpha),'--query-alpha',query_alpha,'--insert-cache',str(cache),'--dump-results','1']
        if pq: command += ['--pq-prefix',str(root/'base.fbin'),'--pq-bytes','32']
        with (d/'run.log').open('w') as stream:
            subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
    spec=importlib.util.spec_from_file_location('runner',Path(__file__).with_name('run_dynamic_lvc_vamana.py'))
    runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    runner.aggregate(root); runner.validate_correctness(root)
    print('Smoke results:',root)


if __name__=='__main__': main()
