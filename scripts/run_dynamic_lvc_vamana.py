#!/usr/bin/env python3
"""Serial Raw/ALP repeated Vamana updates, with shared static graph and stable GT IDs."""
import argparse
import csv
import json
import os
from pathlib import Path
import struct
import subprocess


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, values):
    if not values:
        return
    path = Path(path)
    temporary = path.with_suffix('.tmp.csv')
    with temporary.open('w', newline='') as stream:
        out = csv.DictWriter(stream, fieldnames=list(values[0]))
        out.writeheader()
        out.writerows(values)
    temporary.replace(path)


def shape(path):
    with Path(path).open('rb') as stream:
        n, d = struct.unpack('<II', stream.read(8))
    if not n or not d or Path(path).stat().st_size != 8 + n * d * 4:
        raise ValueError(f'Expected F32 bin: {path}')
    return n, d


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def checked(command, log=None, cwd=None, env=None):
    print('RUN', ' '.join(map(str, command)), flush=True)
    if log:
        with Path(log).open('w') as stream:
            result = subprocess.run(list(map(str, command)), cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Exit {result.returncode}; see {log}')
    else:
        subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def aggregate(root):
    updates, searches, summary = [], [], []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / 'config.json').exists():
            continue
        config = json.loads((directory / 'config.json').read_text())
        if not (directory / 'updates.csv').exists():
            continue
        us = rows(directory / 'updates.csv')
        for row in us:
            row = dict(variant=directory.name, insert_alpha=config['insert_alpha'], **row)
            updates.append(row)
        ss = rows(directory / 'search.csv') if (directory / 'search.csv').exists() else []
        grouped = {}
        for row in ss:
            searches.append(dict(variant=directory.name, insert_alpha=config['insert_alpha'], **row))
            grouped.setdefault((row['batch'], row['query_alpha'], row['L']), []).append(row)
        by_batch = {row['batch']: row for row in us}
        for (batch, alpha, l), group in grouped.items():
            values = dict(variant=directory.name, insert_alpha=config['insert_alpha'], **by_batch[batch])
            values.update(query_alpha=alpha, L=l, rounds=len(group))
            for key in ('recall_at_1', 'qps', 'mean_query_us', 'empty_results'):
                values[key] = sum(float(row[key]) for row in group) / len(group)
            summary.append(values)
    write_csv(root / 'all_updates.csv', updates)
    write_csv(root / 'all_search.csv', searches)
    write_csv(root / 'summary.csv', summary)


def validate_correctness(root):
    raw = root / 'raw'
    if not (raw / 'COMPLETE').exists():
        print('Correctness comparison pending: complete the raw run in the same output directory.', flush=True)
        return
    reference = rows(raw / 'updates.csv')
    for name in ('alp_off', 'alp_alpha1', 'alp_nocache'):
        directory = root / name
        if not (directory / 'COMPLETE').exists():
            continue
        other = rows(directory / 'updates.csv')
        if len(other) != len(reference):
            raise AssertionError(f'{name}: missing batches')
        for a, b in zip(reference, other):
            for key in ('batch', 'active_nodes', 'physical_nodes', 'deleted_nodes', 'update_hash', 'graph_hash'):
                if a[key] != b[key]:
                    raise AssertionError(f'{name}: {key} mismatch in batch {a["batch"]}')
            if float(b['max_abs_error']) != 0:
                raise AssertionError('BIGANN integer-F32 must be bitwise recoverable')
        with (raw / 'results.csv').open() as ra, (directory / 'results.csv').open() as rb:
            # alpha is 1 in both controls. Includes empty results and exact F32 distance bits.
            from itertools import zip_longest
            for line_a, line_b in zip_longest(ra, rb):
                if line_a != line_b:
                    raise AssertionError(f'{name}: exact search result/distance mismatch')
        print(f'PASS {name}: raw graph, stable IDs, exact distance bits, update counts', flush=True)
    for directory in root.iterdir():
        if directory.is_dir() and (directory / 'COMPLETE').exists():
            for row in rows(directory / 'updates.csv'):
                if float(row['max_abs_error']) != 0:
                    raise AssertionError(f'{directory.name}: BIGANN reconstruction is not exact')
    completed = all((root / name / 'COMPLETE').exists() for name in ('raw','alp_off','alp_alpha1','alp_nocache','alp_a01'))
    if completed:
        (root / 'CORRECTNESS_PASS').write_text('PASS raw/exact-ALP, cache off/on, PQ alpha=1, filtered updates\n')
        print('PASS BIGANN dynamic Vamana correctness matrix', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('correctness','pilot','full'), required=True)
    parser.add_argument('--mode', choices=('all','raw','alp'), default=os.getenv('MODE','all'))
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--build-dir', type=Path, default=os.getenv('BUILD_DIR'))
    parser.add_argument('--data-dir', type=Path, default=os.getenv('DATA_DIR'))
    for key in ('raw-prefix','alp-prefix','data','query','gt','pq-prefix','output','raw-binary','lvc-binary'):
        parser.add_argument('--'+key, type=Path, default=os.getenv(key.upper().replace('-','_')))
    parser.add_argument('--pq-bytes', type=int, default=int(os.getenv('PQ_BYTES','32')))
    parser.add_argument('--insert-alpha', type=float, default=float(os.getenv('INSERT_ALPHA','0.1')))
    parser.add_argument('--batches', type=int, default=os.getenv('BATCHES'))
    parser.add_argument('--ratio', type=float, default=os.getenv('BATCH_RATIO'))
    parser.add_argument('--L', default=os.getenv('SEARCH_L'))
    parser.add_argument('--rounds', type=int, default=int(os.getenv('ROUNDS','3')))
    parser.add_argument('--search-every', type=int, default=os.getenv('SEARCH_EVERY'))
    parser.add_argument('--query-count', type=int, default=int(os.getenv('QUERY_COUNT','0')))
    parser.add_argument('--seed', type=int, default=int(os.getenv('SEED','20261015')))
    parser.add_argument('--prepare', action='store_true', default=os.getenv('PREPARE','0')=='1')
    parser.add_argument('--skip-build', action='store_true', default=os.getenv('SKIP_BUILD','0')=='1')
    parser.add_argument('--resume', action='store_true', default=os.getenv('RESUME','0')=='1')
    args = parser.parse_args()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    args.repo = args.repo.resolve()
    build = (args.build_dir or args.repo/'build').resolve()
    data_dir = (args.data_dir or build/'data/bigann').resolve()
    small = args.stage == 'correctness'
    size = '1M' if small else '100M'
    n_expected = 1_000_000 if small else 100_000_000
    raw = (args.raw_prefix or data_dir/f'memory_index_bigann_{size}_R32_L50_A1.2').resolve()
    alp = (args.alp_prefix or data_dir/f'compressed_memory_index_bigann_{size}_R32_L50_A1.2').resolve()
    canonical_data = data_dir/f'bigann_{size}.fbin'
    data = (args.data or canonical_data).resolve()
    if args.data is None and not data.exists() and Path(str(raw)+'.data').exists():
        data = Path(str(raw)+'.data')  # the existing raw F32 payload is already a valid fbin
    query = (args.query or data_dir/'bigann_query.fbin').resolve()
    gt = (args.gt or data_dir/f'bigann_{size}_gt.bin').resolve()
    pq = (args.pq_prefix or canonical_data).resolve()
    root = (args.output or args.repo/f'scripts/dynamic_lvc_vamana_results/bigann_{size}_{args.stage}').resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Fail before expensive preparation if queries or the matching GT are missing.
    identity(query)
    identity(gt)
    lvc_binary = (args.lvc_binary or build/'apps/test_dynamic_lvc_vamana').resolve()
    raw_binary = (args.raw_binary or lvc_binary).resolve()
    if not args.skip_build:
        targets = ['test_dynamic_lvc_vamana']
        if args.prepare:
            targets += ['prepare_dynamic_lvc_vamana','generate_pq']
        checked(['cmake','-S',args.repo,'-B',build,'-DCMAKE_BUILD_TYPE=Release'])
        checked(['cmake','--build',build,'--target',*targets,'-j',os.getenv('BUILD_JOBS','8')])
    if args.prepare:
        if not small:
            raise ValueError('--prepare is only for the 1M correctness fixture; existing 100M indexes are reused')
        if not data.exists():
            source = data_dir/'bigann_100M.fbin'
            n, dim = shape(source)
            if n < n_expected:
                raise ValueError('Not enough source rows')
            tmp = data.with_suffix('.pending')
            with source.open('rb') as inp, tmp.open('wb') as out:
                inp.seek(8); out.write(struct.pack('<II',n_expected,dim))
                remaining = n_expected*dim*4
                while remaining:
                    block = inp.read(min(remaining,16*1024*1024))
                    if not block:
                        raise EOFError(source)
                    out.write(block); remaining -= len(block)
            tmp.replace(data)
        checked([build/'apps/prepare_dynamic_lvc_vamana',data,raw,alp,32,50,os.getenv('PREPARE_THREADS','32')],root/'prepare.log')
        native_pivots = Path(str(pq)+f'pq{args.pq_bytes}_pivots.bin')
        native_codes = Path(str(pq)+f'pq{args.pq_bytes}_compressed.bin')
        if not native_pivots.exists() or not native_codes.exists():
            if native_pivots.exists() or native_codes.exists() or native_pivots.is_symlink() or native_codes.is_symlink():
                raise RuntimeError('Incomplete/broken native PQ pair; repair both codebook and codes before preparation')
            generated = Path(str(pq)+f'_dynamic_pq{args.pq_bytes}')
            preparation_env = os.environ.copy()
            preparation_env['OMP_NUM_THREADS'] = os.getenv('PREPARE_THREADS','32')
            checked([build/'apps/utils/generate_pq','float',data,generated,args.pq_bytes,0.1,0],
                    root/'prepare_pq.log',env=preparation_env)
            # generate_pq's standalone naming differs from PQDataStore's naming.
            for source, target in ((Path(str(generated)+'_pq_pivots.bin'),native_pivots),
                                   (Path(str(generated)+'_pq_compressed.bin'),native_codes)):
                if target.exists():
                    raise FileExistsError(target)
                os.symlink(source.resolve(), target)
    n, dim = shape(data)
    if n != n_expected:
        raise ValueError(f'{args.stage} expects {n_expected} rows, got {n}; override source paths, not the dataset semantics')
    if shape(query)[1] != dim:
        raise ValueError('Query dimension mismatch')
    batches = args.batches if args.batches is not None else (20 if args.stage=='full' else 2)
    ratio = args.ratio if args.ratio is not None else (0.001 if args.stage=='pilot' else 0.01)
    ls = args.L or ('10' if args.stage=='pilot' else ('5,10,20' if args.stage=='full' else '1,2,5,10'))
    every = args.search_every if args.search_every is not None else (2 if args.stage=='full' else 1)
    variants = [('raw','raw',1.0,'1',True,False)]
    if small:
        variants += [('alp_off','alp',1.0,'1',True,False),('alp_alpha1','alp',1.0,'1',True,True),
                     ('alp_nocache','alp',1.0,'1',False,False),('alp_a01','alp',0.1,'1,0.2',True,True)]
    else:
        variants += [(f'alp_a{args.insert_alpha:g}','alp',args.insert_alpha,'1,0.2',True,True)]
    for name, backend, insert_alpha, query_alpha, cache, use_pq in variants:
        if args.mode != 'all' and backend != args.mode:
            continue
        binary = raw_binary if backend=='raw' else lvc_binary
        files = [data,query,gt,raw,Path(str(raw if backend=='raw' else alp)+'.data'),binary]
        if use_pq:
            files += [Path(str(pq)+f'pq{args.pq_bytes}_pivots.bin'),Path(str(pq)+f'pq{args.pq_bytes}_compressed.bin')]
        config = dict(stage=args.stage,backend=backend,insert_alpha=insert_alpha,query_alpha=query_alpha,
                      batches=batches,ratio=ratio,L=ls,rounds=args.rounds,search_every=every,seed=args.seed,
                      query_count=args.query_count,insert_cache=cache,pq_bytes=args.pq_bytes,
                      files=[identity(path) for path in files])
        directory = root/name
        if directory.exists():
            previous = directory/'config.json'
            same = previous.exists() and json.loads(previous.read_text())==config
            if args.resume and same and (directory/'COMPLETE').exists():
                print('REUSE',name,flush=True); aggregate(root); continue
            raise RuntimeError(f'{directory} already exists; RESUME=1 reuses only a complete matching configuration; '
                               'choose a new OUTPUT or move the incomplete directory before restarting')
        directory.mkdir()
        (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        command = [binary,'--backend',backend,'--graph-prefix',raw,'--payload-prefix',raw if backend=='raw' else alp,
                   '--data',data,'--query',query,'--gt',gt,'--output',directory,'--batches',batches,'--ratio',ratio,
                   '--L',ls,'--rounds',args.rounds,'--search-every',every,'--insert-alpha',insert_alpha,
                   '--query-alpha',query_alpha,'--insert-cache',int(cache),'--dump-results',int(small),
                   '--query-count',args.query_count,'--seed',args.seed]
        if use_pq:
            command += ['--pq-prefix',pq,'--pq-bytes',args.pq_bytes]
        try:
            checked(command,directory/'run.log')
        finally:
            aggregate(root)
        print('PASS',name,'results:',directory,flush=True)
    if small:
        validate_correctness(root)
    print('Results:',root/'summary.csv',flush=True)


if __name__=='__main__':
    main()
