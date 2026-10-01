"""Package this conversation's reviewed evidence; never execute a learner.

Build needs the local runs referenced by the stage summaries. Verification of
the published bundle needs only the repository and Python's standard library.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/acnt_self_write_session_2026-10-02'
STEMS = (
    'original_acnt_online', 'self_write_bootstrap', 'self_write_input_diagnostic',
    'address_write_online', 'grouped_write_online', 'grouped_write_feasibility',
    'calibrated_write_online', 'write_vector_budget', 'write_group_limit',
    'write_structure_limit', 'calibrated_write_benchmark',
    'calibrated_write_benchmark_4_threads', 'calibrated_write_10min',
    'goodness_prediction_10min', 'bounded_goodness_10min',
)
REPORTS = tuple(ROOT / f'reports/{stem}_2026-10-01.json' for stem in STEMS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()
                if k not in ('rows', 'queries', 'feedback', 'parameter_changes', 'per_decision_seconds')}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def source_index():
    paths = list((ROOT/'acnt').glob('*.py')) + list((ROOT/'experiments').rglob('*.py'))
    for area in ('original_acnt_online', 'self_write_online', 'self_write_input_diagnostic', 'address_write_online',
                 'grouped_write_online', 'calibrated_write_online', 'calibrated_write_duration'):
        paths.extend(p for p in (ROOT/'runs'/area).rglob('*.py') if 'source_snapshot' in str(p))
    index = {}
    for path in sorted(set(paths)):
        index.setdefault(digest(path), path)
    return index


def publish_sources(hashes, index, files):
    mapping = {}
    for name, expected in hashes.items():
        assert name.endswith('.py') and len(expected) == 64, (name, expected)
        source = index.get(expected)
        if source is None:
            raise FileNotFoundError(f'No matching source for {name}: {expected}')
        target = BASE/'sources'/f'{expected}.py'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert digest(target) == expected
        else:
            target.write_bytes(source.read_bytes())
        assert digest(target) == expected
        relative = target.relative_to(ROOT).as_posix()
        files[relative] = dict(sha256=expected, bytes=target.stat().st_size)
        mapping[name] = relative
    return mapping


def build():
    summaries = {p.relative_to(ROOT).as_posix(): read(p) for p in REPORTS}
    raw_paths = set()
    for value in summaries.values():
        for record in walk(value):
            for key in ('artifact', 'raw_path', 'raw_result'):
                relative = record.get(key)
                if isinstance(relative, str) and relative.replace('\\', '/').startswith('runs/') and relative.endswith('.json'):
                    raw_paths.add(relative.replace('\\', '/'))
    assert raw_paths
    index, files, records = source_index(), {}, []
    for i, relative in enumerate(sorted(raw_paths), 1):
        raw = ROOT/relative
        data = read(raw)
        for key in ('neural_resets', 'subject_reset_count', 'replay_count'):
            if key in data:
                assert data[key] == 0, (relative, key)
        if 'rows' in data and 'decisions' in data:
            assert len(data['rows']) == data['decisions'], relative
        hashes = data.get('source_hashes', {})
        assert hashes, relative
        sources = publish_sources(hashes, index, files)
        published = BASE/'records'/f'{i:03d}.json'
        save(published, compact(data))
        public_path = published.relative_to(ROOT).as_posix()
        files[public_path] = dict(sha256=digest(published), bytes=published.stat().st_size)
        records.append(dict(local_raw_path=relative, local_raw_sha256=digest(raw),
                            local_raw_bytes=raw.stat().st_size, published_record=public_path,
                            seed=data.get('seed'), decisions=data.get('decisions'), sources=sources))
    # Structural diagnostics and benchmarks also retain exactly the source
    # versions their files claim; they are not behavioural learning runs.
    diagnostic_sources = {}
    for relative, data in summaries.items():
        hashes = data.get('source_hashes', {}) if isinstance(data, dict) else {}
        if isinstance(data, dict) and data.get('source_sha256'):
            hashes = {'experiments/grouped_write_feasibility.py': data['source_sha256']}
        if hashes:
            diagnostic_sources[relative] = publish_sources(hashes, index, files)
    plot(summaries)
    for path in REPORTS:
        for extension in ('.json', '.md', '.png', '.svg'):
            artifact = path.with_suffix(extension)
            if artifact.exists():
                if extension == '.svg':
                    artifact.write_text('\n'.join(line.rstrip() for line in artifact.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
                files[artifact.relative_to(ROOT).as_posix()] = dict(sha256=digest(artifact), bytes=artifact.stat().st_size)
    for path in (BASE.with_suffix('.md'), BASE.with_suffix('.png'), BASE.with_suffix('.svg')):
        if path.suffix == '.svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
        files[path.relative_to(ROOT).as_posix()] = dict(sha256=digest(path), bytes=path.stat().st_size)
    manifest = dict(scope='reviewed experiments in this conversation; regenerated reports, not rerun training',
                    assembled_date='2026-10-02', timezone='Asia/Shanghai',
                    source_run_dates='2026-10-01 to 2026-10-02; stage filenames retain their original date',
                    raw_trajectories_published=False, checkpoints_published=False,
                    raw_records_verified=len(records), source_versions=len(list((BASE/'sources').glob('*.py'))),
                    records=records, diagnostic_sources=diagnostic_sources,
                    artifact_hashes=files, validation='all stage summarizers passed before assembly')
    save(BASE/'manifest.json', manifest)
    verify()


def verify():
    manifest = read(BASE/'manifest.json')
    for name, metadata in manifest['artifact_hashes'].items():
        path = (ROOT/name).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(), name
        assert path.stat().st_size == metadata['bytes'] and digest(path) == metadata['sha256'], name
    for record in manifest['records']:
        data = read(ROOT/record['published_record'])
        assert 'rows' not in data
        assert set(data['source_hashes']) == set(record['sources'])
        for name, target in record['sources'].items():
            assert digest(ROOT/target) == data['source_hashes'][name], name
    for report, mappings in manifest['diagnostic_sources'].items():
        data = read(ROOT/report)
        hashes = data.get('source_hashes') or {'experiments/grouped_write_feasibility.py': data['source_sha256']}
        for name, target in mappings.items():
            assert digest(ROOT/target) == hashes[name]
    summaries = {p.stem: read(p) for p in REPORTS}
    for stem in ('calibrated_write_10min', 'goodness_prediction_10min', 'bounded_goodness_10min'):
        data = summaries[stem+'_2026-10-01']
        assert data['validation_passed'] and data['final_finite']
        assert data['requested_wall_seconds'] == 600 and data['observed_wall_seconds'] >= 600
        assert data['subject_reset_count'] == data['replay_count'] == data['outer_parameter_updates'] == 0
    bounded = summaries['bounded_goodness_10min_2026-10-01']
    m = bounded['verified_final_metrics']
    assert 0 <= bounded['total_range'][0] <= bounded['total_range'][1] <= 1
    assert math.isclose(m['total_goodness'], m['main_base']+m['evaluation_correction'], abs_tol=1e-12)
    assert math.isclose(m['evaluation_correction'], -.2*m['prediction_mse'], abs_tol=1e-12)
    print(json.dumps(dict(published_bundle_verified=True, records=manifest['raw_records_verified'],
                         source_versions=manifest['source_versions'], artifacts=len(manifest['artifact_hashes'])), ensure_ascii=False))


def plot(summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = (
        ('Fixed AB / seed 44', 'fixed_cue_seed_44_600s_v1'),
        ('AB + predict raw R / seed 55', 'goodness_seed_55_600s_v1'),
        ('AB + predict total G / seed 66', 'bounded_total_seed_66_600s_v1'),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), sharey=True)
    for axis, (label, folder) in zip(axes, cases):
        directory = ROOT/'runs/calibrated_write_duration'/folder
        result = read(directory/'result.json')
        rows = [json.loads(line) for line in (directory/'observer.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows) == result['decisions']
        windows = [rows[i:i+200] for i in range(0, len(rows)-199, 200)]
        # Include the actual cutoff window rather than omit the final bin.
        # The appended full 200-decision window may overlap its predecessor.
        if len(rows) % 200:
            windows.append(rows[-200:])
        xs = [w[-1]['elapsed_wall_seconds']/60 for w in windows]
        axis.plot(xs, [sum(r['goodness'] for r in w)/200 for w in windows], label='Teacher G')
        if result['task'] == 'bounded_total_goodness':
            axis.plot(xs, [sum(r['action_goodness'] for r in w)/200 for w in windows], label='Raw action accuracy')
        else:
            axis.plot(xs, [sum(r['expected_goodness'] for r in w)/200 for w in windows], label='Correct-action probability', linestyle='--')
        axis.set_title(label, fontsize=10)
        axis.set_xlabel('Compute wall time (minutes)')
        axis.set_ylim(0, 1.04)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc='lower right')
        axis.plot(xs[-1], sum(r['goodness'] for r in windows[-1])/200, 'o', color='C0', markersize=4)
    axes[0].set_ylabel('Mean over 200 decisions')
    fig.suptitle('Three independent 10-minute lives; reward definitions and seeds differ', fontsize=12)
    fig.tight_layout()
    fig.savefig(BASE.with_suffix('.png'), dpi=170)
    fig.savefig(BASE.with_suffix('.svg'))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-published', action='store_true', help='verify the public bundle without local runs')
    args = parser.parse_args()
    verify() if args.verify_published else build()
