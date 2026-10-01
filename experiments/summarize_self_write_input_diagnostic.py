"""Compact observer diagnostics, with source version checks."""
import hashlib
import json
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parents[1]
    runs = root / 'runs/self_write_input_diagnostic'
    paths = ('seed_11_no_write.json', 'target_complete/seed_11_trained_write.json',
             'delta/seed_11_frozen_write.json', 'delta/seed_11_trained_write.json')
    records = []
    for relative in paths:
        result = json.loads((runs/relative).read_text(encoding='utf-8'))
        for name, expected in result['source_hashes'].items():
            candidates = (root/name, runs/'source_snapshot'/Path(name).name)
            assert any(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == expected
                       for p in candidates)
        credit = result.pop('credit_last_100')
        result['credit_mean_last_100'] = {name: statistics.mean(r[name] for r in credit)
            for name in ('neural_path_norm', 'bias_path_norm', 'total_norm')} if credit else {}
        result['raw_path'] = 'runs/self_write_input_diagnostic/' + relative
        records.append(result)
    output = root/'reports/self_write_input_diagnostic_2026-10-01.json'
    output.write_text(json.dumps(records, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps([{'mode': r['mode'], 'write_mode': r.get('write_mode', 'target'),
        'hand_state_probe': r['probes']['hand_z_at_action']['future_accuracy'],
        'hand_supervised_probe': r['original_hand_supervised_probe']['future_accuracy'],
        'policy_future': r['policy_future']} for r in records], indent=2))


if __name__ == '__main__':
    main()
