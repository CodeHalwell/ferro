"""Verify explicitly selected fresh reports; never read/write archived defaults."""
import argparse
import json
import math
import re
import statistics
from pathlib import Path
from benchmark import MODES, error_status, PARITY_RTOL, PARITY_ATOL


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_report(report):
    require(report.get('parity_rtol') == PARITY_RTOL and report.get('parity_atol') == PARITY_ATOL, 'missing or changed parity tolerances; fresh benchmark required')
    require(report.get('modes') in (MODES, MODES[::-1]), 'invalid comparator modes/order')
    require(report.get('samples') == 40 and report.get('warmup') == 10, 'expected 40 samples after 10 warmups')
    expected = [(case, mode) for case in ['mlp', 'residual_mlp', 'transformer'] for mode in report['modes']]
    require([(row.get('case'), row.get('mode')) for row in report['rows']] == expected, 'incomplete, duplicate, unknown or out-of-order rows')
    passed, blocked = [], []
    for row in report['rows']:
        if row['status'] == 'passed':
            samples = row.get('samples_us')
            require(isinstance(samples, list) and len(samples) == report['samples'] and all(finite_nonnegative(v) and v > 0 for v in samples), 'expected 40 finite positive timing samples')
            for field, value in [('median_us', statistics.median(samples)), ('min_us', min(samples)), ('max_us', max(samples))]:
                require(finite_nonnegative(row.get(field)) and row[field] == value, f'invalid {field}')
            require(finite_nonnegative(row.get('prepare_ms')), 'invalid preparation timing')
            fields = ['freshness_max_abs_errors', 'freshness_reference_max_abs', 'freshness_max_tolerance_ratios']
            for field in fields:
                values = row.get(field)
                require(isinstance(values, list) and len(values) == 4 and all(finite_nonnegative(v) for v in values), f'invalid {field}: expected four finite nonnegative values')
            for error, reference, ratio in zip(*(row[field] for field in fields)):
                require(error <= PARITY_ATOL + PARITY_RTOL * reference, 'absolute freshness error exceeds declared tolerance')
                require(ratio <= 1, 'elementwise freshness error exceeds declared tolerance')
            passed.append({k: row[k] for k in ['case', 'mode', 'median_us', 'prepare_ms', 'freshness_max_abs_errors']})
        else:
            require(row['status'] == 'blocked' and 'samples_us' not in row, 'failed or unknown status')
            require(error_status(row['mode'], RuntimeError(row['error'])) == 'blocked', 'unrecognized blocker')
            blocked.append(dict(case=row['case'], mode=row['mode'], reason='Triton unavailable'))
    return dict(passed=passed, blocked=blocked)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs=2, type=Path, required=True, metavar=('FORWARD', 'REVERSE'))
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--integration-dir', type=Path, help='optional directory of integration-results.json and Rust logs')
    args = parser.parse_args(argv)
    paths = [path.resolve(strict=True) for path in args.runs]
    require(paths[0] != paths[1], 'two distinct run files required')
    require(args.summary.resolve() not in paths, 'summary must not overwrite an input')
    require(not args.summary.exists(), 'summary already exists; choose a fresh output path')
    reports = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    require(reports[0]['modes'] == reports[1]['modes'][::-1], 'second run must reverse comparator order')
    summary = dict(runs=[dict(input=str(path), **validate_report(report)) for path, report in zip(paths, reports)])
    if args.integration_dir:
        directory = args.integration_dir.resolve(strict=True)
        integration = json.loads((directory/'integration-results.json').read_text(encoding='utf-8'))
        require(len(integration) == 13 and all(r['exit_code'] == 0 for r in integration), 'integration did not pass all 13 commands')
        summary.update(integration_directory=str(directory), integration_commands=len(integration), rust_passes={
            name: sum(map(int, re.findall(r'test result: ok\. (\d+) passed', (directory/(name+'.log')).read_text(encoding='utf-8'))))
            for name in ['core', 'cpu-tokenizer', 'cuda', 'capture-layout-gpu']})
    with args.summary.open('x', encoding='utf-8') as output:
        output.write(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(benchmark_runs=len(summary['runs']), passed_per_run=[len(r['passed']) for r in summary['runs']], blocked_per_run=[len(r['blocked']) for r in summary['runs']])))


if __name__ == '__main__':
    main()
