"""Explicit, check-by-default launcher for the fixed 3.12 Host candidate."""
import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
REGISTRATION = ROOT / 'module-registry/real_chain_runner/plan-entry-312.json'
REQUIRED_ENV = ('GEMINI_API_KEY', 'MIMO_API_KEY', 'REAL_LLM_STAGES_ONLINE', 'REAL_MIMO_ONLINE')


def setup_resources():
    if sys.version_info[:2] != (3, 12):
        raise ValueError('this resource profile requires Python 3.12')
    sys.path.insert(0, str(ROOT))
    importlib.invalidate_caches()
    dist = importlib.metadata.distribution('json-repair')
    if dist.version != '0.58.3':
        raise ValueError('json-repair 0.58.3 is required')
    from host.runtime import verify_resource_pack
    registration = json.loads(REGISTRATION.read_text(encoding='utf-8'))
    profile = json.loads((ROOT / registration['profile']).read_text(encoding='utf-8'))
    if not profile.get('resource_pack'):
        raise ValueError('run tools/build_runtime.py to generate the verified resource pack')
    _, evidence = verify_resource_pack(profile['resource_pack'])
    return registration, len(evidence)


def load_config(filename):
    data = json.loads(Path(filename).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('plan_id') != 'CHAIN-READY-001' or not isinstance(data.get('config'), dict):
        raise ValueError('expected CHAIN-READY-001 configuration object')
    if set(data) - {'plan_id', 'config', 'required_environment'}:
        raise ValueError('unknown top-level configuration fields')
    if data.get('required_environment', list(REQUIRED_ENV)) != list(REQUIRED_ENV):
        raise ValueError('required_environment must contain the documented environment variable names')
    config = data['config']
    if config.get('retry', False) is not False:
        raise ValueError('launcher creates a new Host; retry requires the same live Host')
    required_files = ['media', 'enrollment', 'weights', 'memory']
    required_dirs = ['model', 'model_cache']
    for name in required_files + required_dirs:
        value = config.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(name + ' must be an absolute path')
        path = Path(value)
        if not (path.is_file() if name in required_files else path.is_dir()):
            raise ValueError(name + ' path missing or wrong type')
    if config.get('ass_reference') is not None:
        reference = config['ass_reference']
        if not isinstance(reference, str) or not Path(reference).is_absolute() or not Path(reference).is_file():
            raise ValueError('ass_reference must be an existing absolute file')
    if not isinstance(config.get('device'), str) or not config['device']:
        raise ValueError('device must be a nonempty string')
    resources = config.get('align_resources')
    kinds = {'cache': True, 'asr_model': True, 'align_model': True,
             'mfa_exe': False, 'mfa_dictionary': False, 'mfa_acoustic': False}
    if not isinstance(resources, dict) or set(resources) != set(kinds):
        raise ValueError('align_resources must specify all six paths')
    for name, directory in kinds.items():
        value = resources[name]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError('align_resources.' + name + ' must be an absolute path')
        path = Path(value)
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError('align_resources.' + name + ' path missing or wrong type')
    for name in ('prepare_options', 'transcribe_options'):
        values = config.get(name, [])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(name + ' must be a list of strings')
    options = config.get('transcribe_options', [])
    if '--asr-context-file' in options:
        index = options.index('--asr-context-file')
        if index + 1 == len(options) or not Path(options[index + 1]).is_absolute() or not Path(options[index + 1]).is_file():
            raise ValueError('ASR context file must be an existing absolute path')
    return data


def output_path(value):
    if not isinstance(value, str) or not re.fullmatch(r'runs/experiments/[A-Za-z0-9][A-Za-z0-9._-]*', value):
        raise ValueError('output must be a new runs/experiments/<id> directory')
    target = ROOT / value
    if target.exists() or target.is_symlink():
        raise ValueError('output already exists')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='explicit JSON configuration')
    parser.add_argument('--output', help='new runs/experiments/<id>; required for --execute')
    parser.add_argument('--execute', action='store_true', help='explicitly permit real model, tool and network stages')
    args = parser.parse_args(argv)
    launch_started = time.perf_counter()
    assembly_seconds = None
    launch_output = None
    try:
        data = load_config(args.config)
        if args.output:
            output_path(args.output)
        if args.execute and not args.output:
            raise ValueError('--output is required with --execute')
        if args.execute:
            if any(not os.environ.get(name) for name in REQUIRED_ENV[:2]):
                raise ValueError('required API key environment variables are missing')
            if any(os.environ.get(name) != '1' for name in REQUIRED_ENV[2:]):
                raise ValueError('both documented online opt-in environment variables must equal 1')
        registration, count = setup_resources()
        if not args.execute:
            print(json.dumps({'mode': 'check', 'status': 'ok', 'resource_distributions': count,
                              'json_repair': '0.58.3', 'models_or_network_tested': False}))
            return 0
        from host.runtime import Host
        assembly_started = time.perf_counter()
        host = Host(ROOT, args.output, registration)
        assembly_seconds = time.perf_counter() - assembly_started
        launch_output = ROOT / args.output
        try:
            result = host.invoke(host.envelope({'run_id': 'bootstrap-run', 'binding_id': 'bootstrap'},
                                               registration['entry'],
                                               {'plan_id': data['plan_id'], 'config': data['config']}))
        finally:
            host.end_runs()
        report = result.get('returned', {}).get('report', {}) if result.get('outcome') == 'returned' else {}
        if result.get('outcome') != 'returned' or report.get('status') != 'completed':
            category = result.get('raised', {}).get('category', 'IncompleteBusinessReport')
            print(json.dumps({'mode': 'execute', 'status': 'failed', 'category': category, 'output': args.output}))
            return 1
        print(json.dumps({'mode': 'execute', 'status': 'completed', 'output': args.output,
                          'attempt': report.get('attempt')}))
        return 0
    except Exception as exc:
        # Do not print environment variables, exception values or provider responses.
        print(json.dumps({'mode': 'execute' if args.execute else 'check', 'status': 'failed',
                          'error_type': type(exc).__name__}), file=sys.stderr)
        return 2
    finally:
        if launch_output is not None:
            try:
                payload = {'schema': 1, 'wall_seconds': time.perf_counter() - launch_started,
                           'host_assembly_seconds': assembly_seconds}
                (launch_output / 'launch-performance.json').write_text(
                    json.dumps(payload, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
            except OSError:
                pass


if __name__ == '__main__':
    raise SystemExit(main())
