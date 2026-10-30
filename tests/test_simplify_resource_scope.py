"""Finite offline scope tests; no model, MFA executable or real subprocess."""
import ast
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

from host.runtime import execute_source
from modules.temporary.real_chain_runner import media
from modules.temporary.real_align_recovery import recover

ROOT = Path(__file__).resolve().parents[1]


def observed(call):
    try:
        return ('returned', call())
    except Exception as exc:
        return ('raised', type(exc).__name__, str(exc))


@dataclass
class WorkPaths:
    audio_path: Path
    aligned_manifest: Path

    @classmethod
    def from_workdir(cls, work):
        return cls(work / 'audio.wav', work / 'aligned.json')


class ScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = []
        for name, module, function in [('real_chain_runner/media', media, 'isolated'),
                                       ('real_align_recovery/recover', recover, 'run_mfa')]:
            path = 'modules/temporary/' + name + '.py'
            source = (ROOT / 'tests/fixtures/baselines/66e8ff6' / path).read_bytes()
            old = execute_source('_scope_baseline_' + name.replace('/', '_'), ROOT / path, source)
            cls.pairs.append((old, module, source, function))
        cls.old_media, cls.old_recover = [pair[0] for pair in cls.pairs]

    def media_case(self, module, work, mode):
        cwd, paths, environ = Path.cwd(), list(sys.path), dict(os.environ)
        chdir = os.chdir
        events = []
        def tracked_chdir(path):
            events.append(('chdir', str(path), os.environ.get('HOME'), list(sys.path)))
            if (mode == 'entry' and Path(path) == work) or (mode == 'exit' and Path(path) == cwd):
                raise OSError(mode)
            chdir(path)
        def body():
            with module.isolated(work):
                events.append(('body', str(Path.cwd()), dict(os.environ), list(sys.path)))
                if mode == 'business':
                    raise ValueError('body')
                if mode == 'nested':
                    with module.isolated(work / 'inner'):
                        events.append(('inner', str(Path.cwd()), os.environ['HOME']))
                    events.append(('outer', str(Path.cwd()), os.environ['HOME']))
        try:
            with patch.object(os, 'chdir', tracked_chdir):
                result = observed(body)
            restored = (Path.cwd() == cwd, sys.path == paths, dict(os.environ) == environ)
            return result, events, restored
        finally:
            chdir(cwd)
            sys.path[:] = paths
            os.environ.clear()
            os.environ.update(environ)

    def test_media_baseline_and_nested(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'HOME': 'old-home', 'TMP': ''}):
            work = Path(directory).resolve()
            os.environ.pop('HF_HUB_OFFLINE', None)
            for mode in ('ok', 'business', 'nested'):
                with self.subTest(mode=mode):
                    old = self.media_case(self.old_media, work, mode)
                    new = self.media_case(media, work, mode)
                    self.assertEqual(old, new)
                    self.assertEqual(new[2], (True, True, True))

    def test_media_entry_failure_now_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory).resolve()
            old = self.media_case(self.old_media, work, 'entry')
            new = self.media_case(media, work, 'entry')
            self.assertEqual(old[0], new[0])
            self.assertEqual(old[2], (True, False, False))
            self.assertEqual(new[2], (True, True, True))

    def test_media_failed_cwd_restore_does_not_skip_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, restored = self.media_case(media, Path(directory).resolve(), 'exit')
            self.assertEqual(result, ('raised', 'OSError', 'exit'))
            self.assertEqual(restored, (False, True, True))

    def test_media_legacy_write_rejected(self):
        for module in (self.old_media, media):
            with self.assertRaisesRegex(ValueError, 'must not be in legacy'):
                with module.isolated(module.LEGACY / 'forbidden-scope-test'):
                    self.fail('must not enter')

    def mfa_case(self, module, work, mode):
        events = []
        original_cwd, chdir = Path.cwd(), os.chdir
        root_fn = lambda project_root=None: 'original-root'
        command_fn = lambda project_root=None: 'original-command'
        def run(command, **kwargs):
            events.append(('command', command, kwargs))
            return NS(returncode=3, stderr='error', stdout='out')
        proc = NS(run=run)
        class Environment(ModuleType):
            def __setattr__(self, name, value):
                if name in ('find_project_mfa_root', 'build_project_mfa_command') and hasattr(self, name):
                    original = root_fn if name == 'find_project_mfa_root' else command_fn
                    events.append(('binding', name, 'original' if value is original else 'injected'))
                    if mode == 'cleanup' and name == 'find_project_mfa_root' and value is root_fn:
                        raise OSError('cleanup')
                super().__setattr__(name, value)
        env = Environment('qwen_asr.mfa_environment')
        env.find_project_mfa_root, env.build_project_mfa_command = root_fn, command_fn
        env.subprocess = proc
        runner = ModuleType('qwen_asr.mfa_runner')
        runner.subprocess = proc
        def local(paths, candidates, **kwargs):
            command = kwargs.pop('run_command')
            factory = kwargs.pop('environ_factory')
            self.assertEqual(factory(), dict(os.environ))
            events.append(('runner', paths, candidates, kwargs))
            command(['align'], env={'custom': 'kept'}, timeout=42)
            return [{'status': 'completed'}]
        runner.run_local_mfa_alignment_experiments = local
        models = ModuleType('qwen_asr.models')
        models.WorkPaths = WorkPaths
        executor = ModuleType('qwen_asr.recovery_executor')
        def execute(paths, **kwargs):
            injected = kwargs.pop('mfa_runner')
            events.append(('execute', paths, kwargs, env.find_project_mfa_root(), env.build_project_mfa_command()))
            env.subprocess.run(['version'])
            injected(paths, [{'id': 'one'}], environment={'extra': 'preserved'}, max_run_candidates=1, padding_ms=700)
            if mode == 'business':
                raise ValueError('business')
            return {'status': 'completed', 'backend_evidence': {'lab_text': 'sample'}}
        executor.execute_alignment_recovery = execute
        package = ModuleType('qwen_asr')
        package.__path__ = []
        modules = {'qwen_asr': package}
        for sub in (env, runner, models, executor):
            modules[sub.__name__] = sub
            setattr(package, sub.__name__.split('.')[-1], sub)
        def tracked_chdir(path):
            events.append(('chdir', str(path), env.find_project_mfa_root is root_fn,
                           env.build_project_mfa_command is command_fn, proc.run is run))
            if mode == 'entry' and Path(path) == work:
                raise OSError('entry')
            chdir(path)
        keys = ('MFA_ROOT_DIR HOME USERPROFILE TEMP TMP APPDATA LOCALAPPDATA XDG_CACHE_HOME '
                'XDG_CONFIG_HOME XDG_DATA_HOME PYTHONDONTWRITEBYTECODE PYTHONPYCACHEPREFIX PATH').split()
        executable = work / 'fake-executable'
        executable.write_text('not executable', encoding='utf-8')
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, modules))
                stack.enter_context(patch.object(os, 'chdir', tracked_chdir))
                for name, value in {
                    '_legacy': lambda: None, 'MFA_EXE': executable,
                    'copy_mfa_root': lambda work: work / 'mfa-root',
                    'assert_outside': lambda *args: None, 'assert_mfa_version_clean': lambda *args: None,
                    'mfa_process_env': lambda *args: {key: str(work / key) for key in keys},
                    'progress': lambda *args, **kwargs: events.append(('progress', args, kwargs)),
                    'segment_row': lambda *args: {'tokens': [], 'text': 'sample', 'status': 'completed'},
                    '_public_mfa': lambda attempt: dict(attempt),
                }.items():
                    stack.enter_context(patch.object(module, name, value))
                result = observed(lambda: module.run_mfa(work, 'segment', work / 'source.wav', include_row=True))
                restored = (env.find_project_mfa_root is root_fn, env.build_project_mfa_command is command_fn,
                            proc.run is run, Path.cwd() == original_cwd)
            return result, events, restored
        finally:
            chdir(original_cwd)

    def test_mfa_baseline_calls_results_and_failure_order(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory).resolve()
            for mode in ('ok', 'business'):
                with self.subTest(mode=mode):
                    old = self.mfa_case(self.old_recover, work, mode)
                    new = self.mfa_case(recover, work, mode)
                    self.assertEqual(old, new)
                    self.assertEqual(new[2], (True, True, True, True))

    def test_mfa_entry_failure_now_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory).resolve()
            old = self.mfa_case(self.old_recover, work, 'entry')
            new = self.mfa_case(recover, work, 'entry')
            self.assertEqual(old[0], new[0])
            self.assertEqual(old[2], (False, False, False, True))
            self.assertEqual(new[2], (True, True, True, True))

    def test_mfa_cleanup_failure_does_not_skip_other_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, restored = self.mfa_case(recover, Path(directory).resolve(), 'cleanup')
            self.assertEqual(result, ('raised', 'OSError', 'cleanup'))
            self.assertEqual(restored, (False, True, True, True))

    def test_other_function_bodies_unchanged(self):
        for _, module, source, target in self.pairs:
            def definitions(text):
                return {node.name: ast.dump(node, include_attributes=False)
                        for node in ast.parse(text).body
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != target}
            self.assertEqual(definitions(source), definitions(Path(module.__file__).read_bytes()))


if __name__ == '__main__':
    unittest.main()
