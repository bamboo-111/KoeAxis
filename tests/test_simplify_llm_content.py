"""Bounded offline equivalence and captured-helper checks for batch 3."""
import ast
import itertools
import json
import shutil
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from host.contracts import ModuleError
from host.runtime import Host, execute_source
from modules.temporary.real_llm_stages import operations as llm
from modules.temporary.real_mimo import operations as mimo
from tests.test_opt54_content import API

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = 'experiments/references/program-content/v1.json'
TERMS = 'experiments/references/togeradi-terms/terms.json'


def observed(call):
    try:
        return ('returned', call())
    except Exception as exc:
        return ('raised', type(exc).__name__, str(exc))


class SimplifyLLMContentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = []
        for name, module in (('real_llm_stages', llm), ('real_mimo', mimo)):
            path = 'modules/temporary/' + name + '/operations.py'
            source = (ROOT / 'tests/fixtures/baselines/9dba954' / path).read_bytes()
            old = execute_source('_baseline_' + name, ROOT / path, source)
            cls.pairs.append((old, module, source))

    def test_wording_matrix_and_errors(self):
        texts = ('', '同じ', '一共12人', '一共13人', '没有12人', '「長い文章」', None)
        for old, new, _ in self.pairs:
            for left, right in itertools.product(texts, repeat=2):
                with self.subTest(module=new.__name__, left=left, right=right):
                    args = (left, right, '12 原文')
                    self.assertEqual(observed(lambda: old.wording_assessment(*args)),
                                     observed(lambda: new.wording_assessment(*args)))
            for args in (([], 'x', 'x'), (1, 'x', 'x'), ('x', 'x', 1)):
                self.assertEqual(observed(lambda: old.wording_assessment(*args)),
                                 observed(lambda: new.wording_assessment(*args)))

    def test_monkeypatch_hooks_and_exceptions(self):
        for old, new, _ in self.pairs:
            results = []
            for module in (old, new):
                with patch.object(module, '_numbers', return_value=['7']) as hook:
                    value = module.wording_assessment('left', 'right', 'source')
                    results.append((value, hook.call_args_list))
            self.assertEqual(results[0], results[1])
            hook_name = 'text_marks' if new is llm else '_language_shares'
            for module in (old, new):
                with patch.object(module, hook_name, side_effect=RuntimeError('hook failed')):
                    self.assertEqual(observed(lambda: module.wording_assessment('a', 'b', 'c')),
                                     ('raised', 'RuntimeError', 'hook failed'))

    def test_context_full_bytes_and_errors(self):
        for old, new, _ in self.pairs:
            name = '_content_context' if new is llm else '_content_glossary'
            for selected in ('', None, False, ' ', 'togeradi-v1', ' togeradi-v1 ', 'unknown'):
                self.assertEqual(observed(lambda: getattr(old, name)(API(selected))),
                                 observed(lambda: getattr(new, name)(API(selected))))
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                registry = root / REGISTRY
                registry.parent.mkdir(parents=True)
                registry.write_bytes((ROOT / REGISTRY).read_bytes())
                terms = root / TERMS
                terms.parent.mkdir(parents=True)
                payload = {'entries': [
                    {'term': term, 'ambiguous': ambiguous, 'force_replace': force}
                    for term, ambiguous, force in [('second', False, False), ('skip', True, False),
                                                   ('skip2', False, True), ('first', False, False),
                                                   ('second', False, False)]]}
                for raw in (json.dumps(payload).encode(), b'bad json', b'{"entries":[{}]}', None):
                    if raw is None:
                        terms.unlink()
                    else:
                        terms.write_bytes(raw)
                    with patch.object(old, 'ROOT', root), patch.object(new, 'ROOT', root):
                        self.assertEqual(observed(lambda: getattr(old, name)(API('togeradi-v1'))),
                                         observed(lambda: getattr(new, name)(API('togeradi-v1'))))

    def test_translate_full_arguments(self):
        old = self.pairs[0][0]
        captured = []
        modules = {'models': NS(WorkPaths=NS(from_workdir=lambda p: p)),
                   'stages': NS(run_translate_stage=lambda **kw: captured.append(kw))}
        for module in (old, llm):
            context = module._content_context(API('togeradi-v1'))
            module._run_translate(modules, Path('synthetic'), {'base': 'fake', 'key': 'fake-key'}, context)
        self.assertEqual(captured[0], captured[1])

    def test_unaffected_execution_and_capture_source(self):
        # All other existing top-level functions/classes (including production
        # guards, SDK capture, retries and comparison orchestration) are unchanged.
        for _, module, original in self.pairs:
            def definitions(source):
                tree = ast.parse(source)
                return {node.name: ast.dump(node, include_attributes=False)
                        for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            before = definitions(original)
            after = definitions(Path(module.__file__).read_bytes())
            changed = {'wording_assessment', '_content_context' if module is llm else '_content_glossary'}
            self.assertEqual(set(after) - set(before), {'_evaluation'})
            for name in before.keys() - changed:
                self.assertEqual(before[name], after[name], name)

    def test_helper_identity_and_captured_bytes(self):
        for module in (llm, mimo):
            module._evaluation.cache_clear()
            with patch.object(Path, 'read_bytes', return_value=b'tampered'):
                with self.assertRaises(ModuleError):
                    module._evaluation()
            loaded = module._evaluation()
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('must use captured bytes')):
                self.assertIs(module._evaluation(), loaded)
                module.wording_assessment('a', 'a', 'a')
            module._evaluation.cache_clear()

    def test_host_captured_entrypoints_load_helpers(self):
        registration = json.loads((ROOT / 'module-registry/real_chain_runner/plan-entry-312.json').read_text(encoding='utf-8'))
        output = 'runs/experiments/unit-host-simplify-content-' + uuid.uuid4().hex
        self.addCleanup(shutil.rmtree, ROOT / output, True)
        host = Host(ROOT, output, registration)
        found = set()
        for record in host.catalog['implementations']:
            for _, module, _ in self.pairs:
                if record['source'] != Path(module.__file__).relative_to(ROOT).as_posix():
                    continue
                instance = host.instances[record['content_digest']]
                namespace = instance.invoke.__func__.__globals__
                self.assertEqual(namespace['wording_assessment']('12人', '12人', '12人'),
                                 module.wording_assessment('12人', '12人', '12人'))
                self.assertIn(REGISTRY, [f['path'] for f in record['content_files']])
                found.add(record['source'])
        self.assertEqual(len(found), 2)
