"""Host entry: synthetic diagnostic by default, explicit JSON request optionally."""
import importlib.util
import json
from pathlib import Path
from host.contracts import ModuleError

ROOT = Path(__file__).resolve().parents[3]


def evaluator():
    spec = importlib.util.spec_from_file_location('_verification_gold54', Path(__file__).with_name('evaluate.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Implementation:
    def __init__(self, settings):
        pass

    def invoke(self, operation, arguments, api):
        if operation != 'run' or arguments['plan_id'] != 'GOLD54-EVALUATION-001':
            raise ModuleError('InvalidPlan', 'unexpected operation or plan')
        ev = evaluator()
        output = ROOT / api.snapshot()['environment']['output']
        request_path = api.environment('GOLD54_REQUEST', '') or ''
        if request_path:
            request = json.loads(Path(request_path).read_text(encoding='utf-8'))
            required = ('reference','split','normalized','ass')
            if not all(isinstance(request.get(k), str) and request[k] for k in required):
                raise ModuleError('InvalidInput', 'reference, split, normalized and ass required')
            if bool(request.get('tokens')) != bool(request.get('legacy_score')):
                raise ModuleError('InvalidInput', 'tokens and legacy_score must be supplied together')
            schema = request.get('schema', 'legacy')
            if schema not in ('legacy', 'gold54-human-reference-diagnostic/v2'):
                raise ModuleError('InvalidInput', 'unsupported diagnostic schema')
            evaluate_request = ev.evaluate if schema == 'legacy' else ev.evaluate_v2
            result = evaluate_request(*(request[k] for k in required), request.get('tokens'), request.get('legacy_score'), request.get('legacy_root'))
            case = {'id':'explicit_evaluation','passed':result['reference']['all'] > 0}
            report = {'mode':'explicit', 'result':result, 'case_count':1, 'passed':int(case['passed']), 'failed':int(not case['passed'])}
        else:
            ref = {'i':1,'s':0,'e':1000,'speaker':'理名','note':'','ja':'こんにちは','empty':False,'question':False,'multi':False}
            candidate = {'i':1,'s':0,'e':1100,'speaker':'理名','ja':'こんにちは'}
            text = ev.content([ref], [candidate]); timing = ev.timings([ref], [candidate]); speaker = ev.speaker_time([ref], [candidate])
            checks = [{'id':'content','passed':text['edits']==0 and text['reference_rows']==1}, {'id':'display_end','passed':timing['end']['median_signed_ms']==100}, {'id':'speaker','passed':speaker['durations_ms']['same']==1000}]
            report = {'mode':'synthetic','checks':checks,'case_count':len(checks),'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks)}
        evidence = output / 'evidence'; evidence.mkdir(parents=True,exist_ok=True)
        for name, value in [('report.json',report),('cases.json',report),('criteria.json',{'offline':True,'human_quality_passed':False})]:
            with (evidence/name).open('x',encoding='utf-8',newline='\n') as f:
                json.dump(value,f,ensure_ascii=False,indent=2); f.write('\n')
        if report['failed']:
            raise ModuleError('GoldEvaluationFailed', 'empty reference or synthetic check failure')
        return {'report':report}, []
