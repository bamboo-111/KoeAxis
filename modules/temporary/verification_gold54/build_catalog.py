"""Generate this candidate's contracts and content-addressed catalog."""
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from host.contracts import digest

version = '0.1.0-gold54'
profile_path = 'module-registry/verification_gold54/contracts-gold54.json'
catalog_path = ROOT / 'module-registry/verification_gold54/catalog-gold54.json'
source = 'modules/temporary/verification_gold54/operations.py'
files = ['host/__main__.py','host/contracts.py','host/runtime.py',profile_path,source,'modules/temporary/verification_gold54/evaluate.py','experiments/specifications/gold54-evaluation.json']
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')
def sha(path): return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
base = json.loads((ROOT/'module-registry/real_align_recovery/contracts-real-align-recovery.json').read_text(encoding='utf-8'))
operation = {'input':{'object':{'plan_id':{'const':'GOLD54-EVALUATION-001'}}},'output':{'object':{'report':'json'}},'calls':[],'files':{},'host_capabilities':['snapshot'],'resource_handles':[],'environment':['GOLD54_REQUEST']}
profile = {'profile':'verification-gold54-v1','protocol_version':'0.1.0-draft','interface_version':version,'schema_dialect':'closed-object-v1','types':base['types'],'modules':{'verification.gold54':{'operations':{'run':operation}}},'open':[]}
write(ROOT/profile_path, profile)
rec = {'module_id':'verification.gold54','implementation_id':'verification-gold54','version':version,'admission':'temporary','source':source,'content_files':[{'path':p,'sha256':sha(p)} for p in files],'settings':{}}
rec['content_digest'] = digest(rec)
write(catalog_path, {'profile':'verification-gold54-v1','admission':'temporary','exemptions':[],'bootstrap':[[rec['implementation_id'],version]],'trusted_bootstrap':{'path':'host/__init__.py','sha256':sha('host/__init__.py'),'scope':'trusted interpreter entry'},'implementations':[rec]})
