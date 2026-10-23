import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import zipfile

repo = Path(__file__).resolve().parents[2]
sys.path.append(str(Path(sys.base_prefix) / 'lib/python3.12/site-packages'))
sys.path.append('/Users/joshuapurtell/GitHub/workspaces/containers-hardening-20261020/.venv-native312/lib/python3.12/site-packages')
import synth_containers
loaded = Path(synth_containers.__file__).resolve()
assert loaded.is_relative_to(repo / '.local/installed'), loaded
wheel = repo / 'artifacts/policy-revision-20261023/wheels/synth_containers-0.4.6.dev2026102302-py3-none-any.whl'
with zipfile.ZipFile(wheel) as archive:
    names = [n for n in archive.namelist() if n.startswith('synth_containers/') and n.endswith('.py')]
    for name in names:
        expected = archive.read(name)
        assert expected == (repo / 'src' / name).read_bytes(), name
        assert expected == (loaded.parent.parent / name).read_bytes(), name
receipt = {'schema':'policy-revision.installed-source.v1','source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'wheel':wheel.name,'sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'version':importlib.metadata.version('synth-containers'),'python_files_matched':len(names),'loaded_module':str(loaded)}
(repo / 'artifacts/policy-revision-20261023/installed-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
import pytest
files = ['test_policy_revision_protocol.py','test_policy_lifecycle.py','test_put_policy_identity.py','test_http_requests.py','test_cispo_http_adapter.py','test_isolated_advertised_policy.py','test_gold_runtime_bounds.py','test_rollout_seal_reference.py']
raise SystemExit(pytest.main(['-q',*[str(repo/'tests'/f) for f in files],'--basetemp='+str(repo/'artifacts/policy-revision-20261023/installed-pytest'),'-o','cache_dir='+str(repo/'artifacts/policy-revision-20261023/pytest-cache')]))
