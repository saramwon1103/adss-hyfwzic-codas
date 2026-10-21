"""Small provenance manifest; no host paths, tokens or local environment secrets."""
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path


def write_manifest(output, root, mode, files, parameters):
    digest = {str(Path(p).relative_to(root)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
              for p in sorted(files)}
    report = {'schema_version':'1.0','mode':mode,'parameters':parameters,
              'python':platform.python_version(),
              'packages':{k:version(k) for k in ['numpy','pandas','streamlit','pytest','matplotlib']},
              'input_sha256':digest}
    (output/'run_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
