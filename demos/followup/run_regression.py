"""Run archived demo suites in isolated processes, without model downloads."""
import hashlib, json, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parent
results={}
for name in ('r1','r2','r3','r4','r5'):
    d=root/name
    if name!='r5':
        for line in (d/'SHA256SUMS').read_text().splitlines():
            sha,file=line.split(maxsplit=1)
            if hashlib.sha256((d/file.lstrip('*')).read_bytes()).hexdigest()!=sha: raise RuntimeError(f'{name}/{file}: hash mismatch')
    files=[p.name for p in sorted(d.glob('*.py'))]
    subprocess.run([sys.executable,'-m','py_compile',*files],cwd=d,check=True)
    p=subprocess.run([sys.executable,'-m','unittest','discover','-v'],cwd=d,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(name,p.stdout,flush=True); p.check_returncode()
    results[name]={'status':'PASS_CPU_TESTS_ONLY','compiled':len(files)}
print(json.dumps(results,sort_keys=True))
