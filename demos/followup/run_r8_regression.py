"""R1-R8 corrected isolated CPU regression. Does not load real model/datasets."""
import hashlib,json,subprocess,sys,re
from pathlib import Path
root=Path(__file__).resolve().parent
results={}
for name in ('r1','r2','r3','r4','r5','r6','r7'):
    d=root/name
    if name in ('r1','r2','r3','r4'):
        for line in (d/'SHA256SUMS').read_text().splitlines():
            sha,file=line.split(maxsplit=1)
            if hashlib.sha256((d/file.lstrip('*')).read_bytes()).hexdigest()!=sha: raise RuntimeError(f'{name}/{file}: hash mismatch')
    files=[p.name for p in sorted(d.glob('*.py'))]
    subprocess.run([sys.executable,'-m','py_compile',*files],cwd=d,check=True)
    p=subprocess.run([sys.executable,'-m','unittest','discover','-v'],cwd=d,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(name,p.stdout,flush=True); p.check_returncode()
    match=re.search(r'Ran (\d+) tests?',p.stdout)
    if not match: raise RuntimeError('missing test count')
    results[name]={'status':'PASS_CPU_TESTS_ONLY','compiled':len(files),'tests':int(match[1])}
print(json.dumps(results,sort_keys=True))
