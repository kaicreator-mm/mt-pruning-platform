"""Fit-only descriptive attribution of rejected stable96. No fresh data or new mask.
Receipts replay exact decision; single-layer reversions are algebraic local estimates,
not measured hybrid-mask or propagated inference. Never admit a model here.
"""
import argparse, ast, hashlib, json, math, pathlib, types
P=pathlib.Path
DEC='7821abc55b356520bdd02e4b60f32057a95d204b8cc723b894e2fdce3d9d100d'
CAND='488a27afd5e75019cb6cc7355b9f847bbca515fc7570da081020a23b3e6213ef'
PLAN='a1047d1a7c5624ed0e3a868bd5d76eef4c754915'
PROXY='248737901db40b50efb0a26a9d5ef76d1928538b'
ORIG='d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed'
LOGS=('aa9d1dd75249a084c4a8a4b142809493393bc0947658a0ea24a8e1490758e8bb','b72ce687ccef58628717de71d37ced00148dabb897feb0c91ac8f6c6659cf7fb')
AD=('8a690e490563011128eaa41b04525bbe55d18db2a91737e9c8aec5f71098f046','6ce037930c68da9a094c922e49fc2be77db6eb220a00438aad42216fda878123')
TEACHER='0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94'
ENGINE='3212303e1e02994a5124db8c85195b91325c7a7c'
DIRS=('zh-my','my-zh','zh-en','en-zh','en-my','my-en')
GROUPS=tuple(d+'/'+p for d in DIRS for p in ('prefill','decode'))
def need(v,m):
    if not v: raise ValueError(m)
def sha(b): return hashlib.sha256(b).hexdigest()
def blob(b): return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def canon(v): return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(p):
    need(p.is_file() and not p.is_symlink(),'regular file '+str(p));return p.read_bytes()
def strict(b):
    def pairs(xs):
        d={}
        for k,v in xs: need(k not in d,'duplicate key');d[k]=v
        return d
    return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
def pinned(p,h,git=False):
    b=read(p);need((blob if git else sha)(b)==h,'hash mismatch '+str(p));return b
def pure_proxy(raw):
    # Compile only the exact pinned pure replay functions and constants.
    # No fcntl, server imports, model load, lock monkeypatch, or runtime main.
    tree=ast.parse(raw)
    wanted={'need','sha','canon','checked_capture','decision','DIRS','GROUPS','NAMES'}
    nodes=[];seen=set()
    for n in tree.body:
        name=n.name if isinstance(n,ast.FunctionDef) else (n.targets[0].id if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) else None)
        if name in wanted:
            need(name not in seen,'duplicate pure definition');seen.add(name);nodes.append(n)
    need(seen==wanted,'pure proxy inventory')
    m=types.ModuleType('proxy_pure');m.__dict__.update(math=math,hashlib=hashlib,json=json)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned proxy pure replay>','exec'),m.__dict__)
    return m
def mean(xs):
    need(bool(xs),'nonempty group');return math.fsum(xs)/len(xs)
def analyze(rows,caps,layers,orig,stable):
    need(set(caps)=={r['id'] for r in rows} and all(sum(r['direction']==d for r in rows)>1 for d in DIRS),'complete fit groups')
    per={}; bysample={}
    for r in rows:
        x=caps[r['id']];v={}
        for o in x['observations']:
            k=(o['layer'],o['phase']);q=o['losses'];a=q['original96']['combined_error'];b=q['stable96']['combined_error'];need(all(math.isfinite(z) and z>=0 for z in (a,b)),'loss')
            v.setdefault(k,[]).append(b-a)
        need(set(v)=={(l,p) for l in layers for p in ('prefill','decode')} and all(len(z)==(1 if k[1]=='prefill' else 2) for k,z in v.items()),'coverage')
        bysample[r['id']]={k:mean(z) for k,z in v.items()}
    for l in layers:
        ds={};folds={}
        for g in GROUPS:
            d,p=g.split('/');rs=[r for r in rows if r['direction']==d];vs=[bysample[r['id']][l,p] for r in rs];ds[g]=mean(vs)
            f=[mean([bysample[r['id']][l,p] for r in rs if r['line']!=line]) for line in sorted({r['line'] for r in rs})]
            folds[g]=dict(lines=len(f),positive=sum(v>1e-12 for v in f),negative=sum(v< -1e-12 for v in f),min=min(f),max=max(f))
        changed=orig[str(l)]!=stable[str(l)]
        if not changed: need(all(abs(z)<=1e-12 for z in ds.values()),'unchanged layer discrepancy')
        per[str(l)]=dict(changed=changed,added=sorted(set(stable[str(l)])-set(orig[str(l)])),removed=sorted(set(orig[str(l)])-set(stable[str(l)])),delta=ds,leave_one_group_line_out=folds)
    total={g:mean([x['delta'][g] for x in per.values()]) for g in GROUPS}
    rank=sorted(per,key=lambda l:(-per[l]['delta']['zh-my/prefill'],int(l)))
    counter=[]
    for l in rank:
        if not per[l]['changed']:continue
        revised={g:total[g]-per[l]['delta'][g]/len(layers) for g in GROUPS}
        counter.append(dict(layer=int(l),local_delta_if_single_layer_reverted=revised))
    return dict(group_delta=total,layers=per,zh_my_prefill_rank=[int(l) for l in rank],single_layer_local_counterfactuals=counter)
def selftest():
    rows=[dict(id=d+str(i),direction=d,line=i) for d in DIRS for i in range(2)]
    def obs(l,p,step,b):return dict(layer=l,phase=p,step=step,losses=dict(original96=dict(combined_error=.25),stable96=dict(combined_error=b)))
    caps={r['id']:dict(observations=[obs(l,p,j,.5 if l==1 and p=='prefill' else .25) for l in (1,2) for p,j in (('prefill',0),('decode',1),('decode',8))]) for r in rows}
    x=analyze(rows,caps,[1,2],{'1':[0],'2':[0]},{'1':[1],'2':[0]})
    need(x['group_delta']['zh-my/prefill']==.125 and x['zh_my_prefill_rank']==[1,2] and x['single_layer_local_counterfactuals'][0]['local_delta_if_single_layer_reverted']['zh-my/prefill']==0,'oracle')
    for bad in (dict(caps,extra={}),{k:v for k,v in caps.items() if k!=rows[0]['id']}):
        try:analyze(rows,bad,[1,2],{'1':[0],'2':[0]},{'1':[1],'2':[0]})
        except ValueError:pass
        else:raise AssertionError('bad inventory accepted')
    try:analyze(rows,caps,[1,2],{'1':[0],'2':[0]},{'1':[0],'2':[0]})
    except ValueError:pass
    else:raise AssertionError('unchanged layer loss accepted')
    print(json.dumps(dict(status='PASS_SYNTHETIC_LAYER_ATTRIBUTION',positive=1,negative=3)))
def main():
    a=argparse.ArgumentParser();a.add_argument('--selftest',action='store_true');a.add_argument('--inputs',type=P);a.add_argument('--out',type=P);q=a.parse_args()
    if q.selftest:selftest();return
    need(q.inputs and q.out,'inputs');d=q.inputs
    plan=strict(pinned(d/'expanded.json',PLAN,True));rows=plan['fit'];need(len(rows)==48 and len({r['line'] for r in rows})==30 and all(sum(r['direction']==g for r in rows)==8 for g in DIRS),'fit scope')
    need(not ({r['line'] for r in rows}&{r['line'] for r in plan['holdout']}),'line isolation')
    original=strict(pinned(d/'original.json',ORIG,True))['candidates']['96']['kept'];c=strict(pinned(d/'candidate.json',CAND));stable=c['kept'];masks={'original96':original,'stable96':stable};layers=list(range(1,48))
    need(set(original)==set(stable)=={str(l) for l in layers} and all(len(v)==96 and v==sorted(set(v)) and all(type(i) is int and 0<=i<128 for i in v) for m in masks.values() for v in m.values()),'mask shape')
    need(c['fit_ids']==[r['id'] for r in rows] and sha(canon(stable))==c['mask_sha256'],'candidate scope')
    proxy=pure_proxy(pinned(d/'proxy.py',PROXY,True))
    need(sorted(p.name for p in (d/'fit').iterdir())==sorted([f'{i:02d}.json' for i in range(48)]+['decision.json']),'fit inventory')
    captures={};hashes=[]
    for b in range(2):
        events=[strict(x) for x in pinned(d/f'fit{b}.log',LOGS[b]).splitlines() if x.startswith(b'{')];need(read(d/f'fit{b}.stderr')==b'','stderr')
        ready=[x for x in events if x.get('status')=='TEACHER_READY'];need(len(ready)==1 and ready[0]['teacher']==TEACHER and ready[0]['admission_sha256']==AD[b],'teacher')
        ev=[x for x in events if x.get('status')=='STABLE96_PROXY_CAPTURED'];need(len(ev)==24,'batch count')
        for i,e in enumerate(ev,b*24):
            r=rows[i];need(e['index']==i and e['id']==r['id'] and e['stage']=='fit' and e['completed']==i+1,'ordered log')
            h=e['receipt_sha256'];x=strict(pinned(d/'fit'/f'{i:02d}.json',h));need(x['index']==i and x['stage']=='fit' and x['teacher_digest']==TEACHER and x['engine_head']==ENGINE and x['plan_blob']==PLAN and x['candidate_receipt_sha256']==CAND and x['admission_sha256']==AD[b] and x['extra_oracle_evaluations']==512,'capture provenance');proxy.checked_capture(x,r,masks,layers);captures[r['id']]=x;hashes.append(h)
    f=strict(pinned(d/'fit'/'decision.json',DEC));need(f==proxy.decision(rows,captures,masks,layers,'fit') and f['status']=='NO_ADMISSION_FROM_EXPANDED_FIT' and f['violations']==['zh-my/prefill'],'exact rejected decision')
    x=analyze(rows,captures,layers,original,stable)
    for g in GROUPS:need(abs(x['group_delta'][g]-(f['means']['stable96'][g]-f['means']['original96'][g]))<1e-12,'aggregate reconciliation')
    out=dict(status='FIT_LAYER_ATTRIBUTION_ONLY_NOT_NEW_MASK',samples=48,aligned_lines=30,decision_sha256=DEC,candidate_receipt_sha256=CAND,receipt_sha256=hashes,**x,limits='Descriptive fit-only local teacher-state attribution; single-layer counterfactuals are algebraic, not measured propagation, no fresh data or new candidate selection.')
    raw=canon(out)+b'\n'
    with q.out.open('xb') as h:h.write(raw)
    print(json.dumps(dict(status=out['status'],sha256=sha(raw),samples=48,top_zh_my_prefill=[dict(layer=l,delta=x['layers'][str(l)]['delta']['zh-my/prefill'],folds=x['layers'][str(l)]['leave_one_group_line_out']['zh-my/prefill']) for l in x['zh_my_prefill_rank'][:8]],group_delta=x['group_delta'])))
if __name__=='__main__':main()
