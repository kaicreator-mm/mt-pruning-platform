"""Descriptive fit-only MoE functional profile map. No mask selection.
Tags label whole calibration sentences, not isolated token spans or expert functions.
Preserve per-sample evidence and group denominators; unobserved REAP is null.
"""
import argparse,hashlib,json,math,pathlib,os,collections,importlib.util
P=pathlib.Path
ROOT=P.home()/".cache/mt-pruning"
PLAN="3c92bd0a944637ee7eb8d36d271417396b313026"
ANNOTATIONS="20ab0ee40141145bb0f9652d6bff651a45a76a3d"
MASKS="34a06701d47ce0e3ab888882a5b20c75745204d4"
CORPUS="9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
ENGINE="3212303e1e02994a5124db8c85195b91325c7a7c"
AUDIT_JOB="job_01M8E6KHBH00K1AWVEMREVR67S"
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def group(exposures,n):
    # vectors [count, weight, norm, weighted_norm, energy], support separate from total tokens
    tokens=sum(x["tokens"] for x in exposures)
    counts=[sum(x["count"][i] for x in exposures) for i in range(n)]
    ws=[math.fsum(x["weight"][i] for x in exposures) for i in range(n)]
    ns=[math.fsum(x["norm"][i] for x in exposures) for i in range(n)]
    weighted=[math.fsum(x["weighted_norm"][i] for x in exposures) for i in range(n)]
    energies=[math.fsum(x["energy"][i] for x in exposures) for i in range(n)]
    support=[sum(x["count"][i]>0 for x in exposures) for i in range(n)]
    return dict(tokens=tokens,samples=len(exposures),count=counts,weight=ws,norm=ns,weighted_norm=weighted,energy=energies,support=support,reap=[weighted[i]/counts[i] if counts[i] else None for i in range(n)])
def summarize(profiles,annotations,layers,n):
    ids=[x["id"] for x in profiles];need(len(ids)==len(set(ids)) and set(ids)==set(annotations),"annotation scope")
    tags=collections.defaultdict(list)
    for x in profiles:
        for tag in annotations[x["id"]]["tags"]:tags[tag].append(x["id"])
    meta={x["id"]:{"direction":x["direction"],"tags":annotations[x["id"]]["tags"],"reference_risk":annotations[x["id"]]["reference_risk"]} for x in profiles}
    out={};both=[];pref=[];dec=[];support=[];total=0
    for l in layers:
        groups={}
        for phase in ("prefill","decode"):
            allrows=[x["profile"][f"{l}/{phase}"] for x in profiles];g=group(allrows,n);groups[f"all/{phase}"]=g;total+=sum(g["count"])
            (pref if phase=="prefill" else dec).append(sum(c>0 for c in g["count"]))
            for d in sorted({x["direction"] for x in profiles}):
                groups[f"{d}/{phase}"]=group([x["profile"][f"{l}/{phase}"] for x in profiles if x["direction"]==d],n)
            # Whole-sentence tag association, not a controlled attribution.
            for tag,tagids in sorted(tags.items()):
                groups[f"tag:{tag}/{phase}"]=group([x["profile"][f"{l}/{phase}"] for x in profiles if x["id"] in tagids],n)
        a,b=groups["all/prefill"],groups["all/decode"]
        both.append(sum(x>0 or y>0 for x,y in zip(a["count"],b["count"])))
        support.extend(a["support"]);support.extend(b["support"])
        out[str(l)]=groups
    def rng(v):return dict(min=min(v),max=max(v),mean=math.fsum(v)/len(v),sum=sum(v))
    summary=dict(status="DESCRIPTIVE_FIT_MAP_NOT_SELECTION",samples=len(ids),expert_nodes=len(layers)*n,observed_either_phase=rng(both),observed_prefill=rng(pref),observed_decode=rng(dec),routed_assignments=total,tag_sample_support={t:len(v) for t,v in sorted(tags.items())},tag_direction_support={t:len({meta[i]["direction"] for i in v}) for t,v in sorted(tags.items())},limits="Whole-sentence overlapping labels, source/target and template confounding; no causality or unobserved-expert importance claims.")
    return dict(summary=summary,samples=meta,tag_samples=dict(tags),layers=out)
def selftest():
    a=dict(tokens=1,count=[1,0],weight=[1.,0.],norm=[2.,0.],weighted_norm=[2.,0.],energy=[4.,0.])
    g=group([a],2);need(g["reap"]==[2.,None] and g["support"]==[1,0],"unobserved")
    q=group([a,a],2);need(q["tokens"]==2 and q["reap"]==[2.,None] and q["support"]==[2,0],"additive")
    x={"id":"a","direction":"zh-my","profile":{"1/prefill":a,"1/decode":a}}
    out=summarize([x],{"a":{"tags":["number"],"reference_risk":None}},[1],2)
    need(out["summary"]["observed_either_phase"]["sum"]==1 and out["summary"]["routed_assignments"]==2,"summary")
    print(json.dumps(dict(status="PASS_SYNTHETIC_FUNCTIONAL_PROFILE_MAP",checks=3)))
def main():
    a=argparse.ArgumentParser();a.add_argument("--selftest",action="store_true");a.add_argument("--inputs",type=P);a.add_argument("--out",type=P);q=a.parse_args()
    if q.selftest:selftest();return
    need(q.inputs and q.out,"explicit paths");p=q.inputs
    for n,h in (("plan.json",PLAN),("annotations.json",ANNOTATIONS),("masks.py",MASKS)):need(blob((p/n).read_bytes())==h,"pin "+n)
    sp=importlib.util.spec_from_file_location("masks",p/"masks.py");m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    rows=json.loads((p/"plan.json").read_bytes())["plan"]["fit"]
    ann=json.loads((p/"annotations.json").read_bytes());annotations={x["id"]:x for x in ann["records"]}
    need(len(rows)==len(annotations)==12 and ann["corpus_sha256"]==CORPUS and ann["plan_blob"]==PLAN,"annotation scope")
    corpus=(ROOT/"run-d01b-r19/corpus/corpus.jsonl").read_bytes();need(sha(corpus)==CORPUS,"corpus pin")
    cs={x["id"]:x for x in map(json.loads,corpus.splitlines()) if x["split"]=="calibration" and x["id"] in annotations}
    hashes=json.loads((p/"hashes.json").read_bytes());need(len(hashes)==12,"hash scope")
    profiles=[];dest=ROOT/"r23-results/reap-full24"
    for i,r in enumerate(rows):
        v=cs[r["id"]];b=annotations[r["id"]]
        need(sha(v["source"].encode())==b["source_sha256"] and sha(v["reference"].encode())==b["reference_sha256"],"annotation source")
        raw=(dest/f"{i:02d}.json").read_bytes();need(sha(raw)==hashes[i],"profile receipt")
        x=json.loads(raw);need(x["id"]==r["id"] and x["subset"]=="fit" and x["teacher"]==TEACHER and x["engine_head"]==ENGINE and x["hook_calls"]==423 and len(x["profile"])==94,"identity")
        for z in x["profile"].values():m.check(z,128,8,x["scale"])
        profiles.append(x)
    result=summarize(profiles,annotations,list(range(1,48)),128)
    result.update(profile_receipt_hashes=hashes,profile_audit_job=AUDIT_JOB,teacher=TEACHER,engine=ENGINE,annotations_blob=ANNOTATIONS,plan_blob=PLAN)
    raw=canon(result)+b"\n"
    with q.out.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(result["summary"],map_sha256=sha(raw),map_bytes=len(raw))),flush=True)
if __name__=="__main__":main()
