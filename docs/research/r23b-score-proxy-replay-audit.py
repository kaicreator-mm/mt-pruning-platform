"""R23B independent receipt and fit-freeze replay before any candidate build.
No model loading, no mask selection from fresh results, no physical writes.
"""
import argparse,hashlib,json,math,pathlib,importlib.util,sys
P=pathlib.Path;ROOT=P.home()/".cache/mt-pruning"
PROXY_BLOB="c3db0bdadd070668fe23ea52b5290073958de5ec"
PLAN_BLOB="3c92bd0a944637ee7eb8d36d271417396b313026"
FRESH_BLOB="bc6d92ef87d23c954cd1c3f86527a39b52b5cc18"
FIT_SHA="f2ab6eb0377932bbddb0335e6543534896f42e01a17b70cc6ae8672f30ff4d55"
MASK_SHA="7f0eca9be13b309e293ff4902c880c8f2af769120ebe2cf516bb166b7f074dec"
AD_SHA="bc467020765d4d3728a28c5f5c6b161be10b51f337006a41fc535f37c33aedc5"
def need(x,m):
    if not x:raise ValueError(m)
def sha(x):return hashlib.sha256(x).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(x):return hashlib.sha1(b"blob "+str(len(x)).encode()+b"\0"+x).hexdigest()
def read(p):
    need(p.is_file() and not p.is_symlink(),"regular file "+str(p));return p.read_bytes()
def strict(raw):
    def pairs(xs):
        d={}
        for k,v in xs:need(k not in d,"duplicate key");d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def audit_stage(root,logs,rows,stage,masks):
    hits=[x for x in logs if x.get("status")==("FIT_CAPTURED" if stage=="fit" else "FRESH_CAPTURED")]
    need(len(hits)==len(rows)==12,"stage coverage")
    out={};expected={n:sha(canon(m)) for n,m in masks.items()}
    for i,(r,h) in enumerate(zip(rows,hits)):
        need(h["completed"]==i+1 and h["id"]==r["id"],"ordered stage")
        raw=read(root/f"{stage}-{i:02d}.json");need(sha(raw)==h["receipt_sha256"],"receipt hash")
        x=strict(raw)
        need(x["status"]=="LOCAL_OUTPUT_PROXY_ONLY" and x["candidate_masks"]==expected and x["logit_equivalence"] is True and x["id"]==r["id"] and x["line"]==r["line"] and x["direction"]==r["direction"],"receipt binding")
        need(x["consumed_prefix"]==r["teacher_prefix_tokens"] and x["threads"]==8 and x["prefix_policy"]=="FIXED_HISTORICAL_PREFIX_LOCAL_PROXY","fixed prefix")
        z=x["teacher_logits"];need(len(z)==9 and [q["input_token"] for q in z[1:]]==r["teacher_prefix_tokens"] and [(q["phase"],q["step"]) for q in z]==[("prefill",0)]+[("decode",j) for j in range(1,9)],"actual input sequence")
        need(all(isinstance(q["sha256"],str) and len(q["sha256"])==64 and all(c in "0123456789abcdef" for c in q["sha256"]) and type(q["next"]) is int for q in z),"logit identities")
        mis=[{"step":j,"expected":t,"actual":q["next"]} for j,(t,q) in enumerate(zip(r["teacher_prefix_tokens"],z[:8]),1) if t!=q["next"]]
        need(x["historical_argmax_mismatches"]==mis and x["historical_greedy_match"]==(not mis),"mismatch ledger")
        need(x["extra_oracle_evaluations"]==512 and x["sparse_counterfactual_evaluations"]==h["sparse"],"oracle/evaluation accounting")
        out[r["id"]]=x
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument("--inputs",type=P,required=True);p.add_argument("--fit-only",action="store_true");q=p.parse_args()
    d=q.inputs
    for n,h in (("proxy.py",PROXY_BLOB),("plan.json",PLAN_BLOB),("fresh.json",FRESH_BLOB)):
        need(blob(read(d/n))==h,"source pin "+n)
    sp=importlib.util.spec_from_file_location("proxy",d/"proxy.py");m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    root=ROOT/"r23-results/score-proxy-v1";log=ROOT/"r23-results/score-proxy-v1.log"
    logs=[strict(l) for l in read(log).splitlines(keepends=True) if l.endswith(b"\n")]
    plan=strict(read(d/"plan.json"))["plan"];fit=plan["fit"];fresh=strict(read(d/"fresh.json"))["rows"];m.check_scope(fit,fresh)
    need(not ({r["line"] for r in plan["fit"]+plan["holdout"]}&{r["line"] for r in fresh}),"fresh isolation")
    ad=strict(read(root/"admission.json"));need(sha(canon(ad))==AD_SHA,"admission")
    inventory=strict(read(root/"inventory.json"));masks={n:v["kept"] for n,v in inventory["methods"].items()}
    need(set(masks)=={"original96","frequency","router_weight","activation_norm","reap","global_weighted","protective_max"} and all(sha(canon(z))==inventory["methods"][n]["sha256"] and set(z)=={str(i) for i in range(1,48)} and all(len(a)==96 and a==sorted(set(a)) and all(type(i) is int and 0<=i<128 for i in a) for a in z.values()) for n,z in masks.items()),"mask inventory")
    pre=[x for x in logs if x.get("status")=="RESOURCE_PREFLIGHT_ONLY"];need(len(pre)==1 and pre[0]["mask_hashes"]=={n:sha(canon(z)) for n,z in masks.items()},"mask log pin")
    fraw=read(root/"fit-frozen.json");f=strict(fraw);need(sha(fraw)==FIT_SHA and f["mask_sha256"]==MASK_SHA and sha(canon(f["kept"]))==MASK_SHA,"frozen identity")
    markers=[x for x in logs if x.get("status")=="FIT_FROZEN_NOT_ADMITTED"];need(len(markers)==1 and markers[0]["receipt_sha256"]==FIT_SHA and markers[0]["mask_sha256"]==MASK_SHA,"freeze marker")
    a=audit_stage(root,logs,fit,"fit",masks)
    replay=m.select_fit(m.aggregate(fit,a,tuple(masks),list(range(1,48))),masks)
    need(replay==f,"exact fit replay")
    print(json.dumps(dict(status="FIT_REPLAY_PASS_NOT_ADMISSION",samples=12,mask_sha256=MASK_SHA,fit_sha256=FIT_SHA,changes=len(f["changes"]))),flush=True)
    if q.fit_only:return
    need(read(ROOT/"r23-results/score-proxy-v1.stderr")==b"","stderr")
    ms={"original96":masks["original96"],"selected96":f["kept"]}
    a=audit_stage(root,logs,fresh,"fresh",ms)
    out=m.reject_holdout(m.aggregate(fresh,a,tuple(ms),list(range(1,48))),f)
    raw=read(root/"decision.json");need(strict(raw)==out,"exact fresh decision")
    hit=[z for z in logs if z.get("status")==out["status"]];need(len(hit)==1 and hit[0]["receipt_sha256"]==sha(raw) and {k:v for k,v in hit[0].items() if k!="receipt_sha256"}==out,"decision log")
    print(json.dumps(dict(status="FULL_SCORE_PROXY_REPLAY_PASS_NOT_QUALITY",decision=out["status"],decision_sha256=sha(raw),candidate_sha256=MASK_SHA if out["candidate"] else None,holdout_primary_relative_improvement=out["holdout_primary_relative_improvement"],violations=out["violations"],overall_guard=out["overall_guard"])),flush=True)
if __name__=="__main__":main()
