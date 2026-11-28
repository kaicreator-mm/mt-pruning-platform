"""R23 bounded continuation receipt/selection support; no model execution.

Validate persisted actual captures against the frozen plan and exact identities.
One file per additional capture with a digest and immutable order. Selection
requires all 24 verified rows and uses fit-only selection with holdout rejection.
Default self-test or explicit read-only verification; no inference or writes.
"""
import argparse, hashlib, importlib.util, json, math, pathlib
P = pathlib.Path
ROOT = P.home()/".cache/mt-pruning"
TEACHER = "0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
ENGINE = "3212303e1e02994a5124db8c85195b91325c7a7c"
SPARSE = "1c38f6e928ff653fbbef584b298ff65e194900fe"
PLAN = "3c92bd0a944637ee7eb8d36d271417396b313026"
SELECTOR = "f343ec873d6ab3801918cb20d795bc4d6a7c5155"
MASKS = {
    "original96":"23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7",
    "guarded96":"4589d613512ac0a07f02be500e8b2fabe3f4beb78ddd171e14fd26cd8078960c",
    "balanced":"4410fdd5dec984f19c766ee0d133035c910b12770c8c09de4756962bdaa9d631",
    "robust":"250ce92b01077c4d65ee77c92a9db5cabe9ec6d2907ea7525e5789ea7eb57d1b",
    "myanmar2":"d6c365d360bc47052fb64d3811b303cea0309b7670d734e16b6de7f63b552cc0"
}
def need(v, msg):
    if not v: raise ValueError(msg)
def sha(b): return hashlib.sha256(b).hexdigest()
def blob(b): return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def digest(s):
    return isinstance(s,str) and len(s)==64 and all(x in "0123456789abcdef" for x in s)
def strict(raw):
    def reject(s): raise ValueError("nonfinite JSON "+s)
    def pairs(ps):
        out={}
        for k,v in ps:
            need(k not in out,"duplicate key");out[k]=v
        return out
    return json.loads(raw,parse_constant=reject,object_pairs_hook=pairs)
def read_regular(p):
    need(p.is_file() and not p.is_symlink(),"regular receipt "+str(p))
    return p.read_bytes()
STRICT="STRICT_HISTORICAL_GREEDY"
FORCED="FIXED_HISTORICAL_PREFIX_LOCAL_PROXY"
def prefix_check(prefix,logits,x,policy):
    need(all(digest(z["sha256"]) and type(z["next"]) is int and z["next"]>=0 for z in logits),"logit hash/token")
    actual=[z["next"] for z in logits[:8]]
    if policy==STRICT:
        need("prefix_policy" not in x and actual==prefix,"historical greedy prefix")
    elif policy==FORCED:
        need(x["prefix_policy"]==FORCED and x["consumed_prefix"]==prefix and x["threads"]==8,"fixed historical prefix")
        need([z["input_token"] for z in logits[1:]]==prefix,"actual supplied tokens")
        mismatches=[{"step":j,"expected":t,"actual":a} for j,(t,a) in enumerate(zip(prefix,actual),1) if t!=a]
        need(x["historical_argmax_mismatches"]==mismatches and x["historical_greedy_match"]==(not mismatches),"explicit historical mismatch")
    else:raise ValueError("unknown prefix policy")
def validate(rec,x,admission_sha,policy=STRICT):
    need(x["status"]=="LOCAL_OUTPUT_PROXY_ONLY" and x["id"]==rec["id"] and x["line"]==rec["line"] and x["direction"]==rec["direction"],"sample identity")
    need(rec["id"].startswith("calibration:") and x["teacher_digest"]==TEACHER and x["engine_head"]==ENGINE and x["r20_exact_head"]==ENGINE and x["sparse_witness_blob"]==SPARSE,"source identity")
    need(digest(admission_sha) and x["admission_sha256"]==admission_sha and x["candidate_masks"]==MASKS,"admission/mask binding")
    need(x["logit_equivalence"] is True and len(x["teacher_logits"])==9,"teacher logits")
    logits=x["teacher_logits"];prefix=rec["teacher_prefix_tokens"]
    need([(z["phase"],z["step"]) for z in logits]==[("prefill",0)]+[("decode",j) for j in range(1,9)],"logit order")
    prefix_check(prefix,logits,x,policy)
    obs=x["observations"];counts=x["expert_compute_counts"]
    need(len(obs)==len(counts)==141,"observation coverage")
    expected={(l,p,j) for l in range(1,48) for p,j in (("prefill",0),("decode",1),("decode",8))}
    seen=set();nominal=actual=0
    for o,c in zip(obs,counts):
        key=(o["layer"],o["phase"],o["step"]);need(key in expected and key not in seen,"observation identity");seen.add(key)
        n=2 if o["phase"]=="prefill" else 1
        need(o["row_count"]==n and (c["layer"],c["phase"])==key[:2] and c["nominal"]==128*n and type(c["actual"]) is int and 8*n<=c["actual"]<=128*n and c["actual"]%n==0,"compute accounting")
        nominal+=c["nominal"];actual+=c["actual"]
        need(all(math.isfinite(o[k]) and 0<=o[k]<=1e-4 for k in ("routed_drift","combined_drift")),"observer drift")
        need(len(o["energy"])==128 and all(math.isfinite(v) and v>=0 for v in o["energy"]),"energy")
        need(set(o["losses"])==set(MASKS),"loss inventory")
        for v in o["losses"].values():
            need(set(v)=={"routed_error","combined_error"} and all(math.isfinite(z) and z>=0 for z in v.values()),"loss values")
    need(seen==expected and nominal==x["token_expert_evaluations"]==x["nominal_exhaustive_evaluations"]==24064 and actual==x["sparse_counterfactual_evaluations"],"total accounting")
    oracle=x["exhaustive_oracle"]
    need([(z["layer"],z["phase"],z["rows"]) for z in oracle]==[(1,"prefill",2),(1,"decode",1),(1,"decode",1)],"first-layer exhaustive coverage")
    need(all(z["candidate_comparisons"]==5 and z["exhaustive_extra_evaluations"]==z["rows"]*128 for z in oracle) and x["extra_oracle_evaluations"]==512,"oracle accounting")
    need(x["max_combined_drift"]==max(o["combined_drift"] for o in obs) and x["max_routed_drift"]==max(o["routed_drift"] for o in obs),"drift summary")
    return {"id":rec["id"],"observations":len(obs),"nominal":nominal,"sparse":actual,"oracle":512}
def frozen(plan_path):
    raw=read_regular(plan_path);need(blob(raw)==PLAN,"frozen plan")
    p=strict(raw)["plan"];need(set(p)=={"fit","holdout"},"plan subsets")
    allrows=p["fit"]+p["holdout"]
    need(len(allrows)==len({r["id"] for r in allrows})==len({r["line"] for r in allrows})==24,"plan isolation")
    return p,allrows
def first_receipt(path,admission_path,expected_sha,rec):
    raw=read_regular(path);need(digest(expected_sha) and sha(raw)==expected_sha,"first capture digest")
    ad=read_regular(admission_path);a=strict(ad)
    need(a["status"]=="R21_R23_PILOT_ADMITTED" and a["sample_id"]==rec["id"] and a["candidate_masks"]==MASKS and a["teacher_digest"]==TEACHER and a["r20_exact_head"]==ENGINE and a["sparse_witness_blob"]==SPARSE and a["admission_comment"]==7213914526,"first admission")
    lines=raw.splitlines();need(len(lines)==2 and raw.endswith(b"\n"),"first complete two-line journal")
    pre,x=map(strict,lines)
    need(pre["status"]=="RESOURCE_PREFLIGHT_ONLY" and not pre["blockers"] and pre["sample_id"]==rec["id"] and pre["candidate_masks"]==MASKS,"first preflight")
    return x,validate(rec,x,sha(ad))
def load_batch(directory,allrows,start=1):
    """Read only contiguous completed records; never infer success from partials."""
    need(directory.is_dir() and not directory.is_symlink(),"batch directory")
    names=sorted(directory.iterdir(),key=lambda p:p.name)
    out={}
    for i,p in enumerate(names,start):
        need(i<len(allrows) and p.name==f"{i:02d}.json","unexpected/gapped/partial file")
        envelope=strict(read_regular(p));r=allrows[i];payload=envelope["capture"]
        need(envelope["index"]==i and envelope["id"]==r["id"],"envelope identity")
        a=envelope["admission"];adraw=json.dumps(a,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        need(a["status"]=="R23_BOUNDED_CALIBRATION_ADMISSION" and a["teacher_digest"]==TEACHER and a["candidate_masks"]==MASKS and a["engine_head"]==ENGINE and a["plan_blob"]==PLAN and a["first"]<=i<a["stop"]<=24,"batch admission")
        need(sha(adraw)==envelope["admission_sha256"],"batch admission digest")
        canon=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        need(sha(canon)==envelope["capture_sha256"],"capture digest")
        policy=a.get("prefix_policy",STRICT)
        if policy==FORCED:
            need(a["threads"]==8 and a["diagnostic_job"]=="job_01M8E3ZB2G00HAW2QFBG6FSQ23" and a["first"]>=10 and a["sample_ids"]==[z["id"] for z in allrows[a["first"]:a["stop"]]],"forced continuation scope")
        validate(r,payload,envelope["admission_sha256"],policy)
        out[r["id"]]=payload
    return out
def selftest():
    import copy,tempfile
    rec={"id":"calibration:zh-my:synthetic","direction":"zh-my","line":0,"teacher_prefix_tokens":list(range(8))}
    obs=[];counts=[]
    for p,j,n in (("prefill",0,2),("decode",1,1),("decode",8,1)):
        for l in range(1,48):
            obs.append(dict(layer=l,phase=p,step=j,row_count=n,routed_drift=0.,combined_drift=0.,energy=[0.]*128,losses={k:dict(routed_error=.1,combined_error=.1) for k in MASKS}))
            counts.append(dict(layer=l,phase=p,nominal=n*128,actual=n*8))
    ad="1"*64
    x=dict(status="LOCAL_OUTPUT_PROXY_ONLY",id=rec["id"],line=rec["line"],direction=rec["direction"],teacher_digest=TEACHER,engine_head=ENGINE,r20_exact_head=ENGINE,sparse_witness_blob=SPARSE,admission_sha256=ad,candidate_masks=MASKS,logit_equivalence=True,teacher_logits=[dict(phase="prefill" if i==0 else "decode",step=i,next=i,sha256="2"*64) for i in range(9)],observations=obs,expert_compute_counts=counts,token_expert_evaluations=24064,nominal_exhaustive_evaluations=24064,sparse_counterfactual_evaluations=1504,exhaustive_oracle=[dict(layer=1,phase=p,rows=n,candidate_comparisons=5,exhaustive_extra_evaluations=n*128) for p,n in (("prefill",2),("decode",1),("decode",1))],extra_oracle_evaluations=512,max_combined_drift=0.,max_routed_drift=0.)
    need(validate(rec,x,ad)["observations"]==141,"valid synthetic")
    def rejects(fn):
        try:fn()
        except (ValueError,KeyError,TypeError):return
        raise AssertionError("accepted malformed evidence")
    for k,v in (("teacher_digest","0"*64),("admission_sha256","0"*64),("logit_equivalence",False),("sparse_counterfactual_evaluations",0),("extra_oracle_evaluations",0)):
        rejects(lambda:validate(rec,dict(x,**{k:v}),ad))
    for field,val in (("combined_drift",.1),("energy",[0.]),("step",9)):
        y=copy.deepcopy(x);y["observations"][0][field]=val;rejects(lambda:validate(rec,y,ad))
    rejects(lambda:strict(b'{"a":1,"a":2}'));rejects(lambda:strict(b'{"x":NaN}'))
    with tempfile.TemporaryDirectory() as d:
        p=P(d);(p/"02.json").write_text("{}")
        rejects(lambda:load_batch(p,[rec]*3))
    y=copy.deepcopy(x)
    y["teacher_logits"][6]["next"]=99
    for j,z in enumerate(y["teacher_logits"][1:]):z["input_token"]=j
    y.update(prefix_policy=FORCED,consumed_prefix=list(range(8)),threads=8,historical_argmax_mismatches=[{"step":7,"expected":6,"actual":99}],historical_greedy_match=False)
    need(validate(rec,y,ad,FORCED)["observations"]==141,"forced witness")
    rejects(lambda:validate(rec,y,ad))
    rejects(lambda:validate(rec,dict(y,historical_argmax_mismatches=[]),ad,FORCED))
    rejects(lambda:validate(rec,dict(y,consumed_prefix=[0]*8),ad,FORCED))
    rejects(lambda:validate(rec,dict(y,historical_greedy_match=True),ad,FORCED))
    rejects(lambda:validate(rec,dict(y,logit_equivalence=False),ad,FORCED))
    z=copy.deepcopy(y);z["teacher_logits"][7]["input_token"]=99
    rejects(lambda:validate(rec,z,ad,FORCED))
    print(json.dumps({"status":"PASS_SYNTHETIC_RECEIPT_VALIDATION_V2","positive":2,"negative":17,"real_capture":"NOT_RUN"}))
def main():
    a=argparse.ArgumentParser();a.add_argument("--selftest",action="store_true")
    a.add_argument("--plan",type=P);a.add_argument("--first",type=P);a.add_argument("--first-admission",type=P);a.add_argument("--first-sha");a.add_argument("--batch-dir",type=P)
    a.add_argument("--select",action="store_true");a.add_argument("--selector",type=P);a.add_argument("--original",type=P);a.add_argument("--guarded",type=P);a.add_argument("--p04",type=P)
    q=a.parse_args()
    if q.selftest:selftest();return
    need(all((q.plan,q.first,q.first_admission,q.first_sha,q.batch_dir)),"explicit receipt paths")
    p,rows=frozen(q.plan);x,report=first_receipt(q.first,q.first_admission,q.first_sha,rows[0]);rest=load_batch(q.batch_dir,rows);captures={rows[0]["id"]:x,**rest}
    if q.select:
        need(len(captures)==24 and all((q.selector,q.original,q.guarded,q.p04)),"full capture coverage and selector inputs")
        raw=read_regular(q.selector);need(blob(raw)==SELECTOR,"selector blob")
        sp=importlib.util.spec_from_file_location("selector",q.selector);s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
        masks=s.masks_from(q.original,q.guarded,q.p04)
        result=s.select(s.aggregate(p,captures,masks),masks)
        result["prefix_policy"]="FIXED_HISTORICAL_PREFIX_LOCAL_PROXY; legacy rows exactly matched"
        result["historical_argmax_mismatches"]={i:x["historical_argmax_mismatches"] for i,x in captures.items() if x.get("historical_argmax_mismatches")}
        print(json.dumps(result,allow_nan=False))
    else:print(json.dumps({"status":"VERIFIED_RECEIPTS_NOT_QUALITY","completed":len(captures),"total":24,"next":rows[len(captures)]["id"] if len(captures)<24 else None,"first":report}))
if __name__=="__main__":main()
