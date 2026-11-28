"""R23B full-routed-token profile, fixed calibration trajectories only.
Capture exact grouped expert contributions already computed by pinned Engine;
no re-execution of expert GEMMs. Compare every observed teacher logit to R23A.
One immutable per-sample profile. No mask selection or validation/test fitting.
"""
import argparse,fcntl,hashlib,inspect,json,math,os,pathlib,time,types
P=pathlib.Path
ROOT=P.home()/".cache/mt-pruning"
BATCH="9711b024631a5d8e0a468f02aaea1b993f268b02"
ENGINE_BLOB="2af43ca1710b029fdde0d1135ea9879fe8defe56"
VERIFIER_BLOB="e56b25b3fd92ccbd91455628606925a02af49450"
FIRST_SHA="6223518953b4de74f267f8ea7913d449fd5f6be38fef80473c880d73e61011ae"
SELECT_SHA="658227f9abc4ccdca6c605234f4817385694a5157b69ace2aced2bbb19d2e6b8"
REAP_BLOB="b223d05813625d597c3a598f8373931324bc089f"
DIAGNOSTIC="job_01M8E6545J00B3S79K4YP869FJ"
LEGACY_ADMISSION="aa91a0c2ba3fd08fde8059483d57b52fc5cf6082e80e3590a6d112e82414ed0f"
LEGACY_RECEIPTS=["f9275fafa90697d01c27a624cdbe81b05fb98029270d1dd096d7d6f9f4c2a3b4","34d3543784d0aea505ca89999d6fc34c3e740abdb9d2c9db9761b398dedd30f7","6b9ac5d95e57f3edcf7520df45133cd763154cc44bbd62139c1bf267cfc3a566","8823ea6d19eeadcce6e0bbca68894b2bc3715d57e308058bb6581b651465fe71","68d241b051802d676e633c767be2713064d3604eecc5dbddf491be59f8f618bb","3777bfff2c4486cedccd7b7f8798a9a7a6b1daf7df9c5559ba8644ca5677b926","0e9e79acc12838a25a8ec60615a8c5705e495fac435c606b6723a4b354d9b11c","241f409dc6ca0b2bcad553828d558e00200702d6ad8f32a9b793dc1ef5f20505","ba840bef8739f67549a9d2b61f45f994b7e440a240070304dcaf932c0d751bb7","cc6f351fbc5a49e765dc7ad0ca12bed0d939bc5edcd868d3c39d8aab9c367507"]
def normalized(xs):
    return [{k:v for k,v in z.items() if k!="input_token"} for z in xs]
def differences(old,new):
    need(len(old)==len(new)==9,"logit inventory")
    need([(z["phase"],z["step"]) for z in old]==[(z["phase"],z["step"]) for z in new],"logit order")
    return [{"step":a["step"],"old_sha":a["sha256"],"new_sha":b["sha256"],"old_next":a["next"],"new_next":b["next"]} for a,b in zip(old,new) if (a["sha256"],a["next"])!=(b["sha256"],b["next"])]
def legacy_prefix(dest,rows,old,v):
    need(dest.is_dir() and not dest.is_symlink(),"legacy directory")
    need(sorted(p.name for p in dest.iterdir())==[f"{i:02d}.json" for i in range(10)],"legacy inventory")
    for i,h in enumerate(LEGACY_RECEIPTS):
        raw=v.read_regular(dest/f"{i:02d}.json");need(sha(raw)==h,"legacy profile hash")
        x=v.strict(raw);r=rows[i];need(x["index"]==i and x["id"]==r["id"] and x["teacher"]==v.TEACHER and x["engine_head"]==v.ENGINE and x["admission_sha256"]==LEGACY_ADMISSION and x["threads"]==8 and x["subset"]=="fit","legacy identity")
        need(x["matched_r23a_logits"] is True and normalized(x["teacher_logits"])==normalized(old[r["id"]]["teacher_logits"]) and x["hook_calls"]==423 and len(x["profile"])==94,"legacy logits/inventory")
    return 10
def need(x,m):
    if not x:raise ValueError(m)
def sha(x):return hashlib.sha256(x).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def stats(torch,ids,weights,outputs,n,k,scale):
    need(ids.ndim==2 and ids.shape==weights.shape and ids.shape[1]==k,"route shape")
    need(outputs.ndim==2 and outputs.shape[0]==ids.numel(),"grouped outputs shape")
    need(bool(torch.isfinite(weights).all() and torch.isfinite(outputs).all() and (weights>=0).all()),"finite weights/output")
    need(bool((ids>=0).all() and (ids<n).all()) and bool((torch.sort(ids,dim=1).values.diff(dim=1)>0).all()),"expert ids")
    need(bool(torch.allclose(weights.sum(dim=1),torch.full_like(weights[:,0],scale),rtol=1e-5,atol=1e-7)),"route scale")
    flat=ids.reshape(-1);order=torch.argsort(flat,stable=True);idx=flat[order]
    g=weights.reshape(-1)[order].double();norm=torch.linalg.vector_norm(outputs.double(),dim=-1)
    count=torch.bincount(idx,minlength=n)
    vals=[g,norm,g*norm,(g*norm)**2]
    sums=[torch.zeros(n,dtype=torch.float64).scatter_add_(0,idx,v).tolist() for v in vals]
    need(int(count.sum())==ids.numel(),"count accounting")
    return dict(tokens=ids.shape[0],count=count.tolist(),weight=sums[0],norm=sums[1],weighted_norm=sums[2],energy=sums[3])
def instrument(module,hook,expected_blob):
    # Only the exact pinned Python source may be adapted, at one observation point.
    original=module.expert_combine;raw=P(inspect.getfile(original)).read_bytes()
    need(hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==expected_blob,"engine source")
    src=inspect.getsource(original)
    marker="    weighted = contribution * weights.reshape(-1)[order][:, None]"
    need(src.count(marker)==1,"insertion point")
    ns=dict(original.__globals__);ns["_profile_hook"]=hook
    exec(compile(src.replace(marker,"    _profile_hook(top, weights, contribution)\n"+marker),"<pinned-expert-combine-observer>","exec"),ns)
    return original,ns["expert_combine"]
def add(a,b):
    if a is None:return b
    return {k:(a[k]+b[k] if k=="tokens" else [x+y for x,y in zip(a[k],b[k])]) for k in a}
def capture(c,b,engine,rec,prompt,old,blob=ENGINE_BLOB):
    module=c.engine_module;torch=c.torch;n=engine.ckpt.num_experts;k=engine.top_k;scale=engine.router_scaling
    profiles={};state={};visited=set();orig_moe=engine._moe
    need("_moe" not in engine.__dict__ and engine.ckpt.kept is None,"unmasked original")
    def hook(ids,g,outputs):
        layer=state["layer"];phase=state["phase"];step=state["step"];key=(layer,phase,step)
        need(key not in visited,"duplicate hook");visited.add(key)
        r=stats(torch,ids,g,outputs,n,k,scale)
        need(r["tokens"]==(len(prompt) if phase=="prefill" else 1),"token accounting")
        profile_key=f"{layer}/{phase}";profiles[profile_key]=add(profiles.get(profile_key),r)
    plain=b.forced_forward(c,engine,prompt,rec["teacher_prefix_tokens"])
    original,modified=instrument(module,hook,blob)
    def moe(layer,x,phase):
        need(state["phase"]==phase,"phase");state["layer"]=layer.idx
        return orig_moe(layer,x,phase)
    try:
        module.expert_combine=modified;engine._moe=moe
        logits=b.forced_forward(c,engine,prompt,rec["teacher_prefix_tokens"],lambda p,j:state.update(phase=p,step=j))
    finally:
        module.expert_combine=original;del engine._moe
    expect={(l,p,j) for l in engine.ckpt.sparse_layers for p,j in [("prefill",0)]+[("decode",j) for j in range(1,9)]}
    need(visited==expect,"full layer/step coverage")
    again=b.forced_forward(c,engine,prompt,rec["teacher_prefix_tokens"])
    need(plain==logits==again,"same-process plain/observer/restored exact equality")
    cross=differences(old["teacher_logits"],logits)
    need(len(profiles)==2*len(engine.ckpt.sparse_layers),"profile inventory")
    for key,r in profiles.items():
        need(sum(r["count"])==r["tokens"]*k and r["tokens"]==(len(prompt) if key.endswith("/prefill") else 8),"profile count")
    return dict(id=rec["id"],direction=rec["direction"],teacher_logits=logits,profile=profiles,matched_r23a_logits=not cross,cross_process_r23a_differences=cross,plain_logits=plain,restored_logits=again,same_process_observer_equivalence=True,scale=scale,top_k=k,num_experts=n,hook_calls=len(visited))
def selftest(c,b,reap):
    torch=c.torch;torch.set_num_threads(1)
    ids=torch.tensor([[0,2],[1,0]]);g=torch.tensor([[.25,.75],[.5,.5]])
    # Actual expert-sorted order: token0 expert0, token1 expert0, token1 expert1, token0 expert2
    values=torch.tensor([[3.,4.],[0.,6.],[8.,0.],[0.,4.]])
    got=stats(torch,ids,g,values,4,2,1.)
    p=reap.Profile(4,2,1.);p.add([0,2],[.25,.75],[5.,4.]);p.add([1,0],[.5,.5],[8.,6.])
    rr=p.report()["experts"]
    for name,field in [("count","count"),("weight","router_weight_sum"),("norm","activation_norm_sum"),("weighted_norm","weighted_norm_sum"),("energy","weighted_energy")]:
        need(got[name]==[r[field] for r in rr],"scalar oracle "+name)
    def reject(fn):
        try:fn()
        except ValueError:return
        raise AssertionError("invalid accepted")
    reject(lambda:stats(torch,ids,g*.5,values,4,2,1.))
    reject(lambda:stats(torch,torch.tensor([[0,0],[1,0]]),g,values,4,2,1.))
    reject(lambda:stats(torch,ids,g,values*float("nan"),4,2,1.))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p=P(d);m=c.tiny_checkpoint(p/"tiny",p/"none",vocab_size=256,seed=19);e=c.engine_module.Engine(c.engine_module.Checkpoint(m),threads=1)
        prompt=[4,8,15,16,23,42];prefix=c.greedy_prefix(e,prompt,8)
        rec=dict(id="calibration:zh-my:synthetic",direction="zh-my",teacher_prefix_tokens=prefix)
        old={"teacher_logits":b.forced_forward(c,e,prompt,prefix)}
        x=capture(c,b,e,rec,prompt,old)
        need(x["hook_calls"]==18 and x["matched_r23a_logits"] and x["same_process_observer_equivalence"] and len(x["profile"])==4,"tiny profile")
        old["teacher_logits"][0]["sha256"]="0"*64
        y=capture(c,b,e,rec,prompt,old)
        need(not y["matched_r23a_logits"] and len(y["cross_process_r23a_differences"])==1 and y["same_process_observer_equivalence"],"cross-run mismatch explicitly distinct")
        old["teacher_logits"]=y["plain_logits"]
        need(b.forced_forward(c,e,prompt,prefix)==old["teacher_logits"],"restoration")
    print(json.dumps(dict(status="PASS_SYNTHETIC_REAP_RESUME_V2",scalar_oracles=5,negative=3,logit_checks=9,hook_calls=18,real_capture="NOT_RUN")),flush=True)
def main():
    p=argparse.ArgumentParser()
    for n in ("batch","verifier","driver","collector","reap","plan","original","guarded"):p.add_argument("--"+n,type=P,required=True)
    p.add_argument("--selftest",action="store_true");p.add_argument("--execute",action="store_true");p.add_argument("--admission",type=P)
    q=p.parse_args()
    import importlib.util,sys
    def load(path,pin,name):
        raw=path.read_bytes();need(hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==pin,"source "+name)
        sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
    b=load(q.batch,BATCH_BLOB,"batch");v=load(q.verifier,VERIFIER_BLOB,"verifier");d=load(q.driver,b.DRIVER,"driver");c=load(q.collector,b.COLLECTOR,"collector");reap=load(q.reap,REAP_BLOB,"reap");c.verify_code()
    if q.selftest:selftest(c,b,reap);return
    corpus=ROOT/"run-d01b-r19/corpus/corpus.jsonl";baseline=ROOT/"run-d01b-r19/baseline/records.jsonl"
    c.pilot_inputs(q.plan,corpus,baseline,q.original,q.guarded)
    plan,rows=v.frozen(q.plan)
    first,_=v.first_receipt(ROOT/"r23-results/sparse-first.jsonl",ROOT/"r23-admissions/sparse-first.json",FIRST_SHA,rows[0])
    old={rows[0]["id"]:first,**v.load_batch(ROOT/"r23-results/batch",rows)}
    need(len(old)==24,"complete R23A")
    selection=(ROOT/"r23-results/selection-24-v2.json").read_bytes()
    need(sha(selection)==SELECT_SHA and v.strict(selection)["status"]=="NO_ADMISSION_FROM_LOCAL_PROXY","R23A rejection pin")
    ctx={x["id"]:x for x in map(json.loads,corpus.read_bytes().splitlines()) if x["split"]=="calibration"}
    dest=ROOT/"r23-results/reap-full24"
    start=legacy_prefix(dest,rows,old,v)
    pre=d.preflight();print(json.dumps(dict(pre,completed=start,planned_samples=24,existing_replayed=24,scope="full prompt + eight fixed decode tokens")),flush=True)
    if not q.execute:return
    need(not pre["blockers"] and q.admission is not None,"execution blocked")
    raw=q.admission.read_bytes();a=v.strict(raw);need(raw==canon(a),"canonical admission")
    need(a["status"]=="R23B_PROFILE_ADMITTED" and a["teacher_digest"]==v.TEACHER and a["engine_head"]==v.ENGINE and a["plan_blob"]==v.PLAN and a["selection_sha256"]==SELECT_SHA and a["threads"]==8 and a["ids"]==[r["id"] for r in rows[start:]] and a["first"]==start and a["diagnostic_job"]==DIAGNOSTIC and a["legacy_receipts"]==LEGACY_RECEIPTS and a["reap_blob"]==REAP_BLOB and type(a["admission_comment"]) is int,"admission")
    with (ROOT/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not d.preflight()["blockers"] and legacy_prefix(dest,rows,old,v)==start,"exclusive resource/path")
        with (ROOT/"r23-admissions/reap-full24-resume-v2.json").open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
        torch=c.torch;torch.set_num_threads(8);torch.set_num_interop_threads(1)
        ident=d.checkpoint_identity(d.SOURCE,cache_dir=None,fresh=True,exact_inventory=True)
        need(ident["teacher"]==v.TEACHER and ident["num_experts"]==128 and not d.preflight()["blockers"],"teacher identity/resources")
        e=c.engine_module.Engine(c.engine_module.Checkpoint(d.SOURCE),threads=8)
        print(json.dumps(dict(status="TEACHER_READY",teacher=ident["teacher"],admission_sha256=sha(raw))),flush=True)
        for i in range(start,len(rows)):
            r=rows[i]
            need(not d.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"unexpected GPU")
            t=time.monotonic();x=capture(c,b,e,r,ctx[r["id"]]["prompt_ids"],old[r["id"]])
            x.update(index=i,teacher=ident["teacher"],engine_head=v.ENGINE,admission_sha256=sha(raw),plan_blob=v.PLAN,threads=8,seconds=time.monotonic()-t,subset="fit" if i<12 else "holdout")
            h=b.publish(dest/f"{i:02d}.json",x)
            print(json.dumps(dict(status="PROFILE_CAPTURED",id=r["id"],index=i,completed=i+1,receipt_sha256=h,hook_calls=x["hook_calls"],logit_checks=9,restored_logit_checks=9,cross_process_differences=len(x["cross_process_r23a_differences"]),seconds=x["seconds"])),flush=True)
        need(len(list(dest.iterdir()))==24,"profile inventory")
        print(json.dumps(dict(status="R23B_FULL24_PROFILE_COMPLETE_NOT_QUALITY",samples=24,legacy_exact_cross_run_samples=10,new_same_process_triple_samples=14,hook_calls=24*47*9)),flush=True)
BATCH_BLOB="08abbbfbd930e532d41602dd29a5c70f2bc474e3"
if __name__=="__main__":main()
