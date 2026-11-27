"""R23 finite calibration continuation, one teacher load and immutable receipts.

Requires exact first real receipt and explicit bounded admission. Default is
preflight; no model opens without --execute. No validation or test data used.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,pathlib,resource,tempfile,time
P=pathlib.Path
ROOT=P.home()/".cache/mt-pruning"
FIRST_SHA="6223518953b4de74f267f8ea7913d449fd5f6be38fef80473c880d73e61011ae"
VERIFIER="e56b25b3fd92ccbd91455628606925a02af49450"
POLICY="FIXED_HISTORICAL_PREFIX_LOCAL_PROXY"
DIAGNOSTIC="job_01M8E3ZB2G00HAW2QFBG6FSQ23"
def forced_forward(c,engine,prompt,prefix,callback=None):
    cache=engine.new_cache(len(prompt)+len(prefix)+1);seen=[]
    with c.torch.inference_mode():
        if callback:callback("prefill",0)
        logits=engine.forward(prompt,cache,"prefill")
        seen.append({"phase":"prefill","step":0,"sha256":c.logits_sha(logits),"next":int(logits[-1].float().argmax())})
        for j,t in enumerate(prefix,1):
            if callback:callback("decode",j)
            logits=engine.forward([t],cache,"decode")
            seen.append({"phase":"decode","step":j,"input_token":t,"sha256":c.logits_sha(logits),"next":int(logits[-1].float().argmax())})
    return seen
def annotate(got,rec):
    prefix=rec["teacher_prefix_tokens"];logits=got["teacher_logits"]
    need([z["input_token"] for z in logits[1:]]==prefix,"forced input trace")
    mismatches=[{"step":j,"expected":t,"actual":z["next"]} for j,(t,z) in enumerate(zip(prefix,logits[:8]),1) if t!=z["next"]]
    got.update(prefix_policy=POLICY,consumed_prefix=prefix,threads=8,historical_argmax_mismatches=mismatches,historical_greedy_match=not mismatches)
    return got
PILOT="19c8c851b1647cc0f495d6a7f1122219c42a0cdb"
DRIVER="0735238e15f13fd75e3f6939fc31aa834d8b5854"
COLLECTOR="02daf65bb7fe0526b2a700d17013054f7b0eea46"
SELECTOR="f343ec873d6ab3801918cb20d795bc4d6a7c5155"
SPARSE="1c38f6e928ff653fbbef584b298ff65e194900fe"
def need(v,m):
    if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def load(p,pin,name):
    b=p.read_bytes();need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==pin,"source "+name)
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def validate_admission(a,v,rows,index,stop,verifier_job):
    need(a["status"]=="R23_BOUNDED_CALIBRATION_ADMISSION" and a["teacher_digest"]==v.TEACHER and a["candidate_masks"]==v.MASKS and a["engine_head"]==v.ENGINE and a["plan_blob"]==v.PLAN,"admission identity")
    need(a["first_sha256"]==FIRST_SHA and a["first_verifier_job"]==verifier_job and a["verifier_blob"]==VERIFIER and a["pilot_blob"]==PILOT,"first/source identity")
    need(type(index)==type(stop)==int and 1<=index<stop<=24 and a["first"]==index and a["stop"]==stop and a["sample_ids"]==[r["id"] for r in rows[index:stop]],"bounded scope")
    need(type(a["admission_comment"]) is int and a["admission_comment"]>0,"coordinator admission")
    need(a["prefix_policy"]==POLICY and a["threads"]==8 and a["diagnostic_job"]==DIAGNOSTIC and index>=10,"fixed prefix continuation admission")
def publish(dst,value):
    """Exclusive atomic link after file fsync; never replace a previous result."""
    need(dst.parent.is_dir() and dst.parent==dst.parent.resolve(),"result directory")
    need(not os.path.lexists(dst),"destination exists")
    raw=canon(value)+b"\n"
    fd,tmp=tempfile.mkstemp(prefix=".r23-unpublished-",dir=dst.parent.parent)
    try:
        with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.link(tmp,dst)
        directory=os.open(dst.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        os.unlink(tmp)
    return sha(raw)
def selftest():
    from types import SimpleNamespace
    v=SimpleNamespace(TEACHER="t",MASKS={"a":"m"},ENGINE="e",PLAN="p")
    rows=[{"id":str(i)} for i in range(24)]
    a=dict(status="R23_BOUNDED_CALIBRATION_ADMISSION",teacher_digest="t",candidate_masks={"a":"m"},engine_head="e",plan_blob="p",first_sha256=FIRST_SHA,first_verifier_job="j",verifier_blob=VERIFIER,pilot_blob=PILOT,first=10,stop=24,sample_ids=[r["id"] for r in rows[10:]],admission_comment=1,prefix_policy=POLICY,threads=8,diagnostic_job=DIAGNOSTIC)
    validate_admission(a,v,rows,10,24,"j")
    def rejects(fn):
        try:fn()
        except (ValueError,FileExistsError):return
        raise AssertionError("invalid scope accepted")
    rejects(lambda:validate_admission(a,v,rows,11,24,"j"))
    rejects(lambda:validate_admission(dict(a,sample_ids=["wrong"]),v,rows,10,24,"j"))
    rejects(lambda:validate_admission(dict(a,first_sha256="bad"),v,rows,10,24,"j"))
    rejects(lambda:validate_admission(a,v,rows,10,24,"bad"))
    with tempfile.TemporaryDirectory() as d:
        p=P(d)/"batch";p.mkdir();f=p/"01.json"
        h=publish(f,{"ok":True});need(h==sha(f.read_bytes()),"receipt SHA")
        rejects(lambda:publish(f,{"ok":False}));need(f.read_bytes()==b'{"ok":true}\n',"no overwrite")
        need(list(P(d).iterdir())==[p],"no published staging")
    rejects(lambda:validate_admission(dict(a,prefix_policy="strict"),v,rows,10,24,"j"))
    rejects(lambda:validate_admission(dict(a,threads=16),v,rows,10,24,"j"))
    # Independently inspect forced actual token delivery and callback order with a fake engine.
    from contextlib import nullcontext
    class Logits:
        def __init__(self,n):self.n=n
        def __getitem__(self,i):return self
        def float(self):return self
        def argmax(self):return self.n
    class E:
        def __init__(self):self.inputs=[]
        def new_cache(self,n):return object()
        def forward(self,ids,cache,phase):
            self.inputs.append((list(ids),phase));return Logits(99)
    e=E();calls=[];c=SimpleNamespace(torch=SimpleNamespace(inference_mode=nullcontext),logits_sha=lambda t:"0"*64)
    got=forced_forward(c,e,[12,13],[3,4,5],lambda p,j:calls.append((p,j)))
    assert e.inputs==[([12,13],"prefill"),([3],"decode"),([4],"decode"),([5],"decode")]
    assert calls==[("prefill",0),("decode",1),("decode",2),("decode",3)]
    x=annotate({"teacher_logits":got},{"teacher_prefix_tokens":[3,4,5]})
    assert len(x["historical_argmax_mismatches"])==3 and not x["historical_greedy_match"] and x["consumed_prefix"]==[3,4,5]
    print(json.dumps({"status":"PASS_SYNTHETIC_FIXED_PREFIX_BATCH_V2","negative":7,"real_capture":"NOT_RUN"}))
def main():
    p=argparse.ArgumentParser()
    for n in ("verifier","pilot","driver","collector","witness","selector","sparse","plan","original","guarded","p04"):p.add_argument("--"+n,type=P)
    p.add_argument("--selftest",action="store_true");p.add_argument("--execute",action="store_true");p.add_argument("--admission",type=P)
    q=p.parse_args()
    if q.selftest:selftest();return
    need(all(getattr(q,n) for n in ("verifier","pilot","driver","collector","witness","selector","sparse","plan","original","guarded","p04")),"input inventory")
    v=load(q.verifier,VERIFIER,"verifier");d=load(q.driver,DRIVER,"driver")
    c=load(q.collector,COLLECTOR,"collector");s=load(q.sparse,SPARSE,"sparse")
    pilot=load(q.pilot,PILOT,"pilot");selector=load(q.selector,SELECTOR,"selector");c.verify_code();w=c.load_witness(q.witness)
    corpus=ROOT/"run-d01b-r19/corpus/corpus.jsonl";baseline=ROOT/"run-d01b-r19/baseline/records.jsonl"
    c.pilot_inputs(q.plan,corpus,baseline,q.original,q.guarded)
    plan,rows=v.frozen(q.plan)
    ctx={x["id"]:x for x in map(json.loads,corpus.read_bytes().splitlines()) if x["split"]=="calibration"}
    masks=selector.masks_from(q.original,q.guarded,q.p04)
    need({n:selector.canon(m) for n,m in masks.items()}==v.MASKS,"five masks")
    masks={n:{int(l):ids for l,ids in m.items()} for n,m in masks.items()}
    first,report=v.first_receipt(ROOT/"r23-results/sparse-first.jsonl",ROOT/"r23-admissions/sparse-first.json",FIRST_SHA,rows[0])
    batch=ROOT/"r23-results/batch";done=v.load_batch(batch,rows);index=1+len(done)
    pre=d.preflight();print(json.dumps(dict(pre,completed=index,total=24,next=rows[index]["id"] if index<24 else None)),flush=True)
    if not q.execute:return
    need(index<24 and not pre["blockers"] and q.admission is not None,"execution admission")
    raw=q.admission.read_bytes();a=v.strict(raw);need(raw==canon(a),"canonical admission bytes")
    stop=a["stop"];validate_admission(a,v,rows,index,stop,"job_01M8E36PQZ00R8A9S4FFJPP57F")
    # Separate exclusive lock for the full batch; no unrelated model job authorized.
    lock=ROOT/"r23-admissions/batch.lock"
    with lock.open("a+") as lf:
        fcntl.flock(lf.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(len(v.load_batch(batch,rows))+1==index,"progress changed")
        need(not d.preflight()["blockers"],"exclusive resources changed")
        adpath=ROOT/"r23-admissions"/f"batch-{index:02d}-{stop:02d}.json"
        with adpath.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
        import torch
        torch.set_num_threads(8);torch.set_num_interop_threads(1)
        started=time.monotonic()
        ident=d.checkpoint_identity(d.SOURCE,cache_dir=None,fresh=True,exact_inventory=True)
        need(ident["teacher"]==d.TEACHER and ident["kind"]=="pinned-snapshot" and ident["num_experts"]==128,"full teacher identity")
        need(not d.preflight()["blockers"],"resources changed after identity")
        engine=c.engine_module.Engine(c.engine_module.Checkpoint(d.SOURCE),threads=8)
        c.forward_sequence=lambda engine,prompt,prefix,callback=None:forced_forward(c,engine,prompt,prefix,callback)
        print(json.dumps({"status":"TEACHER_READY","teacher_digest":ident["teacher"],"admission_sha256":sha(raw),"first":index,"stop":stop}),flush=True)
        for i in range(index,stop):
            need(not d.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"unexpected GPU workload")
            rec=dict(rows[i],split="calibration");prompt=ctx[rec["id"]]["prompt_ids"];t=time.monotonic()
            got=annotate(pilot.capture(c,w,s,engine,rec,prompt,masks),rec)
            got.update(teacher_digest=ident["teacher"],admission_sha256=sha(raw),r20_exact_head=d.ENGINE_HEAD,engine_head=d.ENGINE_HEAD,elapsed_with_identity_seconds=time.monotonic()-started,sample_seconds=time.monotonic()-t,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,torch=torch.__version__)
            report=v.validate(rec,got,sha(raw),POLICY)
            report.update(historical_argmax_mismatches=got["historical_argmax_mismatches"])
            envelope=dict(index=i,id=rec["id"],admission=a,admission_sha256=sha(raw),capture_sha256=sha(canon(got)),capture=got)
            h=publish(batch/f"{i:02d}.json",envelope)
            print(json.dumps(dict(report,index=i,completed=i+1,receipt_sha256=h,sample_seconds=got["sample_seconds"])),flush=True)
        verified=v.load_batch(batch,rows);need(len(verified)+1==stop,"complete batch receipt coverage")
        print(json.dumps({"status":"BOUNDED_CAPTURE_COMPLETE_NOT_QUALITY","completed":stop,"total":24,"first_sha256":FIRST_SHA,"admission_sha256":sha(raw)}),flush=True)
if __name__=="__main__":main()
