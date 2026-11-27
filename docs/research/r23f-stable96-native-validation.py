"""Stable96 native validation adapter. Reuse pinned native protocol and scorer.
Default preflight. No implicit build, GPU launch, cleanup or qualification.
Requires exact accepted stable96 proxy evidence and publication receipt.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,pathlib,re,sys
P=pathlib.Path;ROOT=P.home()/".cache/mt-pruning"
PINS={"driver":"0cdca8d43df86722775aad3d391001c78872926f","score":"ee010f8e92e649dfca9b930c8d19c414d99e26e2"}
BUILD_SHA="2ab4766135355e8d2a0328af316ae28054cf159768d7ce5749c360fad2054cb7"
SOURCE_SHA="c0f31f9243f2be7c67971b3195606bbb5ed4fa9ceb035add2520480b67f38e5b"
BASE_FULL="a4eb8d2adc9d446139449d11571359ea28d2a7a959a1dc644e5e6cab16da7ea3"
SIZE=45953803936
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def read(p):
    need(p.is_file() and not p.is_symlink(),"regular file "+str(p));return p.read_bytes()
def load(p,n):
    raw=read(p);need((sha(raw)==BUILD_SHA if n=="build" else blob(raw)==PINS[n]),"source "+n)
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def publication(pub,mask_sha):
    a=pub["artifact"]
    need(pub["status"]=="PUBLISHED_VERIFIED_BYTES_NOT_RUNTIME_QUALIFIED" and pub["slicer_blob"]=="b2cd41401b73fbad04e4ced615d7ea1861c14183" and pub["runtime_qualification"]==pub["quality_qualification"]=="NOT_RUN","publication status")
    need(a["status"]=="BYTE_SLICE_VERIFIED_NOT_RUNTIME_QUALIFIED" and a["keep"]==96 and a["source_sha256"]==SOURCE_SHA and a["mask_sha256"]==mask_sha and a["bytes"]==SIZE and re.fullmatch("[0-9a-f]{64}",a["sha256"]),"artifact identity")
    w=pub["reader_witness"];need(w["status"]=="PASS" and w["tensor_count"]==a["tensors_verified"]==len(a["tensor_sha256"]) and w["reader_module_sha256"]=="9501f9df1f81ba5697600e5ee668c82588945d558e0974e506011af64cd76e7c","reader evidence")
    return a
def check_run(plan,identity,phase,prov):
    need(plan["name"]=="stable96" and plan["phase"]==phase and plan["expected_model_sha256"]==plan["actual_model_sha256"]==identity[2] and plan["model"]==str(identity[0]) and plan["model_bytes"]==identity[1],"run model")
    need(all(plan[k]==v for k,v in prov.items()),"run evidence binding")
def overlap(pilot,full):
    need(len(pilot)==6 and len(full)==120 and set(pilot)<=set(full),"pilot/full inventory")
    need(all((p["raw_tokens"],p["text"])==(full[i]["raw_tokens"],full[i]["text"]) for i,p in pilot.items()),"pilot/full reproducibility")
def selftest():
    a=dict(status="BYTE_SLICE_VERIFIED_NOT_RUNTIME_QUALIFIED",keep=96,source_sha256=SOURCE_SHA,mask_sha256="m",bytes=SIZE,sha256="a"*64,tensors_verified=1,tensor_sha256={"x":"b"*64})
    p=dict(status="PUBLISHED_VERIFIED_BYTES_NOT_RUNTIME_QUALIFIED",slicer_blob="b2cd41401b73fbad04e4ced615d7ea1861c14183",runtime_qualification="NOT_RUN",quality_qualification="NOT_RUN",artifact=a,reader_witness=dict(status="PASS",tensor_count=1,reader_module_sha256="9501f9df1f81ba5697600e5ee668c82588945d558e0974e506011af64cd76e7c"))
    need(publication(p,"m")==a,"publication positive")
    def reject(f):
        try:f()
        except ValueError:return
        raise AssertionError("bad evidence")
    reject(lambda:publication(p,"other"));reject(lambda:publication(dict(p,reader_witness=dict(p["reader_witness"],status="FAIL")),"m"))
    x={str(i):dict(raw_tokens=[i],text=str(i)) for i in range(120)};y={k:x[k] for k in list(x)[:6]};overlap(y,x)
    reject(lambda:overlap(dict(y,**{"0":dict(raw_tokens=[1],text="0")}),x))
    identity=(P("/tmp/stable96"),SIZE,"a"*64);prov={"audit_sha256":"b"*64}
    plan=dict(name="stable96",phase="pilot",expected_model_sha256="a"*64,actual_model_sha256="a"*64,model=str(identity[0]),model_bytes=SIZE,**prov);check_run(plan,identity,"pilot",prov)
    reject(lambda:check_run(dict(plan,audit_sha256="c"*64),identity,"pilot",prov))
    print(json.dumps(dict(status="PASS_SYNTHETIC_NATIVE_STABLE96_ADAPTER",negative=4,real_runtime="NOT_RUN")))
def main():
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true");p.add_argument("--inputs",type=P);p.add_argument("--phase",choices=("pilot","full"));p.add_argument("--audit",type=P);p.add_argument("--audit-sha");p.add_argument("--candidate-sha");p.add_argument("--publication-sha");p.add_argument("--admission",type=P);p.add_argument("--admission-sha");p.add_argument("--execute",action="store_true");p.add_argument("--verify",action="store_true");q=p.parse_args()
    if q.selftest:selftest();return
    need(q.inputs and q.phase and q.audit and q.audit_sha and q.candidate_sha and q.publication_sha and not(q.execute and q.verify),"explicit inputs")
    d=q.inputs;b=load(d/"build.py","build");drv=load(d/"driver.py","driver");score=load(d/"score.py","score");proto=drv.load_protocol(d/"protocol.py")
    root=ROOT/"r23-results";dst=root/"expanded-stable96-proxy-v1";cr=read(root/"expanded-stable96-v1.json");fr=read(dst/"fit/decision.json");hr=read(dst/"fresh/decision.json");ar=read(q.audit)
    need(sha(cr)==q.candidate_sha and sha(ar)==q.audit_sha,"candidate/audit pins")
    mask=b.decision_mask(json.loads(cr),json.loads(fr),json.loads(hr),json.loads(ar),q.candidate_sha,sha(fr),sha(hr));mh=sha(canon(mask))
    bundle=ROOT/"r23-candidates/stable96-f16";pr=read(bundle/"receipt.json");need(sha(pr)==q.publication_sha,"publication pin")
    artifact=publication(json.loads(pr),mh);path=bundle/"model.gguf";need(path.is_file() and not path.is_symlink() and path.stat().st_size==SIZE,"published model size")
    identity=(path,SIZE,artifact["sha256"])
    prov=dict(candidate_receipt_sha256=q.candidate_sha,fit_decision_sha256=sha(fr),fresh_decision_sha256=sha(hr),audit_sha256=q.audit_sha,publication_receipt_sha256=q.publication_sha,mask_sha256=mh)
    original=drv.model_identity
    drv.model_identity=lambda name:identity if name=="stable96" else original(name)
    full,smoke=drv.inputs()
    def checked(phase):
        dest=ROOT/"native-greedy-runs"/f"stable96-{phase}"
        pl,seen,receipt=score.read_run(proto,drv,dest,full,smoke);check_run(pl,identity,phase,prov);return seen,receipt
    with (ROOT/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if q.verify:
            seen,rec=checked(q.phase)
            if q.phase=="pilot":
                print(json.dumps(dict(status="NATIVE_STABLE96_PILOT_PROTOCOL_PASS_NOT_QUALITY",**rec,**prov)),flush=True);return
            pilot,_=checked("pilot");overlap(pilot,seen)
            bp,base,br=score.read_run(proto,drv,ROOT/"native-greedy-runs/128-full",full,smoke)
            need(bp["name"]=="128" and bp["phase"]=="full" and bp["actual_model_sha256"]==SOURCE_SHA and br["journal_sha256"]==BASE_FULL,"exact baseline")
            raw=read(ROOT/"run-d01b-r19/corpus/corpus.jsonl");need(sha(raw)==drv.CORPUS_SHA,"corpus")
            ctx={x["id"]:x for x in map(json.loads,raw.splitlines()) if x["split"]=="validation"}
            result=score.compare(score.evaluator(),ctx,base,seen,"full")
            print(json.dumps(dict(result,baseline=br,candidate=rec,pilot_full_overlap_exact=True,**prov),allow_nan=False),flush=True);return
        if q.phase=="full":checked("pilot")
        plan,rows=drv.plan("stable96",q.phase,proto);plan.update(prov)
        print(json.dumps(plan),flush=True)
        if not q.execute:return
        need(not plan["blockers"] and q.admission and q.admission_sha,"resource admission")
        raw=read(q.admission);need(sha(raw)==q.admission_sha,"admission bytes");ad=json.loads(raw);drv.validate_admission(ad,plan)
        need(all(ad[k]==v for k,v in prov.items()),"exact provenance admission")
        drv.execute(plan,rows,proto,q.admission_sha)
if __name__=="__main__":main()
