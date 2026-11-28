"""R23F accepted-stable96 GGUF materialization, separate explicit resource admission.
No deletion, quantization, inference, or automatic acceptance. Default header preflight.
Requires exact actual replay audit and accepted fresh decision, not merely fit improvement.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,pathlib,shutil,subprocess,tempfile
P=pathlib.Path
ROOT=P.home()/".cache/mt-pruning"
SOURCE=ROOT/"gguf-r20-r1/k128-f16.gguf"
DEST=ROOT/"r23-candidates/stable96-f16"
READER=ROOT/"r20-r1/llama.cpp/gguf-py"
PUBLISHER_BLOB="ee0a4b5871d664016d0cfd5d2fba4085d162a985"
READER_SHA="9501f9df1f81ba5697600e5ee668c82588945d558e0974e506011af64cd76e7c"
HEADER_SHA="aa7df89983ceb9a0278905e683f0568ba426d62a824155a538e0359eba6f049f"
PLAN="a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
ORIGINAL="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
EXPECTED_BYTES=45953803936
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def read(p):
    need(p.is_file() and not p.is_symlink(),"regular file");return p.read_bytes()
def load(p,pin,name):
    need(blob(read(p))==pin,"source "+name);spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def gate_pass(x,stage):
    return x["status"]==("FIT_LOCAL_PROXY_PASS_NOT_ADMISSION" if stage=="fit" else "LOCAL_PROXY_CANDIDATE_READY_FOR_REAL_VALIDATION") and x["stage"]==stage and x["samples"]==(48 if stage=="fit" else 24) and x["unique_aligned_lines"]==(30 if stage=="fit" else 14) and x["overall_guard"] is True and not x["violations"] and x["primary_relative_improvement"]>=.05
def decision_mask(c,fit,fresh,audit,csha,fsha,hsha):
    need(c["status"]=="FROZEN_STABLE96_NOT_ADMITTED" and c["plan_blob"]==PLAN and c["base_mask_sha256"]==ORIGINAL and c["fit_lines"]==30 and c["swaps"]>0 and len(c["fit_ids"])==48,"frozen candidate")
    m=c["kept"];mh=sha(canon(m))
    need(mh==c["mask_sha256"] and set(m)=={str(i) for i in range(1,48)} and all(v==sorted(set(v)) and len(v)==96 and all(type(i) is int and 0<=i<128 for i in v) for v in m.values()),"mask shape")
    need(gate_pass(fit,"fit") and gate_pass(fresh,"fresh") and fit["candidate_mask_sha256"]==fresh["candidate_mask_sha256"]==mh,"fit/fresh gates")
    need(audit["status"]=="FULL_STABLE96_PROXY_REPLAY_NOT_QUALITY" and audit["candidate_receipt_sha256"]==csha and audit["mask_sha256"]==mh and audit["recovery_manifest_sha256"]==c["recovery_manifest_sha256"] and audit["fit_decision_sha256"]==fsha and audit["fresh_decision_sha256"]==hsha and audit["decision"]==fresh["status"] and audit["primary_relative_improvement"]==fresh["primary_relative_improvement"] and audit["overall_guard"] is True and not audit["violations"],"exact replay audit")
    return m
def admitted(a,csha,fsha,hsha,audit_sha,mask_sha,s):
    need(a["status"]=="R23F_STABLE96_RESOURCE_ADMITTED" and a["candidate_receipt_sha256"]==csha and a["fit_decision_sha256"]==fsha and a["fresh_decision_sha256"]==hsha and a["audit_sha256"]==audit_sha and a["source_sha256"]==s.SOURCE_SHA and a["mask_sha256"]==mask_sha and a["destination"]==str(DEST) and type(a["admission_comment"]) is int and a["admission_comment"]>0,"resource admission")
def preflight(s,mask):
    h,header,out,size,k=s.plan(SOURCE,mask)
    need(h["size"]==60159655456 and sha(h["raw"])==HEADER_SHA and size==EXPECTED_BYTES and k==96,"exact header plan")
    need(sha(read(READER/"gguf/gguf_reader.py"))==READER_SHA,"reader")
    gpu=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=15,check=True)
    pids=[int(x.strip()) for x in gpu.stdout.splitlines() if x.strip()]
    proc=subprocess.run(["ps","-eo","pid=,comm=,args="],capture_output=True,text=True,timeout=15,check=True)
    busy=[]
    for l in proc.stdout.splitlines():
        v=l.split(None,2)
        if len(v)<3:continue
        pid,comm,args=v
        if int(pid)==os.getpid():continue
        if comm.startswith("llama-") or (comm.startswith("python") and any(x in args for x in ("llama_run.py","full_quality.py","convert_hf_to_gguf.py","profile.py --","proxy.py --","cd.py --","prop.py --"))):busy.append({"pid":int(pid),"comm":comm})
    anchor=DEST.parent if DEST.parent.exists() else ROOT
    need(anchor.is_dir() and not anchor.is_symlink() and anchor.stat().st_dev==SOURCE.stat().st_dev,"same local filesystem")
    free=shutil.disk_usage(anchor).free;blockers=[]
    if pids or busy:blockers.append("ACTIVE_GPU_OR_MODEL_PROCESS")
    if free<size+s.RESERVE:blockers.append("DISK_RESERVE_SHORTFALL")
    if os.path.lexists(DEST):blockers.append("DESTINATION_EXISTS")
    return dict(status="BLOCKED" if blockers else "RESOURCE_PREFLIGHT_ONLY",blockers=blockers,source_header_sha256=sha(h["raw"]),mask_sha256=s.canon(mask),planned_bytes=size,free_bytes=free,reserve_bytes=s.RESERVE,shortfall_bytes=max(0,size+s.RESERVE-free),gpu_pids=pids,model_processes=busy,full_source_hash_rechecked=False)
def selftest(pub,s):
    pub.selftest(s)
    m={str(i):list(range(96)) for i in range(1,48)};mh=sha(canon(m))
    c=dict(status="FROZEN_STABLE96_NOT_ADMITTED",plan_blob=PLAN,base_mask_sha256=ORIGINAL,fit_lines=30,swaps=1,fit_ids=list(range(48)),kept=m,mask_sha256=mh,recovery_manifest_sha256="a"*64)
    fit=dict(status="FIT_LOCAL_PROXY_PASS_NOT_ADMISSION",stage="fit",samples=48,unique_aligned_lines=30,overall_guard=True,violations=[],primary_relative_improvement=.1,candidate_mask_sha256=mh)
    fresh=dict(fit,status="LOCAL_PROXY_CANDIDATE_READY_FOR_REAL_VALIDATION",stage="fresh",samples=24,unique_aligned_lines=14)
    audit=dict(status="FULL_STABLE96_PROXY_REPLAY_NOT_QUALITY",candidate_receipt_sha256="c",mask_sha256=mh,recovery_manifest_sha256="a"*64,fit_decision_sha256="f",fresh_decision_sha256="h",decision=fresh["status"],primary_relative_improvement=.1,overall_guard=True,violations=[])
    need(decision_mask(c,fit,fresh,audit,"c","f","h")==m,"positive gate")
    def rejects(f):
        try:f()
        except ValueError:return
        raise AssertionError("bad evidence accepted")
    rejects(lambda:decision_mask(c,fit,dict(fresh,violations=["my-zh/decode"]),audit,"c","f","h"))
    rejects(lambda:decision_mask(c,dict(fit,primary_relative_improvement=.049),fresh,audit,"c","f","h"))
    rejects(lambda:decision_mask(c,fit,dict(fresh,samples=23),audit,"c","f","h"))
    rejects(lambda:decision_mask(c,fit,fresh,dict(audit,recovery_manifest_sha256="wrong"),"c","f","h"))
    rejects(lambda:decision_mask(c,fit,fresh,audit,"other","f","h"))
    rejects(lambda:admitted({"status":"NO_ADMISSION"},"c","f","h","a",mh,s))
    print(json.dumps(dict(status="PASS_SYNTHETIC_STABLE96_BUILD_GATE",negative=6,real_build="NOT_RUN")),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument("--publisher",type=P,required=True);p.add_argument("--slicer",type=P,required=True);p.add_argument("--audit",type=P);p.add_argument("--audit-sha");p.add_argument("--candidate-sha");p.add_argument("--admission",type=P);p.add_argument("--admission-sha");p.add_argument("--selftest",action="store_true");p.add_argument("--execute",action="store_true");q=p.parse_args()
    pub=load(q.publisher,PUBLISHER_BLOB,"publisher");s=pub.load_slicer(q.slicer)
    if q.selftest:selftest(pub,s);return
    need(q.audit and q.audit_sha and q.candidate_sha,"exact replay and candidate receipt required")
    araw=read(q.audit);need(sha(araw)==q.audit_sha,"audit bytes");audit=json.loads(araw)
    root=ROOT/"r23-results";dst=root/"expanded-stable96-proxy-v1"
    craw=read(root/"expanded-stable96-v1.json");need(sha(craw)==q.candidate_sha,"candidate bytes")
    fraw=read(dst/"fit/decision.json");draw=read(dst/"fresh/decision.json");fsha=sha(fraw);dsha=sha(draw)
    mask=decision_mask(json.loads(craw),json.loads(fraw),json.loads(draw),audit,q.candidate_sha,fsha,dsha);mask_sha=sha(canon(mask))
    with (ROOT/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=preflight(s,mask);print(json.dumps(result),flush=True)
        if not q.execute:return
        need(not result["blockers"] and q.admission and q.admission_sha,"live resources and explicit admission")
        raw=read(q.admission);need(sha(raw)==q.admission_sha,"resource admission bytes");admitted(json.loads(raw),q.candidate_sha,fsha,dsha,q.audit_sha,mask_sha,s)
        DEST.parent.mkdir(mode=0o700,exist_ok=True);need(DEST.parent==DEST.parent.resolve() and not DEST.parent.is_symlink(),"dedicated directory")
        need(not preflight(s,mask)["blockers"],"resource recheck")
        rec=pub.publish(s,SOURCE,DEST,mask,s.SOURCE_SHA,mask_sha,s.RESERVE,lambda path:pub.check_reader(s,path,READER,READER_SHA))
        print(json.dumps(dict(status="REAL_BYTES_PUBLISHED_NOT_RUNTIME_QUALIFIED",admission_sha256=q.admission_sha,candidate_receipt_sha256=q.candidate_sha,fit_decision_sha256=fsha,fresh_decision_sha256=dsha,audit_sha256=q.audit_sha,publication_receipt_sha256=sha(read(DEST/"receipt.json")),receipt=rec),allow_nan=False),flush=True)
if __name__=="__main__":main()
