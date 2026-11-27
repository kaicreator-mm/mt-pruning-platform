"""R23B accepted-score96 GGUF materialization, separate explicit resource admission.
No deletion, quantization, inference, or automatic acceptance. Default header preflight.
Requires exact actual replay audit and accepted fresh decision, not merely fit improvement.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,pathlib,shutil,subprocess,tempfile
P=pathlib.Path
ROOT=P.home()/".cache/mt-pruning"
SOURCE=ROOT/"gguf-r20-r1/k128-f16.gguf"
DEST=ROOT/"r23-candidates/score96-f16"
READER=ROOT/"r20-r1/llama.cpp/gguf-py"
PUBLISHER_BLOB="ee0a4b5871d664016d0cfd5d2fba4085d162a985"
READER_SHA="9501f9df1f81ba5697600e5ee668c82588945d558e0974e506011af64cd76e7c"
HEADER_SHA="aa7df89983ceb9a0278905e683f0568ba426d62a824155a538e0359eba6f049f"
FIT_SHA="f2ab6eb0377932bbddb0335e6543534896f42e01a17b70cc6ae8672f30ff4d55"
MASK_SHA="7f0eca9be13b309e293ff4902c880c8f2af769120ebe2cf516bb166b7f074dec"
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
def decision_mask(fit,decision,audit,decision_sha):
    need(fit["status"]=="FIT_FROZEN_NOT_ADMITTED" and fit["mask_sha256"]==MASK_SHA and bool(fit["changes"]),"fit freeze")
    need(decision["status"]=="LOCAL_PROXY_CANDIDATE_READY_FOR_REAL_VALIDATION" and decision["fit_mask_sha256"]==MASK_SHA and decision["overall_guard"] is True and not decision["violations"] and decision["holdout_primary_relative_improvement"]>=.05,"fresh gate")
    need(audit["status"]=="FULL_SCORE_PROXY_REPLAY_PASS_NOT_QUALITY" and audit["decision"]==decision["status"] and audit["decision_sha256"]==decision_sha and audit["candidate_sha256"]==MASK_SHA and audit["holdout_primary_relative_improvement"]==decision["holdout_primary_relative_improvement"] and audit["overall_guard"] is True and not audit["violations"],"actual replay audit")
    m=decision["candidate"]["kept"]
    need(m==fit["kept"] and decision["candidate"]["mask_sha256"]==MASK_SHA and sha(canon(m))==MASK_SHA,"exact accepted mask")
    need(set(m)=={str(i) for i in range(1,48)} and all(v==sorted(set(v)) and len(v)==96 and all(type(i) is int and 0<=i<128 for i in v) for v in m.values()),"mask shape")
    return m
def admitted(a,decision_sha,audit_sha,s):
    need(a["status"]=="R23B_SCORE96_RESOURCE_ADMITTED" and a["decision_sha256"]==decision_sha and a["audit_sha256"]==audit_sha and a["source_sha256"]==s.SOURCE_SHA and a["mask_sha256"]==MASK_SHA and a["destination"]==str(DEST) and type(a["admission_comment"]) is int and a["admission_comment"]>0,"resource admission")
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
    # Reuse pinned publisher production-independent synthetic tests.
    pub.selftest(s)
    try:decision_mask({"status":"FIT_FROZEN_NOT_ADMITTED","mask_sha256":MASK_SHA,"changes":[1]},{"status":"NO_ADMISSION_FROM_LOCAL_PROXY"},{}, "x")
    except ValueError:pass
    else:raise AssertionError("rejected decision accepted")
    try:admitted({"status":"NO_ADMISSION"},"x","y",s)
    except ValueError:pass
    else:raise AssertionError("missing admission accepted")
    print(json.dumps(dict(status="PASS_SYNTHETIC_SCORE96_BUILD_GATE",negative=2,real_build="NOT_RUN")),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument("--publisher",type=P,required=True);p.add_argument("--slicer",type=P,required=True);p.add_argument("--audit",type=P);p.add_argument("--audit-sha");p.add_argument("--admission",type=P);p.add_argument("--admission-sha");p.add_argument("--selftest",action="store_true");p.add_argument("--execute",action="store_true");q=p.parse_args()
    pub=load(q.publisher,PUBLISHER_BLOB,"publisher");s=pub.load_slicer(q.slicer)
    if q.selftest:selftest(pub,s);return
    need(q.audit and q.audit_sha,"exact replay receipt required")
    araw=read(q.audit);need(sha(araw)==q.audit_sha,"audit bytes");audit=json.loads(araw)
    dst=ROOT/"r23-results/score-proxy-v1";fraw=read(dst/"fit-frozen.json");need(sha(fraw)==FIT_SHA,"fit bytes");draw=read(dst/"decision.json");dsha=sha(draw)
    mask=decision_mask(json.loads(fraw),json.loads(draw),audit,dsha)
    with (ROOT/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=preflight(s,mask);print(json.dumps(result),flush=True)
        if not q.execute:return
        need(not result["blockers"] and q.admission and q.admission_sha,"live resources and explicit admission")
        raw=read(q.admission);need(sha(raw)==q.admission_sha,"resource admission bytes");admitted(json.loads(raw),dsha,q.audit_sha,s)
        DEST.parent.mkdir(mode=0o700,exist_ok=True);need(DEST.parent==DEST.parent.resolve() and not DEST.parent.is_symlink(),"dedicated directory")
        need(not preflight(s,mask)["blockers"],"resource recheck")
        rec=pub.publish(s,SOURCE,DEST,mask,s.SOURCE_SHA,MASK_SHA,s.RESERVE,lambda path:pub.check_reader(s,path,READER,READER_SHA))
        print(json.dumps(dict(status="REAL_BYTES_PUBLISHED_NOT_RUNTIME_QUALIFIED",admission_sha256=q.admission_sha,decision_sha256=dsha,audit_sha256=q.audit_sha,receipt=rec),allow_nan=False),flush=True)
if __name__=="__main__":main()
