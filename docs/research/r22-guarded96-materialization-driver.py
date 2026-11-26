"""R22 exact-candidate execution entry point, serial and fail closed.

Default preflight reads model header only. Explicit --execute requires actual
R21 admission receipt and exclusive resources; no generation/inference launch.
"""
import argparse,hashlib,importlib.util,json,os,pathlib,shutil,subprocess,tempfile
P=pathlib.Path
PUBLISHER_BLOB="ee0a4b5871d664016d0cfd5d2fba4085d162a985"
MASK_BLOB="150616a10fbf503f815fb2f7fc281a81dcc28fea"
READER_SHA="9501f9df1f81ba5697600e5ee668c82588945d558e0974e506011af64cd76e7c"
HEADER_SHA="aa7df89983ceb9a0278905e683f0568ba426d62a824155a538e0359eba6f049f"
EXPECTED_BYTES=45953803936
ROOT=P.home()/".cache/mt-pruning"
SOURCE=ROOT/"gguf-r20-r1/k128-f16.gguf"
READER=ROOT/"r20-r1/llama.cpp/gguf-py"
DEST_PARENT=ROOT/"r22-candidates"
DEST=DEST_PARENT/"guarded96-f16"
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def load(path,pin,name):
    b=path.read_bytes();need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==pin,name+" blob")
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def mask_from(path,s):
    b=path.read_bytes();need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==MASK_BLOB,"mask blob")
    q=json.loads(b);need(q["keep"]==96 and q["expert_count"]==128 and q["top_k"]==8,"mask metadata")
    m=q["plans"]["global_guarded96"]["kept"];need(s.canon(m)==s.MASK_SHA,"canonical mask");return m
def preflight(s,mask):
    h,header,out,size,k=s.plan(SOURCE,mask)
    need(h["size"]==60159655456 and sha(h["raw"])==HEADER_SHA and size==EXPECTED_BYTES and k==96,"exact header plan")
    need(sha((READER/"gguf/gguf_reader.py").read_bytes())==READER_SHA,"reader")
    gpu=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=15,check=True)
    gpu_pids=[int(x.strip()) for x in gpu.stdout.splitlines() if x.strip()]
    proc=subprocess.run(["ps","-eo","pid=,comm=,args="],capture_output=True,text=True,timeout=15,check=True)
    busy=[]
    for line in proc.stdout.splitlines():
        parts=line.split(None,2)
        if len(parts)<3:continue
        pid,comm,args=parts
        if int(pid)==os.getpid():continue
        if comm.startswith("llama-") or (comm.startswith("python") and any(x in args for x in ("llama_run.py","full_quality.py","convert_hf_to_gguf.py","r23-collector"))):
            busy.append({"pid":int(pid),"comm":comm})
    anchor=DEST_PARENT if DEST_PARENT.exists() else ROOT
    need(anchor.is_dir() and not anchor.is_symlink() and anchor.stat().st_dev==SOURCE.stat().st_dev,"same local filesystem")
    free=shutil.disk_usage(anchor).free
    blockers=[]
    if gpu_pids or busy:blockers.append("ACTIVE_GPU_OR_MODEL_PROCESS")
    if free<size+s.RESERVE:blockers.append("DISK_RESERVE_SHORTFALL")
    if os.path.lexists(DEST):blockers.append("DESTINATION_EXISTS")
    return {"status":"BLOCKED" if blockers else "RESOURCE_PREFLIGHT_ONLY","blockers":blockers,"source_header_sha256":sha(h["raw"]),"mask_sha256":s.canon(mask),"planned_bytes":size,"free_bytes":free,"reserve_bytes":s.RESERVE,"shortfall_bytes":max(0,size+s.RESERVE-free),"gpu_pids":gpu_pids,"model_processes":busy,"full_source_hash_rechecked":False}
def admission(path,digest,s):
    need(path is not None and isinstance(digest,str) and len(digest)==64,"R21 admission required")
    b=path.read_bytes();need(sha(b)==digest,"admission SHA256")
    a=json.loads(b)
    need(a.get("status")=="R21_RESOURCE_ADMITTED" and a.get("r20_author_completed") is True and a.get("source_sha256")==s.SOURCE_SHA and a.get("mask_sha256")==s.MASK_SHA and a.get("destination")==str(DEST),"admission contract")
    head=a.get("r20_exact_head","");need(isinstance(head,str) and len(head)==40 and all(c in "0123456789abcdef" for c in head),"R20 exact HEAD")
    # Admission is a coordinator record, not proof by itself; live process and
    # free-space checks are repeated, and source bytes verified by P13.
    return a
def selftest(pub,s):
    with tempfile.TemporaryDirectory(prefix="r22-real-reader-") as d:
        p=P(d);src=p/"source.gguf";table=s.fixture(src);mask={"1":[0,3],"2":[1,2]}
        before=s.sha_file(src)
        def reader(path):return pub.check_reader(s,path,READER,READER_SHA)
        dest=p/"verified"
        receipt=pub.publish(s,src,dest,mask,before,s.canon(mask),0,reader,experts=4,layers=3,hidden=3,width=2,top=2)
        actual=s.parse(dest/"model.gguf");payload=(dest/"model.gguf").read_bytes();checks=0
        for t,(_,_,_,_,data) in zip(actual["tensors"],table):
            m=s.EXPERT.fullmatch(t["name"])
            if m:
                stride=len(data)//4;data=b"".join(data[i*stride:(i+1)*stride] for i in mask[m[1]])
            start=actual["start"]+t["off"];need(payload[start:start+t["bytes"]]==data,"payload oracle");checks+=1
        need(receipt["reader_witness"]["tensor_count"]==checks==46 and s.sha_file(src)==before,"reader and source")
        need(json.loads((dest/"receipt.json").read_bytes())==receipt,"receipt")
        try:admission(None,None,s)
        except ValueError:pass
        else:raise AssertionError("missing admission accepted")
        print(json.dumps({"status":"PASS_ATOMIC_PRODUCTION_READER_SYNTHETIC","reader":receipt["reader_witness"],"tensor_byte_checks":checks,"source_unchanged":True,"missing_admission_rejected":True}))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--slicer",type=P,required=True);p.add_argument("--publisher",type=P,required=True);p.add_argument("--mask",type=P,required=True)
    p.add_argument("--selftest",action="store_true");p.add_argument("--execute",action="store_true")
    p.add_argument("--admission",type=P);p.add_argument("--admission-sha256")
    q=p.parse_args();pub=load(q.publisher,PUBLISHER_BLOB,"publisher");s=pub.load_slicer(q.slicer);mask=mask_from(q.mask,s)
    if q.selftest:selftest(pub,s)
    result=preflight(s,mask);print(json.dumps(result),flush=True)
    if not q.execute:return
    need(not result["blockers"],"resource preflight blocked")
    a=admission(q.admission,q.admission_sha256,s)
    # Do not create or modify anything in the retained R20 tree.
    DEST_PARENT.mkdir(mode=0o700,exist_ok=True)
    need(DEST_PARENT==DEST_PARENT.resolve() and not DEST_PARENT.is_symlink(),"dedicated path")
    again=preflight(s,mask);need(not again["blockers"],"resource recheck blocked")
    receipt=pub.publish(s,SOURCE,DEST,mask,s.SOURCE_SHA,s.MASK_SHA,s.RESERVE,lambda path:pub.check_reader(s,path,READER,READER_SHA))
    print(json.dumps({"status":"REAL_BYTES_PUBLISHED_NOT_RUNTIME_QUALIFIED","admission_sha256":q.admission_sha256,"r20_exact_head":a["r20_exact_head"],"receipt":receipt},allow_nan=False))
if __name__=="__main__":main()

