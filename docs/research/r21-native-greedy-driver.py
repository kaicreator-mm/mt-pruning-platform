"""Finite native greedy validation driver, R20 correction / R22 reuse.
Default read-only preflight, never auto-launch. Execute one model/one phase.
Require explicit coordinator admission after existing R20 writer releases resources.
"""
import argparse,hashlib,importlib.util,json,os,re,shutil,socket,subprocess,time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError,URLError
P=Path
ROOT=P.home()/".cache/mt-pruning"
R20=ROOT/"r20-r1"
PROTOCOL_BLOB="85b3bbabeafa4b6bf318a2f8fb25e44900d5626f"
FULL_MANIFEST_SHA="b4fd031ae37e5fff2ac50d7cd7b098e5c2a7c4a11f23a7420c983a3eb80098d9"
WITNESS_SHA="f7c7f94a691047d7d03340c43e4615598f0b3b81a1ed71de3b7a0dc3cbd71c23"
CORPUS_SHA="9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
BASE_MODELS={
"128":("k128-f16.gguf",60159655456,"c0f31f9243f2be7c67971b3195606bbb5ed4fa9ceb035add2520480b67f38e5b"),
"96":("k96-f16.gguf",45953803808,"baf285001817a5e3b623c4d4e20844f1dbe23e14d79bf930ffc0c8b9aea3d9bd"),
"64":("k64-f16.gguf",31747952288,"c77b03276ecc1e4cb70d32f249996ed6c3f49a9e78f087bd734e7d187c20b444"),
}
BINS={
"llama-server":"cf2b8861dc6e1de1205bbd32770885d6212bdecf977cb186e78e4c703a86972f",
"libllama-server-impl.so":"18ffb29a829b2539eac260ad7ef91dd0ed6b2f7837ab7931c8735ad143747ce6",
"libllama.so":"56539d76d4ac449177efb10eb548be6dc3bb9a7b9218c22de413b7be9728fe04",
"libggml-cpu.so":"4a84bac13b73ac69b2d3bc3bfb503fc0000e8a513edb6614e0d3728ae9083bd8",
"libggml-cuda.so":"6917f4f7c97db04bb7096a77abe9835aeb4ae85b392e8bc75cf72bf99e9a7a6b",
"libggml.so":"2f6ed0e405ab2482ec99d6a4d72fb239d77a969f9b8604a5ec2c1cd6858f304c",
"libggml-base.so":"f5f23cb9ffd543a2b2571c6a038bf9d279f0a2eb8d0b0008d2e210406dad87bd",
"libllama-common.so":"fac5ee8d5dd6e3ead2fa3b6926bb61a1ae97ce453c94e4ef44657c01a422148a",
"libmtmd.so":"9fff963cd63201f617757f010effd160e774728b03f791daf0cac0e17085b190",
}
def need(v,m):
    if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def file_sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def load_protocol(p):
    b=p.read_bytes();need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==PROTOCOL_BLOB,"protocol blob")
    s=importlib.util.spec_from_file_location("native_protocol",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def inputs():
    raw=(ROOT/"run-d01b-r19/corpus/corpus.jsonl").read_bytes();need(sha(raw)==CORPUS_SHA,"corpus")
    ctx={x["id"]:x for x in map(json.loads,raw.splitlines()) if x["split"]=="validation"}
    need(len(ctx)==120,"validation corpus")
    a=(R20/"validation120/manifest.json").read_bytes();b=(R20/"witnesses/manifest.json").read_bytes()
    need(sha(a)==FULL_MANIFEST_SHA and sha(b)==WITNESS_SHA,"manifest identity")
    full=json.loads(a)["witnesses"];smoke=json.loads(b)["witnesses"]
    need(len(full)==len({x["id"] for x in full})==120 and {x["id"] for x in full}==set(ctx),"full coverage")
    need(len(smoke)==len({x["direction"] for x in smoke})==6,"smoke coverage")
    for x in full+smoke:
        c=ctx[x["id"]];need(x["prompt_ids"]==c["prompt_ids"] and x["direction"]==c["direction"] and len(x["prompt_ids"])<=1408,"frozen prompt IDs")
        need(sha(P(x["prompt_file"]).read_bytes())==x["prompt_sha256"],"prompt file digest")
    return full,smoke
def model_identity(name):
    if name in BASE_MODELS:
        f,n,h=BASE_MODELS[name];path=ROOT/"gguf-r20-r1"/f
        need(path.is_file() and path.stat().st_size==n,"existing model size")
        return path,n,h
    need(name=="guarded96","model")
    bundle=ROOT/"r22-candidates/guarded96-f16"
    receipt=json.loads((bundle/"receipt.json").read_bytes())
    a=receipt["artifact"];path=bundle/"model.gguf"
    need(receipt["status"]=="PUBLISHED_VERIFIED_BYTES_NOT_RUNTIME_QUALIFIED" and a["keep"]==96 and a["mask_sha256"]=="4589d613512ac0a07f02be500e8b2fabe3f4beb78ddd171e14fd26cd8078960c","candidate receipt")
    need(a["bytes"]==45953803936 and path.stat().st_size==a["bytes"],"candidate size")
    return path,a["bytes"],a["sha256"]
def run_text(cmd):
    return subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=20).stdout.strip()
def resources():
    gpu=[int(x.strip()) for x in run_text(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]).splitlines() if x.strip()]
    busy=[]
    for line in run_text(["ps","-eo","pid=,comm=,args="]).splitlines():
        p=line.split(None,2)
        if len(p)!=3:continue
        pid,comm,args=p
        if int(pid)==os.getpid():continue
        if comm.startswith("llama-") or (comm.startswith("python") and any(x in args for x in ("llama_run.py","full_quality.py","convert_hf_to_gguf.py","r23-real-pilot","r22-guarded96","native-greedy-driver"))):busy.append({"pid":int(pid),"comm":comm})
    free=shutil.disk_usage(ROOT).free
    return {"gpu_pids":gpu,"model_processes":busy,"free_bytes":free,"blockers":(["ACTIVE_GPU_OR_MODEL_PROCESS"] if gpu or busy else [])+(["EVIDENCE_DISK_RESERVE"] if free<2*(1<<30) else [])}
def plan(name,phase,protocol):
    full,smoke=inputs();path,n,h=model_identity(name)
    rows=smoke if phase=="pilot" else full
    # Pilot has a fixed six-direction sample set, plus repeat of the two
    # primary samples to detect nondeterminism. No quality fitting.
    if phase=="pilot":rows=rows+[x for x in smoke if x["direction"] in ("zh-my","my-zh")]
    for x in rows:protocol.body(x["prompt_ids"])
    for k,v in BINS.items():need(file_sha(R20/"build/bin"/k)==v,"runtime pin "+k)
    rs=resources()
    dest=ROOT/"native-greedy-runs"/f"{name}-{phase}"
    if os.path.lexists(dest):rs["blockers"].append("DESTINATION_EXISTS")
    return {"status":"BLOCKED" if rs["blockers"] else "RESOURCE_PREFLIGHT_ONLY","name":name,"phase":phase,"model":str(path),"model_bytes":n,"expected_model_sha256":h,"model_full_hash_rechecked":False,"manifest_sha256":WITNESS_SHA if phase=="pilot" else FULL_MANIFEST_SHA,"sample_ids":[x["id"] for x in rows],"requests":len(rows),"destination":str(dest),"runtime_file_hashes":BINS,**rs},rows
def validate_admission(a,pl):
    need(a.get("status")=="NATIVE_GREEDY_EXCLUSIVE_ADMISSION" and a.get("prior_writer_released") is True,"exclusive admission")
    need(all(a.get(k)==pl[k] for k in ("name","phase","expected_model_sha256","manifest_sha256","destination")),"admission scope")
    need(a.get("protocol_blob")==PROTOCOL_BLOB,"protocol identity")
    h=a.get("reviewed_source_head","");need(isinstance(h,str) and re.fullmatch("[0-9a-f]{40}",h),"source head")
def save_json(path,obj):
    with path.open("x",encoding="utf-8") as f:json.dump(obj,f,ensure_ascii=False,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
def execute(pl,rows,proto,admission_sha):
    path=P(pl["model"]);dest=P(pl["destination"])
    need(file_sha(path)==pl["expected_model_sha256"],"actual model full digest")
    need(not resources()["blockers"],"resource changed during full hash")
    dest.parent.mkdir(mode=0o700,exist_ok=True)
    need(dest.parent==dest.parent.resolve(),"dedicated run parent")
    dest.mkdir(mode=0o700,exist_ok=False)
    # Single persistent server, explicit one slot, no cache reuse, exact
    # frozen prompt IDs. This is a NEW matched protocol, not v2 reuse.
    with socket.socket() as s:s.bind(("127.0.0.1",0));port=s.getsockname()[1]
    cmd=[str(R20/"build/bin/llama-server"),"--model",str(path),"--n-gpu-layers","99","--n-cpu-moe","40","--ctx-size","2048","--batch-size","512","--threads","16","--parallel","1","--cache-type-k","f16","--cache-type-v","f16","--no-warmup","--host","127.0.0.1","--port",str(port),"--log-verbosity","2147483647","--temp","0","--top-k","1","--top-p","1","--seed","0","--n-predict","640"]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PATH=str(R20/"conda/bin")+":/usr/bin:/bin",LD_LIBRARY_PATH=str(R20/"conda/lib"))
    save_json(dest/"run-plan.json",{**pl,"actual_model_sha256":pl["expected_model_sha256"],"model_full_hash_rechecked":True,"admission_sha256":admission_sha,"server_command":cmd})
    log=dest/"server.log";seen={}
    with log.open("xb") as f:
        child=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+180
            while True:
                need(child.poll() is None,"server exited before ready")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health",timeout=2) as r:
                        if r.status==200:break
                except (HTTPError,URLError,TimeoutError):pass
                need(time.monotonic()<deadline,"server startup deadline");time.sleep(.25)
            with urlopen(f"http://127.0.0.1:{port}/props",timeout=10) as f2:props=json.load(f2)
            need(props.get("model_path")==str(path),"loaded model path")
            save_json(dest/"props.json",props)
            with (dest/"runs.jsonl").open("x",encoding="utf-8") as journal:
                for i,x in enumerate(rows):
                    need(child.poll() is None,"server exited")
                    raw_dir=dest/f"request-{i:03d}"
                    q=proto.body(x["prompt_ids"])
                    rec=proto.request_once(port,q,log,timeout=180,capture_dir=raw_dir)
                    v=rec["verified"];need(rec["response"]["model"] in (str(path),props.get("model_alias")),"response model")
                    if x["id"] in seen:need((v["raw_tokens"],v["text"])==seen[x["id"]],"repeat determinism")
                    else:seen[x["id"]]=(v["raw_tokens"],v["text"])
                    row={"id":x["id"],"direction":x["direction"],"model_sha256":pl["expected_model_sha256"],"repeat":sum(y["id"]==x["id"] for y in rows[:i])>0,"request_directory":raw_dir.name,"text":v["text"],"tokens":v["tokens"],"raw_tokens":v["raw_tokens"],"finish":v["finish"],"timings":v["timings"],"request_sha256":rec["request_sha256"],"response_sha256":rec["response_sha256"],"server_log_sha256":rec["server_log_sha256"],"wall_seconds":rec["wall_seconds"],"task":v["task"],"slot":v["slot"]}
                    journal.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");journal.flush();os.fsync(journal.fileno())
                    print(json.dumps({"id":row["id"],"finish":row["finish"],"repeat":row["repeat"],"tokens":len(row["raw_tokens"])}),flush=True)
            save_json(dest/"complete.json",{"status":"NATIVE_GREEDY_RECORDS_COMPLETE_NOT_QUALITY","requests":len(rows),"unique_ids":len(seen),"journal_sha256":file_sha(dest/"runs.jsonl")})
        finally:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol",type=P,required=True);p.add_argument("--name",choices=("128","96","64","guarded96"),required=True);p.add_argument("--phase",choices=("pilot","full"),required=True)
    p.add_argument("--execute",action="store_true");p.add_argument("--admission",type=P);p.add_argument("--admission-sha256")
    q=p.parse_args();proto=load_protocol(q.protocol);pl,rows=plan(q.name,q.phase,proto);print(json.dumps(pl),flush=True)
    if not q.execute:return
    need(not pl["blockers"] and q.admission is not None and q.admission_sha256 is not None,"blocked or missing admission")
    raw=q.admission.read_bytes();need(sha(raw)==q.admission_sha256,"admission hash");validate_admission(json.loads(raw),pl)
    execute(pl,rows,proto,q.admission_sha256)
if __name__=="__main__":main()

