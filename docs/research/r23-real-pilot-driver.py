"""R23 one-real-sample pilot entry. Default read-only preflight.

No real weights opened until explicit admission and live resource checks.
No automatic full pilot or validation/test generation.
"""
import argparse,hashlib,importlib.util,json,os,resource,subprocess,time
from pathlib import Path
import torch
from mtpruning.identity import checkpoint_identity
ROOT=Path.home()/".cache/mt-pruning"
REPO=Path.home()/"mt-pruning-platform"
SOURCE=ROOT/"hf_home/hub/models--tencent--Hy-MT3-30B-A3B/snapshots/08ae9acea91abe265c6236e757e49d8851e1ce06"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
ENGINE_HEAD="3212303e1e02994a5124db8c85195b91325c7a7c"
COLLECTOR_BLOB="02daf65bb7fe0526b2a700d17013054f7b0eea46"
SELECTOR_BLOB="f343ec873d6ab3801918cb20d795bc4d6a7c5155"
FIRST="calibration:zh-my:348"
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def run(args):
    return subprocess.run(args,check=True,capture_output=True,text=True,timeout=20).stdout.strip()
def load(p,pin,name):
    b=p.read_bytes();need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==pin,name+" blob")
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def preflight():
    need(run(["git","-C",str(REPO),"rev-parse","HEAD"])==ENGINE_HEAD,"engine HEAD")
    need(not run(["git","-C",str(REPO),"status","--porcelain","--","mtpruning","scripts","demos/followup"]),"dirty engine")
    cfg=json.loads((SOURCE/"config.json").read_bytes())
    need(cfg["num_experts"]==128 and cfg["num_hidden_layers"]==48 and cfg["num_experts_per_tok"]==8,"teacher metadata")
    idx=json.loads((SOURCE/"model.safetensors.index.json").read_bytes())
    shards=sorted(set(idx["weight_map"].values()))
    need(len(shards)==13 and all((SOURCE/p).is_file() for p in shards),"teacher shard inventory")
    gpu=run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"])
    pids=[int(x.strip()) for x in gpu.splitlines() if x.strip()]
    busy=[]
    for line in run(["ps","-eo","pid=,comm=,args="]).splitlines():
        parts=line.split(None,2)
        if len(parts)<3:continue
        pid,comm,args=parts
        if int(pid)==os.getpid():continue
        if comm.startswith("llama-") or (comm.startswith("python") and any(s in args for s in ("llama_run.py","full_quality.py","convert_hf_to_gguf.py","r22-guarded96","r23-real-pilot-driver"))):
            busy.append({"pid":int(pid),"comm":comm})
    mem={k:int(v.strip().split()[0])*1024 for k,v in (l.split(":",1) for l in Path("/proc/meminfo").read_text().splitlines()) if k in ("MemAvailable","SwapFree")}
    blockers=[]
    if pids or busy:blockers.append("ACTIVE_GPU_OR_MODEL_PROCESS")
    if mem["MemAvailable"]<55*(1<<30) or mem["SwapFree"]<1*(1<<30):blockers.append("MEMORY_PREFLIGHT")
    return {"status":"BLOCKED" if blockers else "RESOURCE_PREFLIGHT_ONLY","blockers":blockers,"gpu_pids":pids,"model_processes":busy,"memory":mem,"source":str(SOURCE),"shard_count":len(shards),"teacher_full_hash_rechecked":False}
def validate_admission(a,sample,mask_sha):
    need(a.get("status")=="R21_R23_PILOT_ADMITTED" and a.get("r20_author_completed") is True,"R21 admission")
    need(a.get("teacher_digest")==TEACHER and a.get("sample_id")==sample==FIRST and a.get("candidate_masks")==mask_sha,"admission scope")
    h=a.get("r20_exact_head","");need(isinstance(h,str) and len(h)==40 and all(c in "0123456789abcdef" for c in h),"R20 exact HEAD")
def selftest(mask_sha):
    a={"status":"R21_R23_PILOT_ADMITTED","r20_author_completed":True,"teacher_digest":TEACHER,"sample_id":FIRST,"candidate_masks":mask_sha,"r20_exact_head":ENGINE_HEAD}
    validate_admission(a,FIRST,mask_sha)
    for bad in (dict(a,r20_author_completed=False),dict(a,teacher_digest="0"*64),dict(a,sample_id="validation:zh-my:1"),dict(a,candidate_masks={}),dict(a,r20_exact_head="unknown")):
        try:validate_admission(bad,FIRST,mask_sha)
        except ValueError:continue
        raise AssertionError("bad admission accepted")
    print(json.dumps({"status":"PASS_ADMISSION_CONTRACT_TEST","negative_cases":5,"real_execution":"NOT_RUN"}))
def main():
    p=argparse.ArgumentParser()
    for n in ("collector","witness","selector","plan","original","guarded","p04"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--selftest",action="store_true");p.add_argument("--execute",action="store_true");p.add_argument("--admission",type=Path);p.add_argument("--admission-sha256")
    q=p.parse_args();c=load(q.collector,COLLECTOR_BLOB,"collector");s=load(q.selector,SELECTOR_BLOB,"selector")
    c.verify_code();w=c.load_witness(q.witness)
    rec,prompt,_=c.pilot_inputs(q.plan,ROOT/"run-d01b-r19/corpus/corpus.jsonl",ROOT/"run-d01b-r19/baseline/records.jsonl",q.original,q.guarded)
    need(rec["id"]==FIRST,"first sample")
    masks=s.masks_from(q.original,q.guarded,q.p04)
    pins={n:s.canon(m) for n,m in masks.items()}
    if q.selftest:selftest(pins)
    result=preflight();result.update(sample_id=FIRST,candidate_masks=pins,planned_token_expert_evaluations=4*47*128)
    print(json.dumps(result),flush=True)
    if not q.execute:return
    need(not result["blockers"],"resource blocked")
    need(q.admission is not None and q.admission_sha256 is not None,"explicit admission")
    raw=q.admission.read_bytes();need(sha(raw)==q.admission_sha256,"admission digest");a=json.loads(raw);validate_admission(a,FIRST,pins)
    # Admission is a coordination record; the coordinator must hold an exclusive
    # scheduling window. This is not a lock against unrelated external jobs.
    need(not preflight()["blockers"],"resource changed")
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    started=time.monotonic()
    ident=checkpoint_identity(SOURCE,cache_dir=None,fresh=True,exact_inventory=True)
    need(ident["teacher"]==TEACHER and ident["kind"]=="pinned-snapshot" and ident["num_experts"]==128,"actual teacher bytes")
    need(not preflight()["blockers"],"resource changed after identity")
    engine=c.engine_module.Engine(c.engine_module.Checkpoint(SOURCE),threads=8)
    got=c.capture(w,engine,rec,prompt,{n:{int(l):ids for l,ids in m.items()} for n,m in masks.items()})
    got.update(teacher_digest=ident["teacher"],admission_sha256=q.admission_sha256,r20_exact_head=a["r20_exact_head"],engine_head=ENGINE_HEAD,elapsed_with_identity_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,torch=torch.__version__)
    need(got["token_expert_evaluations"]==24064,"real pilot coverage")
    print(json.dumps(got,allow_nan=False,separators=(",",":")),flush=True)
if __name__=="__main__":main()

