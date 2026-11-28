"""Direct native llama.cpp completion protocol; no implicit model launch.

Frozen token-array prompt, explicit greedy settings, disabled prefix cache.
Response and independent server log must both agree before a row is usable.
"""
import argparse,hashlib,json,math,os,re,socket,subprocess,tempfile,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError

ROOT=Path.home()/".cache/mt-pruning/r20-r1"
EOS=120025
VOCAB=120832
def need(v,m):
    if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def body(ids,limit=640):
    need(isinstance(ids,list) and 0<len(ids) and all(type(x)==int and 0<=x<VOCAB for x in ids),"prompt token IDs")
    need(type(limit)==int and 1<=limit<=640 and len(ids)+limit<=2048,"context/limit")
    return {"prompt":ids,"stream":False,"temperature":0.0,"top_k":1,"top_p":1.0,"min_p":0.0,"seed":0,"n_predict":limit,"cache_prompt":False,"return_tokens":True,"stop":[],"ignore_eos":False,"repeat_last_n":0,"repeat_penalty":1.0,"frequency_penalty":0.0,"presence_penalty":0.0,"dynatemp_range":0.0,"dry_multiplier":0.0,"mirostat":0,"xtc_probability":0.0}
def validate(q,r,log):
    need(isinstance(r,dict) and r.get("stop") is True and r.get("truncated") is False,"completion status")
    settings=r["generation_settings"]
    for k in ("temperature","top_k","top_p","min_p","seed","n_predict","stream","stop","ignore_eos","repeat_last_n","repeat_penalty","frequency_penalty","presence_penalty","dynatemp_range","dry_multiplier","mirostat","xtc_probability"):
        need(k in settings and settings[k]==q[k],"effective parameter "+k+" "+repr(settings.get(k)))
    t=r["timings"];ts=r["tokens"]
    need(isinstance(ts,list) and bool(ts) and all(type(x)==int and 0<=x<VOCAB for x in ts),"actual tokens")
    need(len(ts)==r["tokens_predicted"]==t["predicted_n"],"generated token count")
    need(r["tokens_evaluated"]==len(q["prompt"]) and t["prompt_n"]==len(q["prompt"]) and t["cache_n"]==0,"prompt count/cache")
    need(r["stop_type"] in ("eos","limit") and isinstance(r["content"],str) and r.get("stopping_word","")=="","stop reason")
    if r["stop_type"]=="eos":
        need(ts[-1]==EOS and EOS not in ts[:-1] and len(ts)<=q["n_predict"],"EOS evidence")
    else:need(EOS not in ts and len(ts)==q["n_predict"],"length evidence")
    need(all(type(t[k]) in (int,float) and math.isfinite(t[k]) and t[k]>=0 for k in ("prompt_ms","predicted_ms")),"timing")
    actual=re.findall(r"sampler params: \n(?:[^\n]*\n){2}\s*top_k = (\d+), top_p = ([\d.]+),.*?temp = ([\d.]+)",log)
    need(actual==[("1","1.000","0.000")],"effective server sampler log")
    events=re.findall(r"slot\s+process_toke:\s+id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|\s+n_gen\s*=\s*(\d+),.*?next token:\s*(\d+)",log)
    end=re.findall(r"slot\s+release:\s+id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|\s+stop processing: n_tokens\s*=\s*(\d+), truncated\s*=\s*(\d+)",log)
    need(len(end)==1 and end[0][3]=="0","raw release")
    slot,task,_,_=end[0]
    need([(a,b,int(j),int(x)) for a,b,j,x in events]==[(slot,task,i+1,x) for i,x in enumerate(ts)],"raw token evidence")
    return {"text":r["content"].strip(),"raw_tokens":ts,"tokens":ts[:-1] if r["stop_type"]=="eos" else ts,"finish":"eos" if r["stop_type"]=="eos" else "length","task":int(task),"slot":int(slot),"timings":t}
def request_once(port,q,logfile,timeout=120,capture_dir=None):
    raw=json.dumps(q,separators=(",",":")).encode()
    off=logfile.stat().st_size
    started=time.monotonic()
    req=Request(f"http://127.0.0.1:{port}/completion",data=raw,headers={"Content-Type":"application/json"},method="POST")
    with urlopen(req,timeout=timeout) as f:
        result=f.read(16<<20)
    r=json.loads(result)
    deadline=time.monotonic()+5
    while True:
        with logfile.open("rb") as f:f.seek(off);log=f.read(16<<20)
        if b"stop processing: n_tokens" in log:break
        need(time.monotonic()<deadline,"release log not observed");time.sleep(.05)
    if capture_dir is not None:
        capture_dir.mkdir(mode=0o700,parents=False,exist_ok=False)
        for name,data in (("request.json",raw),("response.json",result),("server.log",log)):
            with (capture_dir/name).open("xb") as f:
                f.write(data);f.flush();os.fsync(f.fileno())
    verified=validate(q,r,log.decode("utf-8",errors="replace"))
    return {"request":q,"request_sha256":sha(raw),"response":r,"response_sha256":sha(result),"server_log_sha256":sha(log),"server_log_bytes":len(log),"verified":verified,"wall_seconds":time.monotonic()-started}
def synthetic_test():
    # Actual frozen llama-server, tiny random-weight HY-V3 fixture.
    # No original 30B tensors read, no GPU compute, one CPU thread.
    import torch
    from mtpruning.testing import tiny_checkpoint
    torch.set_num_threads(1)
    source=ROOT.parent/"hf_home/hub/models--tencent--Hy-MT3-30B-A3B/snapshots/08ae9acea91abe265c6236e757e49d8851e1ce06"
    server=ROOT/"build/bin/llama-server"
    need(sha(server.read_bytes())=="cf2b8861dc6e1de1205bbd32770885d6212bdecf977cb186e78e4c703a86972f","server binary")
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PATH=str(ROOT/"conda/bin")+":/usr/bin:/bin",LD_LIBRARY_PATH=str(ROOT/"conda/lib"))
    with tempfile.TemporaryDirectory(prefix="r20-greedy-tiny-") as d:
        p=Path(d);hf=tiny_checkpoint(p/"hf",source,seed=73)
        model=p/"tiny.gguf"
        converter=ROOT/"llama.cpp/convert_hf_to_gguf.py"
        command=[os.sys.executable,str(converter),str(hf),"--outtype","f16","--outfile",str(model)]
        # converter uses current existing Python packages and local tokenizer only.
        converted=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
        need(converted.returncode==0,"tiny conversion "+converted.stderr[-2200:])
        with socket.socket() as s:s.bind(("127.0.0.1",0));port=s.getsockname()[1]
        log=p/"server.log"
        args=[str(server),"--model",str(model),"--n-gpu-layers","0","--ctx-size","128","--batch-size","32","--threads","1","--threads-batch","1","--parallel","1","--cache-type-k","f16","--cache-type-v","f16","--no-warmup","--host","127.0.0.1","--port",str(port),"--log-verbosity","2147483647"]
        with log.open("wb") as f:
            child=subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                deadline=time.monotonic()+45
                while True:
                    need(child.poll() is None,"tiny server exit "+log.read_text(errors="replace")[-2200:])
                    try:
                        with urlopen(f"http://127.0.0.1:{port}/health",timeout=2) as response:
                            if response.status==200:break
                    except (HTTPError,URLError,TimeoutError):pass
                    need(time.monotonic()<deadline,"tiny server startup timeout");time.sleep(.2)
                q=body([120000,123,456,789,1000],8)
                a=request_once(port,q,log,timeout=25,capture_dir=p/"request-a")
                b=request_once(port,q,log,timeout=25,capture_dir=p/"request-b")
                for v,d in ((a,p/"request-a"),(b,p/"request-b")):
                    need(all(sha((d/f).read_bytes())==v[k] for f,k in (("request.json","request_sha256"),("response.json","response_sha256"),("server.log","server_log_sha256"))),"raw capture bytes")
                need(a["verified"]["raw_tokens"]==b["verified"]["raw_tokens"] and a["verified"]["text"]==b["verified"]["text"],"greedy repeat")
                def reject(v):
                    try:validate(q,v,"")
                    except (ValueError,KeyError):return
                    raise AssertionError("accepted missing log or settings")
                import copy
                for key in ("temperature","top_k","seed","n_predict"):
                    v=copy.deepcopy(a["response"]);v["generation_settings"][key]=999;reject(v)
                reject(a["response"])
                print(json.dumps({"status":"PASS_ACTUAL_TINY_CPU_SERVER_GREEDY_TRANSPORT","model_bytes":model.stat().st_size,"binary_sha256":sha(server.read_bytes()),"requests":[a,b],"identical_tokens":True,"raw_capture_hash_checks":6,"negative_checks":5,"real_30b_inference":"NOT_RUN"},ensure_ascii=False,separators=(",",":"),allow_nan=False))
            finally:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
def main():
    p=argparse.ArgumentParser();p.add_argument("--synthetic-test",action="store_true");a=p.parse_args()
    need(a.synthetic_test,"real inference requires resource admission and exact model identity")
    synthetic_test()
if __name__=="__main__":main()

