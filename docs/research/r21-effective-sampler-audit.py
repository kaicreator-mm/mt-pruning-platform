"""Read-only actual R20 effective-sampling audit and actual CLI request witness.

Uses the existing executable against a tiny local mock HTTP server (no
model, no GPU). The mock only receives a synthetic prompt.
"""
import collections,hashlib,http.server,json,os,pathlib,re,subprocess,threading
p=pathlib.Path.home()/".cache/mt-pruning/r20-r1"
def sha(b):return hashlib.sha256(b).hexdigest()
def need(x,m):
    if not x:raise ValueError(m)
def audit(k):
    root=p/f"runs/full-f16-k{k}-v2";raw=(root/"runs.jsonl").read_bytes();lines=raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):lines=lines[:-1]
    rows=[json.loads(l) for l in lines];need(rows,"no rows")
    log=(p/f"k{k}-v2-server.log").read_bytes()
    text=log.decode("utf-8",errors="replace")
    params=re.findall(r"sampler params: \n(?:[^\n]*\n){2}\s*top_k = (\d+), top_p = ([\d.]+),.*?temp = ([\d.]+)",text)
    need(len(params)>=len(rows),"parameter coverage")
    # First N tasks are exactly the completed journal prefix, not the next
    # currently generating request. All request launches are sequential.
    release=re.findall(r"slot\s+release:\s+id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|\s+stop processing: n_tokens\s*=\s*(\d+), truncated\s*=\s*(\d+)",text)
    need(len(release)>=len(rows),"release count")
    need(all(all(r["command"][r["command"].index(a)+1]==b for a,b in (("--temp","0"),("--top-k","1"),("--top-p","1"),("--seed","0"),("--n-predict","640"))) for r in rows),"requested arguments")
    counter=collections.Counter("/".join(x) for x in params[:len(rows)])
    # Debug 'launching slot' is the *previous task* slot state before the
    # current task assignment. It is NOT used to establish current seed.
    return {"records":len(rows),"driver_completed":(root/"driver.exit").exists() and (root/"driver.exit").read_text().strip()=="EXIT_CODE=0","record_prefix_sha256":sha(b"".join(lines)),"effective_top_k_top_p_temp_counts":dict(counter),"sampler_blocks":len(params),"release_blocks":len(release),"log_snapshot_sha256":sha(log)}
requests=[]
class Mock(http.server.BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send_json(self,obj):
        b=json.dumps(obj).encode();self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/health"):self.send_json({"status":"ok"})
        elif self.path.startswith("/v1/models"):self.send_json({"object":"list","data":[{"id":"synthetic-mock","object":"model"}]})
        elif self.path.startswith("/props"):self.send_json({"model_alias":"synthetic-mock","modalities":{"vision":False,"audio":False}})
        else:self.send_error(404)
    def do_POST(self):
        obj=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests.append({"path":self.path,"body":obj})
        events=[{"choices":[{"index":0,"delta":{"role":"assistant","content":"synthetic response"},"finish_reason":None}]},{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"timings":{"prompt_n":5,"predicted_n":2,"prompt_per_second":1.0,"predicted_per_second":1.0}}]
        b=("".join("data: "+json.dumps(e)+"\n\n" for e in events)+"data: [DONE]\n\n").encode()
        self.send_response(200);self.send_header("Content-Type","text/event-stream");self.send_header("Content-Length",str(len(b)));self.end_headers();self.wfile.write(b)
def mock_probe():
    server=http.server.ThreadingHTTPServer(("127.0.0.1",0),Mock)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    import tempfile
    try:
        with tempfile.TemporaryDirectory(prefix="r20-cli-mock-") as d:
            q=pathlib.Path(d);prompt=q/"prompt.txt";prompt.write_text("Synthetic parameter transport test.")
            cmd=[str(p/"build/bin/llama-cli"),"--server-base",f"http://127.0.0.1:{server.server_port}","--file",str(prompt),"--single-turn","--output-file",str(q/"out.txt"),"--no-display-prompt","--temp","0","--top-k","1","--top-p","1","--seed","0","--n-predict","640","--no-warmup"]
            env=dict(os.environ,PATH=str(p/"conda/bin")+":/usr/bin:/bin",LD_LIBRARY_PATH=str(p/"conda/lib"),CUDA_VISIBLE_DEVICES="")
            run=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=20)
            need(run.returncode==0,"CLI mock exit "+str(run.returncode)+" "+run.stderr[-1000:])
            need(len(requests)==1 and requests[0]["path"]=="/v1/chat/completions","mock request count")
            body=requests[0]["body"];need(not any(k in body for k in ("temperature","top_k","top_p","seed","max_tokens","n_predict")),"unexpected forwarded parameters")
            need("synthetic response" in (q/"out.txt").read_text(),"mock output")
            return {"status":"ACTUAL_CLI_MOCK_CONFIRMED_MISSING_REQUEST_PARAMETERS","cli_exit":run.returncode,"request":requests[0],"binary_sha256":sha((p/"build/bin/llama-cli").read_bytes()),"requested_flags":cmd[-11:],"source_sha256":sha((p/"llama.cpp/tools/cli/cli-context.cpp").read_bytes())}
    finally:
        server.shutdown();server.server_close();thread.join()
print(json.dumps({"status":"EFFECTIVE_SAMPLING_MISMATCH_CONFIRMED","runs":{str(k):audit(k) for k in (128,96,64)},"mock":mock_probe()},separators=(",",":"),ensure_ascii=False))

