import pathlib,tarfile,json,hashlib,csv,io,sys,math
p=pathlib.Path(sys.argv[1]);R="r23-cuda-hybrid-v1/"
def need(v,m):
 if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
with tarfile.open(p) as t:
 files={}
 for m in t.getmembers():
  need(m.name.startswith(R) and ".." not in pathlib.PurePosixPath(m.name).parts and m.isfile() and m.size<256*1024*1024 and m.name not in files,"inventory")
  files[m.name]=t.extractfile(m).read()
def raw(n):return files[R+n]
need(raw("terminal.txt").strip()==b"COMPLETE_CUDA_DIAGNOSTIC_NOT_ORACLE_REPLACEMENT","terminal")
need(raw("cpu-reference.exit").strip()==raw("cuda-full-forward.exit").strip()==b"0","exit")
for line in raw("source.sha256").decode().splitlines():
 h,n=line.split(None,1);need(sha(raw(n))==h,"source "+n)
q=json.loads(raw("hybrid-report.json"));need(q["reference_sha256"]==sha(raw("cpu-reference.pt")) and q["status"]=="CUDA_FULL_FORWARD_DIAGNOSTIC_NOT_EXACT_ORACLE" and q["teacher"]=="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94","report identity")
events=[json.loads(l) for l in raw("cpu-reference.log").splitlines() if l.startswith(b"{")]
done=[x for x in events if x.get("status")=="CPU_EXACT_REFERENCE_COMPLETE"]
need(len(done)==1 and done[0]["sha256"]==q["reference_sha256"] and done[0]["trajectories"]==6 and done[0]["steps"]==54,"oracle terminal")
need(len(q["results"])==2 and {r["id"].split(":")[1] for r in q["results"]}=={"zh-my","my-zh"},"fit scope")
metrics=[]
for r in q["results"]:
 need(len(r["steps"])==27 and {(v["step"],v["name"]) for v in r["steps"]}=={(j,n) for j in range(9) for n in ("teacher","original96","l42")},"step inventory")
 need(r["cpu_reference_seconds"]>0 and r["cuda_seconds"]>0 and abs(r["batched_speedup"]-r["cpu_reference_seconds"]/r["cuda_seconds"])<1e-9,"timing")
 for v in r["steps"]:
  need(all(math.isfinite(v[k]) and v[k]>=0 for k in ("nmse","kl","route_overlap","seconds")) and v["route_overlap"]<=8 and type(v["argmax_equal"]) is bool,"metrics")
  metrics.append(v)
need(q["max_nmse"]==max(v["nmse"] for v in metrics) and q["all_argmax_equal"]==all(v["argmax_equal"] for v in metrics),"aggregate")
for name,budget in (("static",6<<30),("experts",4<<30)):
 c=q["cache"][name];need(0<=c["bytes"]<=budget and c["misses"]>0 and c["hits"]>0,"cache bounds")
g=list(csv.DictReader(io.StringIO(raw("gpu-samples.csv").decode())));need(g,"gpu samples")
util=max(int(x[" utilization.gpu [%]"].split()[0]) for x in g)
mem=max(int(x[" memory.used [MiB]"].split()[0]) for x in g);need(util>0 and mem>0,"GPU active")
print(json.dumps(dict(status="CUDA_DIAGNOSTIC_EVIDENCE_REPLAY_PASS_NOT_ORACLE_ADMISSION",archive_sha256=sha(p.read_bytes()),report_sha256=sha(raw("hybrid-report.json")),reference_sha256=q["reference_sha256"],steps=len(metrics),max_nmse=q["max_nmse"],all_argmax_equal=q["all_argmax_equal"],peak_sampled_gpu_pct=util,peak_sampled_gpu_mib=mem,cache=q["cache"],timings=[{k:r[k] for k in ("id","cpu_reference_seconds","cuda_seconds","batched_speedup")} for r in q["results"]],limits="Checks source, evidence identity, reported metrics and aggregates, not independent numerical recomputation from raw CUDA logits or quality admission."),indent=2,sort_keys=True))
