import pathlib,json,hashlib,math,time,torch,sys
import mtpruning.engine as E
from r23_cuda_hybrid_engine import Hybrid
P=pathlib.Path;R=P.home()/".cache/mt-pruning";D=P(__file__).parent
def need(v,m):
 if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(p):
 b=p.read_bytes();return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
need(torch.__version__=="2.5.1+cu121" and torch.cuda.is_available() and torch.cuda.get_device_capability()==(7,0),"V100 runtime")
need(blob(P(E.__file__))=="2af43ca1710b029fdde0d1135ea9879fe8defe56","pinned CPU engine")
refp=D/"cpu-reference.pt";raw=refp.read_bytes();ref=torch.load(refp,weights_only=True)
need(ref["teacher"]=="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94" and len(ref["rows"])==2,"oracle")
src=R/"hf_home/hub/models--tencent--Hy-MT3-30B-A3B/snapshots/08ae9acea91abe265c6236e757e49d8851e1ce06"
names=["teacher","original96","l42"];masks=[ref["masks"][n] for n in names]
torch.set_num_threads(8);torch.cuda.reset_peak_memory_stats();engine=Hybrid(E.Checkpoint(src))
results=[]
for r in ref["rows"]:
 prompt=r["prompt"];prefix=r["prefix"];need(len(prompt)+8<=256 and len(prefix)==8,"fit trajectory")
 engine.new_sequence(len(names),len(prompt)+8)
 start=time.monotonic();stats=[]
 for j,ids in enumerate([prompt]+[[v] for v in prefix]):
  t=time.monotonic();out,rs=engine.forward(ids,masks);torch.cuda.synchronize();dt=time.monotonic()-t
  rows=[len(prompt)//2-1,len(prompt)-1] if j==0 else [0]
  for k,n in enumerate(names):
   a=out[k,rows].double();b=r["results"][n]["logits"][j].double()
   need(a.shape==b.shape and torch.isfinite(a).all(),"logits")
   nmse=float((a-b).square().sum()/b.square().sum().clamp_min(1e-30))
   lp=b.log_softmax(-1);lq=a.log_softmax(-1);kl=float((lp.exp()*(lp-lq)).sum(-1).mean())
   old={x["layer"]:x["ids"] for x in r["results"][n]["routes"] if x["step"]==j}
   need(len(rs)==len(old)==47,"route coverage")
   overlap=[float((rs[l][k,rows,:,None]==v[:,None,:]).any(-1).sum(-1).float().mean()) for l,v in old.items()]
   if masks[k] is not None:
    need(all(torch.isin(rs[l][k],torch.tensor(masks[k][str(l)])).all() for l in rs),"excluded expert")
   stats.append(dict(step=j,name=n,nmse=nmse,kl=max(0.,kl),argmax_equal=bool(torch.equal(a.argmax(-1),b.argmax(-1))),route_overlap=sum(overlap)/47,seconds=dt))
  print(json.dumps(dict(status="CUDA_FULL_FORWARD_STEP",id=r["id"],step=j,seconds=dt,peak_bytes=torch.cuda.max_memory_allocated())),flush=True)
 elapsed=time.monotonic()-start;cpu_seconds=sum(r["results"][n]["seconds"] for n in names)
 results.append(dict(id=r["id"],steps=stats,cuda_seconds=elapsed,cpu_reference_seconds=cpu_seconds,batched_speedup=cpu_seconds/elapsed))
report=dict(status="CUDA_FULL_FORWARD_DIAGNOSTIC_NOT_EXACT_ORACLE",teacher=ref["teacher"],reference_sha256=sha(raw),gpu=torch.cuda.get_device_name(),torch=torch.__version__,results=results,cache=engine.stats(),max_nmse=max(v["nmse"] for r in results for v in r["steps"]),all_argmax_equal=all(v["argmax_equal"] for r in results for v in r["steps"]),limits="Two fixed Fit prompts, three masks, nine steps each. GPU full-model numerical/throughput diagnostic only, no quality or CPU-oracle replacement. Cache transfers synchronous, no asynchronous preload.")
with (D/"hybrid-report.json").open("x") as f:json.dump(report,f,sort_keys=True,indent=2)
print(json.dumps({k:v for k,v in report.items() if k!="results"}),flush=True)
