"""Causal whole fixed trajectory vs historical CPU chunked oracle; approximate only."""
import pathlib,hashlib,json,time,math,torch
import mtpruning.engine as E
from r23_cuda_hybrid_engine_v2 import Hybrid
P=pathlib.Path;R=P.home()/".cache/mt-pruning";D=P(__file__).parent
def need(v,m):
 if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def pin(p,h):
 b=p.read_bytes();need(sha(b)==h,"pin "+str(p));return b
need(torch.__version__=="2.5.1+cu121" and torch.cuda.is_available() and torch.cuda.get_device_capability()==(7,0),"V100")
refp=R/"r23-cuda-hybrid-v1/cpu-reference.pt"
raw=pin(refp,"13e8cf69c159fa2aa6b61860617947ac3ceb2f15f915203007bb9499bfd1b097")
ref=torch.load(refp,weights_only=True)
need(len(ref["rows"])==2 and ref["teacher"]=="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94","reference")
src=R/"hf_home/hub/models--tencent--Hy-MT3-30B-A3B/snapshots/08ae9acea91abe265c6236e757e49d8851e1ce06"
torch.set_num_threads(8);torch.cuda.reset_peak_memory_stats();engine=Hybrid(E.Checkpoint(src),mode='fp16')
names=["teacher","original96","l42"];masks=[ref["masks"][n] for n in names];results=[];rawout=[]
for r in ref["rows"]:
 prompt=r["prompt"];prefix=r["prefix"];ids=prompt+prefix;engine.new_sequence(3,len(ids))
 start=time.monotonic();out,rs=engine.forward(ids,masks);torch.cuda.synchronize();dt=time.monotonic()-start
 metrics=[];saved={}
 for k,n in enumerate(names):
  saved[n]=[]
  for j in range(9):
   rows=[len(prompt)//2-1,len(prompt)-1] if j==0 else [len(prompt)+j-1]
   a=out[k,rows].double();b=r["results"][n]["logits"][j].double();need(a.shape==b.shape and torch.isfinite(a).all(),"logits")
   nmse=float((a-b).square().sum()/b.square().sum().clamp_min(1e-30));lp=b.log_softmax(-1);lq=a.log_softmax(-1);kl=float((lp.exp()*(lp-lq)).sum(-1).mean())
   old={x["layer"]:x["ids"] for x in r["results"][n]["routes"] if x["step"]==j}
   overlap=[float((rs[l][k,rows,:,None]==v[:,None,:]).any(-1).sum(-1).float().mean()) for l,v in old.items()]
   if masks[k] is not None:need(all(torch.isin(rs[l][k],torch.tensor(masks[k][str(l)])).all() for l in rs),"excluded expert")
   metrics.append(dict(step=j,name=n,nmse=nmse,kl=max(0.,kl),argmax_equal=bool(torch.equal(a.argmax(-1),b.argmax(-1))),route_overlap=sum(overlap)/47))
   saved[n].append(out[k,rows].clone())
 cpu=sum(r["results"][n]["seconds"] for n in names)
 result=dict(id=r["id"],cuda_seconds=dt,cpu_seconds=cpu,speedup=cpu/dt,metrics=metrics)
 results.append(result);rawout.append(dict(id=r["id"],logits=saved));print(json.dumps(dict(id=r["id"],seconds=dt,speedup=cpu/dt,max_nmse=max(x["nmse"] for x in metrics))),flush=True)
with (D/"fp16-logits.pt").open("xb") as f:torch.save(rawout,f)
report=dict(status="CUDA_FP16_WHOLE_TRAJECTORY_DIAGNOSTIC_NOT_ORACLE",reference_sha256=sha(raw),logits_sha256=sha((D/"fp16-logits.pt").read_bytes()),results=results,cache=engine.stats(),max_nmse=max(v["nmse"] for r in results for v in r["metrics"]),all_argmax_equal=all(v["argmax_equal"] for r in results for v in r["metrics"]),limits="Fixed forced tokens with causal masking; whole-chunk vs stepwise CPU may differ numerically. No free-running quality or acceptance.")
with (D/"fp16-report.json").open("x") as f:json.dump(report,f,sort_keys=True,indent=2)
print(json.dumps({k:v for k,v in report.items() if k!="results"}),flush=True)
