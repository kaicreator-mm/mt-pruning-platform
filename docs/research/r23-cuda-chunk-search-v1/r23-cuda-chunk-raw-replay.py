"""Numerical replay from raw selected CUDA logits on CPU; not independent inference."""
import pathlib,json,hashlib,math,torch
P=pathlib.Path;D=P(__file__).parent
def need(v,m):
 if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(n):return json.loads((D/n).read_bytes())
torch.set_num_threads(1);torch.set_num_interop_threads(1)
plan=read("gpu-propagation-plan.json");report=read("gpu-propagation-report.json")
need(plan["status"]=="GPU_FIXED_FIT_ONLY" and report["status"]=="GPU_APPROXIMATE_PROPAGATION_COMPLETE_NOT_ADMISSION","scope")
need(len(plan["known"])==13 and len(plan["novel"])==19 and len(plan["rows"])==10 and len(plan["masks"])==35,"inventory")
names=sorted(set(plan["masks"])-{"teacher"});allrows=[]
for i,id in enumerate(plan["rows"]):
 rec=read(f"gpu-propagation-{i:02d}.json");p=D/f"gpu-propagation-{i:02d}.pt";need(sha(p.read_bytes())==rec["logits_sha256"],"raw hash")
 x=torch.load(p,map_location="cpu",weights_only=True);need(x["id"]==rec["id"]==id and x["direction"]==rec["direction"]==id.split(":")[1] and set(x["logits"])==set(plan["masks"]),"identity")
 t=x["logits"]["teacher"].double();need(t.ndim==2 and t.shape[0]==10 and bool(torch.isfinite(t).all()),"teacher")
 out={}
 for n in names:
  a=x["logits"][n].double();need(a.shape==t.shape and bool(torch.isfinite(a).all()),"candidate")
  z=[]
  for j in range(9):
   sl=slice(0,2) if j==0 else slice(j+1,j+2)
   u=t[sl];v=a[sl];q=u.log_softmax(-1);w=v.log_softmax(-1)
   m=dict(step=j,nmse=float((v-u).square().sum()/u.square().sum().clamp_min(1e-30)),kl=max(0.,float((q.exp()*(q-w)).sum(-1).mean())),argmax_mismatch=int(v[-1].argmax()!=u[-1].argmax()))
   old=rec["metrics"][n][j]
   need(m["step"]==old["step"] and m["argmax_mismatch"]==old["argmax_mismatch"] and all(abs(m[k]-old[k])<1e-11 for k in ("nmse","kl")),"raw metric")
   z.append(m)
  out[n]=z
 allrows.append(dict(direction=rec["direction"],metrics=out))
 print(json.dumps(dict(status="RAW_LOGITS_REPLAYED",id=id,candidates=len(names))),flush=True)
groups={}
for n in names:
 groups[n]={}
 for d in ("zh-my","my-zh"):
  for phase in ("prefill","decode"):
   z=[v for r in allrows if r["direction"]==d for v in r["metrics"][n] if (v["step"]==0)==(phase=="prefill")]
   need(len(z)==(8 if d=="zh-my" else 2)*(1 if phase=="prefill" else 8),"coverage")
   q=dict(nmse=math.fsum(v["nmse"] for v in z)/len(z),kl=math.fsum(v["kl"] for v in z)/len(z),argmax_mismatch=sum(v["argmax_mismatch"] for v in z),positions=len(z))
   old=report["groups"][n][d+"/"+phase];need(all(abs(q[k]-old[k])<1e-11 for k in q),"aggregate");groups[n][d+"/"+phase]=q
def score(n,gs):
 a=gs[n];b=gs["original96"];rr=[(a[g][k]-b[g][k])/max(b[g][k],1e-8) for g in a for k in ("nmse","kl")]
 return max(rr)+.25*math.fsum(max(0,x) for x in rr)
rank=sorted(plan["novel"],key=lambda n:(score(n,groups),n))
need(rank==report["rank_novel"],"rank")
cpu={}
for i,h in ((1,"97f923d7a20390cf77a664f2324a994fc0f2f92a660f04c156ed9a5bfb37101e"),(2,"938a1771adda06a6e08d698cb9d749b18b7a1a41028550caf7d07c594a77c322")):
 p=D/f"r23-gpu-wave{i}-replay.json";need(sha(p.read_bytes())==h,"exact replay")
 x=json.loads(p.read_bytes())
 for n,g in x["groups"].items():
  cpu[n]={k:v["candidate96"] for k,v in g.items()}
  if "original96" not in cpu:cpu["original96"]={k:v["original96"] for k,v in g.items()}
need(set(cpu)==set(plan["known"])|{"original96"},"calibration inventory")
known=plan["known"];a=sorted(known,key=lambda n:(score(n,groups),n));b=sorted(known,key=lambda n:(score(n,cpu),n))
rho=1-6*sum((a.index(n)-b.index(n))**2 for n in known)/(len(known)*(len(known)**2-1))
deltas=[]
for n in known:
 for g in cpu[n]:
  for k in ("nmse","kl"):
   x=groups[n][g][k]-groups["original96"][g][k];y=cpu[n][g][k]-cpu["original96"][g][k]
   deltas.append(dict(name=n,group=g,metric=k,gpu_delta=x,cpu_delta=y,same_sign=(x<=0)==(y<=0),absolute_error=abs(x-y)))
res=dict(status="RAW_GPU_LOGITS_REPLAY_PASS_APPROXIMATE_ONLY",raw_rows=10,groups=groups,rank_novel=rank,known_score_spearman=rho,known_directional_sign_agreement=sum(x["same_sign"] for x in deltas)/len(deltas),max_delta_error=max(x["absolute_error"] for x in deltas),known_gpu_rank=a,known_cpu_rank=b,known_delta_records=deltas,report_sha256=sha((D/"gpu-propagation-report.json").read_bytes()),limits="Recomputed raw GPU logits on CPU; no exact oracle equivalence, no automatic candidate acceptance.")
with (D/"raw-replay.json").open("x") as f:json.dump(res,f,indent=2,sort_keys=True)
print(json.dumps({k:v for k,v in res.items() if k not in ("groups","known_delta_records")}),flush=True)
