"""Full causal fixed-Fit GPU propagation. Known-mask calibration and novel proposals.
No exact admission, no Holdout, no free-running inference."""
import pathlib,hashlib,json,itertools,time,math,torch,sys
import mtpruning.engine as E
from r23_cuda_hybrid_engine_v2 import Hybrid
P=pathlib.Path;R=P.home()/".cache/mt-pruning";D=P(__file__).parent
def need(v,m):
 if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def pin(p,h,git=False):
 b=p.read_bytes();z=hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest() if git else sha(b);need(z==h,"pin "+str(p));return b
def write(n,x):
 with (D/n).open("xb") as f:f.write(canon(x)+b"\n")
mode=sys.argv[1];need(mode in ("fp16","fp32"),"precision")
need(torch.__version__=="2.5.1+cu121" and torch.cuda.is_available() and torch.cuda.get_device_capability()==(7,0),"GPU")
pr=json.loads(pin(R/"r23-results/r23h-all8-one-swap-prefill-v1/plan.json","a80858f97ff201b3cf62f25c779ab0786bb6d0a60973204c6fc1f3dd1f8c700b"))
base=next(x["kept"] for x in pr["candidates"] if x["name"]=="l42_remove63_add120");need(sha(canon(base))=="f5d93021de52c065e0dddede1045d6410e0cbf6f645649731d1580626e66a6dc","base")
original=json.loads(pin(R/"r23h-pareto-screen-v1/original.json","d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed",True))["candidates"]["96"]["kept"]
known={}
for i,h in ((1,"97f923d7a20390cf77a664f2324a994fc0f2f92a660f04c156ed9a5bfb37101e"),(2,"938a1771adda06a6e08d698cb9d749b18b7a1a41028550caf7d07c594a77c322")):
 audit=json.loads(pin(D/f"r23-gpu-wave{i}-replay.json",h));need(audit["primary_gate_pass"]==[] and audit["captures"]==(60 if i==1 else 70),"audit")
 plan=json.loads((R/f"r23-pipeline-v2/wave{i}-plan.json").read_bytes())
 for c in plan["candidates"]:
  need(sha(canon(c["kept"]))==c["mask_sha256"]==audit["candidates"][c["name"]],"known mask");known[c["name"]]=c["kept"]
need(len(known)==13,"known inventory")
# Test all pair combinations among five exact near-gate single swaps, plus 3-layer combinations.
# L39 singleton has only one failed metric; L32/34/36/25 provide complementary changes.
swaps=[(39,18,19),(32,9,37),(34,61,100),(36,2,123),(25,1,115)]
novel={}
for k in (2,3):
 for v in itertools.combinations(swaps,k):
  m={l:x[:] for l,x in base.items()}
  for l,rem,add in v:
   l=str(l);need(rem in m[l] and add not in m[l],"swap");m[l]=sorted((set(m[l])-{rem})|{add})
  if sha(canon(m)) in {sha(canon(z)) for z in known.values()}:continue
  novel["joint_"+"_".join(str(x[0]) for x in v)]=m
need(len(novel)>=17 and len(novel)<=20 and len({sha(canon(z)) for z in novel.values()})==len(novel),"novel inventory")
for m in list(known.values())+list(novel.values())+[original,base]:
 need(set(m)==set(map(str,range(1,48))) and all(len(x)==96 and x==sorted(set(x)) and all(type(i) is int and 0<=i<128 for i in x) for x in m.values()),"mask")
pp=json.loads(pin(R/"r23g-recovery-ssd-38ed4798/prop-plan.json","cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde",True))
xp=json.loads(pin(R/"r23g-recovery-ssd-38ed4798/expanded.json","a1047d1a7c5624ed0e3a868bd5d76eef4c754915",True))
rows=pp["rows"]+[r for r in xp["fit"] if r["direction"]=="zh-my" and r["id"] not in {z["id"] for z in pp["rows"]}]
rows=[r for r in rows if r["direction"] in ("zh-my","my-zh")]
need(len(rows)==10 and not ({r["line"] for r in rows}&{r["line"] for r in xp["holdout"]}),"Fit")
raw=pin(R/"run-d01b-r19/corpus/corpus.jsonl","9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4")
ctx={r["id"]:r for r in map(json.loads,raw.splitlines()) if r["split"]=="calibration"}
src=R/"hf_home/hub/models--tencent--Hy-MT3-30B-A3B/snapshots/08ae9acea91abe265c6236e757e49d8851e1ce06"
engine=Hybrid(E.Checkpoint(src),mode=mode)
masks={"teacher":None,"original96":original,"l42":base,**known,**novel}
names=list(masks)
write("gpu-propagation-plan.json",dict(status="GPU_FIXED_FIT_ONLY",mode=mode,rows=[r["id"] for r in rows],known=list(known),novel=list(novel),masks={n:None if m is None else sha(canon(m)) for n,m in masks.items()},novel_masks=novel))
allrows=[]
for ri,r in enumerate(rows):
 prompt=ctx[r["id"]]["prompt_ids"];prefix=r["teacher_prefix_tokens"];need(sha(canon(prompt))==r["prompt_ids_sha256"] and len(prefix)==8 and len(prompt)+8<=384,"input")
 ids=prompt+prefix;positions=[len(prompt)//2-1,len(prompt)-1]+list(range(len(prompt),len(ids)))
 saved={}
 for i in range(0,len(names),3):
  ns=names[i:i+3];mm=[masks[n] for n in ns];start=time.monotonic()
  engine.new_sequence(len(ns),len(ids));out,rs=engine.forward(ids,mm);torch.cuda.synchronize()
  for j,n in enumerate(ns):
   if mm[j] is not None:need(all(torch.isin(rs[l][j],torch.tensor(mm[j][str(l)])).all() for l in rs),"excluded expert")
   saved[n]=out[j,positions].clone()
  print(json.dumps(dict(status="GPU_PROPAGATION_BATCH",id=r["id"],names=ns,seconds=time.monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated())),flush=True)
  del out,rs
 t=saved["teacher"].double();lp=t.log_softmax(-1)
 metrics={}
 for n in names[1:]:
  a=saved[n].double();lq=a.log_softmax(-1);z=[]
  for j in range(9):
   ii=[0,1] if j==0 else [j+1]
   nmse=float((a[ii]-t[ii]).square().sum()/t[ii].square().sum().clamp_min(1e-30))
   kl=float((lp[ii].exp()*(lp[ii]-lq[ii])).sum(-1).mean())
   z.append(dict(step=j,nmse=nmse,kl=max(0.,kl),argmax_mismatch=int(a[ii[-1]].argmax()!=t[ii[-1]].argmax())))
  metrics[n]=z
 # Persist selected raw GPU logits for independent numerical replay.
 outp=D/f"gpu-propagation-{ri:02d}.pt"
 with outp.open("xb") as f:torch.save(dict(id=r["id"],direction=r["direction"],logits=saved),f)
 rec=dict(id=r["id"],direction=r["direction"],metrics=metrics,logits_sha256=sha(outp.read_bytes()))
 write(f"gpu-propagation-{ri:02d}.json",rec);allrows.append(rec);del saved,t,lp
 print(json.dumps(dict(status="GPU_PROPAGATION_ROW_COMPLETE",id=r["id"],index=ri,receipts=len(metrics))),flush=True)
groups={}
for n in names[1:]:
 groups[n]={}
 for d in ("zh-my","my-zh"):
  for phase in ("prefill","decode"):
   z=[v for r in allrows if r["direction"]==d for v in r["metrics"][n] if (v["step"]==0)==(phase=="prefill")]
   groups[n][d+"/"+phase]=dict(nmse=math.fsum(v["nmse"] for v in z)/len(z),kl=math.fsum(v["kl"] for v in z)/len(z),argmax_mismatch=sum(v["argmax_mismatch"] for v in z),positions=len(z))
def score(n):
 a=groups[n];b=groups["original96"];rat=[(a[g][k]-b[g][k])/max(b[g][k],1e-8) for g in a for k in ("nmse","kl")]
 return max(rat)+.25*math.fsum(max(0,x) for x in rat)
rank=sorted(novel,key=lambda n:(score(n),n))
write("gpu-propagation-report.json",dict(status="GPU_APPROXIMATE_PROPAGATION_COMPLETE_NOT_ADMISSION",mode=mode,groups=groups,rank_novel=rank,scores={n:score(n) for n in rank},cache=engine.stats(),limits="Fixed Fit GPU-only approximate whole-trajectory propagation; historical CPU metrics may disagree. Ranking not a pass; independent exact CPU verification required."))
print(json.dumps(dict(status="GPU_PROPAGATION_COMPLETE",known=len(known),novel=len(novel),rows=len(rows),top=rank[:4])),flush=True)
