import pathlib,json,hashlib,time,torch,math
from safetensors import safe_open
P=pathlib.Path;D=P(__file__).parent;R=P.home()/".cache/mt-pruning"
def need(v,m):
 if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def pin(p,h):
 b=p.read_bytes();need(sha(b)==h,"pin "+str(p));return b
def put(n,x):
 with (D/n).open("xb") as f:f.write(canon(x)+b"\n")
need(torch.__version__=="2.5.1+cu121" and torch.cuda.is_available() and torch.cuda.get_device_capability()==(7,0),"V100")
torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
wf=D/"search-witnesses.pt";w=torch.load(wf,weights_only=True);need(w["teacher"]=="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94" and len(w["rows"])==540,"witness inventory")
replay=json.loads(pin(D/"replay.json","51a037ce691ef71d1417fe655d76dde14090b3f854d4789007831074dce4118c"));need(replay["primary_gate_pass"]==[] and replay["report_sha256"]=="f6519ae29eca0680a222d541b5206ba871351098ccf1d7aea8b19881d7062477","previous rejection")
p=json.loads(pin(R/"r23-results/r23h-all8-one-swap-prefill-v1/plan.json","a80858f97ff201b3cf62f25c779ab0786bb6d0a60973204c6fc1f3dd1f8c700b"))
base=next(x["kept"] for x in p["candidates"] if x["name"]=="l42_remove63_add120")
src=R/"hf_home/hub/models--tencent--Hy-MT3-30B-A3B/snapshots/08ae9acea91abe265c6236e757e49d8851e1ce06"
cfg=json.loads((src/"config.json").read_bytes());sc=float(cfg["router_scaling_factor"]);idx=json.loads((src/"model.safetensors.index.json").read_bytes())["weight_map"];handles={}
def tensor(n):
 if idx[n] not in handles:handles[idx[n]]=safe_open(src/idx[n],framework="pt",device="cpu")
 return handles[idx[n]].get_tensor(n)
# Search new layers only. Score all 96x32 swaps at each layer on real teacher states.
# This is a local proxy; cannot establish propagated logits or translation quality.
layers=[25,26,27,28,32,34,35,36,39,41,44]
rank=[]
for l in layers:
 rows=[x for x in w["rows"] if x["layer"]==l];need(len(rows)==36,"layer witnesses")
 z=torch.cat([x["x"] for x in rows]).float().cuda();scores=torch.cat([x["scores"] for x in rows]).cuda();choice=torch.cat([x["choice"] for x in rows]).cuda();ref=torch.cat([x["reference"] for x in rows]).float().cuda()
 groups=[]
 for d in ("zh-my","my-zh"):
  for phase in ("prefill","decode"):
   tags=[(x["id"].split(":")[1],x["phase"]) for x in rows for _ in range(len(x["x"]))]
   a=torch.tensor([i for i,t in enumerate(tags) if t==(d,phase)],device="cuda");need(len(a)>0,"group");groups.append(a)
 vals=[]
 for e in range(128):
  g,u,down=[tensor(f"model.layers.{l}.mlp.experts.{e}.{n}_proj.weight").float().cuda() for n in ("gate","up","down")]
  a=torch.nn.functional.linear(z,g).bfloat16();b=torch.nn.functional.linear(z,u).bfloat16()
  h=(torch.nn.functional.silu(a.float()).bfloat16()*b).bfloat16()
  vals.append(torch.nn.functional.linear(h.float(),down).bfloat16().float())
  del g,u,down,a,b,h
 v=torch.stack(vals,dim=1);del vals
 # Full routed teacher check: exact CPU routing IDs with CUDA FFN.
 t=torch.cat([x["top"] for x in rows]).cuda();wt=torch.cat([x["weights"] for x in rows]).cuda()
 ar=torch.arange(len(z),device="cuda")[:,None]
 full=(v[ar,t]*wt[:,:,None]).sum(1).bfloat16().float()
 drift=float((full-ref).square().sum()/ref.square().sum().clamp_min(1e-30));need(math.isfinite(drift) and drift<=1e-5,"CUDA routed drift")
 kept=base[str(l)]
 def losses(masks):
  allowed=torch.tensor(masks,device="cuda",dtype=torch.bool)
  ch=choice[None,:,:].expand(len(masks),-1,-1).masked_fill(~allowed[:,None,:],-torch.inf)
  top=ch.topk(8,dim=-1,sorted=False).indices
  weights=scores[None,:,:].expand(len(masks),-1,-1).gather(2,top);weights=weights/(weights.sum(2,keepdim=True)+1e-20)*sc
  pred=(v[ar[None,:,:],top]*weights[:,:,:,None]).sum(2).bfloat16().float()
  return torch.stack([(pred[:,a]-ref[a]).square().sum((1,2))/ref[a].square().sum().clamp_min(1e-30) for a in groups],dim=1).cpu().tolist()
 mask=[i in kept for i in range(128)];baseline=losses([mask])[0];items=[]
 pairs=[(rem,add) for rem in kept for add in range(128) if add not in kept]
 for start in range(0,len(pairs),16):
  block=pairs[start:start+16];ms=[]
  for rem,add in block:
   a=mask[:];a[rem]=False;a[add]=True;ms.append(a)
  for (rem,add),q in zip(block,losses(ms)):
   ratios=[(x-y)/max(y,1e-8) for x,y in zip(q,baseline)]
   # Balance both directions and emphasize decode while penalizing worst regression.
   score=max(ratios)+0.5*(ratios[1]+ratios[3])+0.15*(ratios[0]+ratios[2])
   need(math.isfinite(score),"finite score")
   items.append(dict(layer=l,remove=rem,add=add,score=score,ratios=ratios,losses=q))
 items.sort(key=lambda x:(x["score"],x["remove"],x["add"]))
 rank.extend(items[:8]);print(json.dumps(dict(layer=l,full_route_drift=drift,baseline=baseline,best=items[:3],evaluated=len(items))),flush=True)
 del v,z,scores,choice,ref,full,t,wt
 torch.cuda.empty_cache()
rank.sort(key=lambda x:(x["score"],x["layer"],x["remove"],x["add"]))
# Two disjoint finite waves, max one swap per layer in each wave.
seen=set();chosen=[]
for item in rank:
 l=item["layer"]
 if l not in seen:chosen.append(item);seen.add(l)
need(len(chosen)==11,"one per layer")
# First six and remaining five layer candidates, plus two novel cross-layer combinations.
candidates=[]
def add(name,items):
 m={k:v[:] for k,v in base.items()}
 for it in items:
  l=str(it["layer"]);need(it["remove"] in m[l] and it["add"] not in m[l],"swap");m[l]=sorted((set(m[l])-{it["remove"]})|{it["add"]})
 candidates.append(dict(name=name,kept=m,mask_sha256=sha(canon(m)),changes=[{k:it[k] for k in ("layer","remove","add")} for it in items]))
for it in chosen:add("gpu_l%d_%d_%d"%(it["layer"],it["remove"],it["add"]),[it])
add("gpu_joint_best2",chosen[:2]);add("gpu_joint_best3",chosen[:3])
need(len({x["mask_sha256"] for x in candidates})==13,"distinct masks")
put("gpu-search-report.json",dict(status="GPU_LOCAL_PROXY_SEARCH_COMPLETE_NOT_ADMISSION",teacher=w["teacher"],witness_sha256=sha(wf.read_bytes()),rank=rank,evaluated=11*96*32,peak_bytes=torch.cuda.max_memory_allocated(),groups=["zh-my/prefill","zh-my/decode","my-zh/prefill","my-zh/decode"]))
for wave,cs in enumerate((candidates[:6],candidates[6:]),1):
 put(f"wave{wave}-plan.json",dict(status="GPU_PROPOSED_FIXED_FIT_ONLY",base_mask_sha256=sha(canon(base)),parent_report_sha256=replay["report_sha256"],search_report_sha256=sha((D/"gpu-search-report.json").read_bytes()),candidates=cs,wave=wave))
print(json.dumps(dict(status="GPU_SEARCH_COMPLETE",candidates=13,waves=[6,7],evaluated=11*96*32)),flush=True)
