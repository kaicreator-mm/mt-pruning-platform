import pathlib,hashlib,json,time,math,torch
from safetensors import safe_open
P=pathlib.Path;D=P(__file__).parent;R=P.home()/".cache/mt-pruning"
src=R/"hf_home/hub/models--tencent--Hy-MT3-30B-A3B/snapshots/08ae9acea91abe265c6236e757e49d8851e1ce06"
def need(x,m):
 if not x:raise ValueError(m)
need(torch.__version__=="2.5.1+cu121" and torch.cuda.is_available() and torch.cuda.get_device_capability()==(7,0),"CUDA V100")
torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
wfile=D/"witnesses.pt";raw=wfile.read_bytes();x=torch.load(wfile,weights_only=True);need(x["teacher"]=="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94" and len(x["rows"])==20,"CPU witnesses")
index=json.loads((src/"model.safetensors.index.json").read_bytes())["weight_map"];handles={}
def tensor(n):
 if index[n] not in handles:handles[index[n]]=safe_open(src/index[n],framework="pt",device="cpu")
 return handles[index[n]].get_tensor(n)
def sync():torch.cuda.synchronize()
def calc(z,top,w,cache):
 y=torch.empty((top.numel(),z.shape[1]),device="cuda",dtype=torch.float32)
 flat=top.flatten()
 for expert,(g,u,d) in cache.items():
  slots=(flat==expert).nonzero().flatten()
  if not len(slots):continue
  rows=slots//top.shape[1]
  # FP32 GEMM with explicit BF16 rounding at every oracle operation boundary.
  a=torch.nn.functional.linear(z[rows],g).bfloat16()
  b=torch.nn.functional.linear(z[rows],u).bfloat16()
  h=(torch.nn.functional.silu(a.float()).bfloat16()*b).bfloat16()
  y[slots]=torch.nn.functional.linear(h.float(),d).bfloat16().float()
 return (y.view(*top.shape,-1)*w[:,:,None]).sum(1).bfloat16()
results=[]
for row in x["rows"]:
 l=row["layer"];z=row["x"].float().cuda();top=row["top"].cuda();w=row["weights"].cuda();ref=row["reference"]
 sync();t=time.perf_counter()
 cache={int(e):tuple(tensor(f"model.layers.{l}.mlp.experts.{int(e)}.{n}_proj.weight").float().cuda() for n in ("gate","up","down")) for e in row["top"].unique()}
 sync();transfer=time.perf_counter()-t
 y=calc(z,top,w,cache);sync()
 t=time.perf_counter()
 for _ in range(20):y=calc(z,top,w,cache)
 sync();sec=(time.perf_counter()-t)/20
 cpu=y.cpu();need(cpu.shape==ref.shape and torch.isfinite(cpu).all(),"finite result")
 nmse=float((cpu.double()-ref.double()).square().sum()/ref.double().square().sum().clamp_min(1e-30))
 results.append(dict(id=row["id"],layer=l,phase=row["phase"],tokens=len(z),experts=len(cache),nmse=nmse,bit_exact=bool(torch.equal(cpu,ref)),exact_fraction=float((cpu==ref).float().mean()),transfer_seconds=transfer,resident_seconds=sec))
 print(json.dumps(results[-1]),flush=True);del cache,z,top,w,y
report=dict(status="CUDA_EXPERT_KERNEL_DIAGNOSTIC_NOT_ORACLE_REPLACEMENT",teacher=x["teacher"],witness_sha256=hashlib.sha256(raw).hexdigest(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,rows=results,all_bit_exact=all(v["bit_exact"] for v in results),max_nmse=max(v["nmse"] for v in results),peak_allocated_bytes=torch.cuda.max_memory_allocated(),limits="Real Fit hidden inputs, selected routed expert FFN only; CPU-vs-CUDA numerical comparison, no full-model equivalence or throughput claim. No candidate promotion.")
with (D/"gpu-report.json").open("x") as f:json.dump(report,f,sort_keys=True,indent=2)
print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
