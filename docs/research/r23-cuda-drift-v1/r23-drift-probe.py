"""FP32 CUDA diagnostic: local operator errors, exact route anchoring, free propagation."""
import pathlib,hashlib,json,math,time,torch,torch.nn.functional as F
import mtpruning.engine as E
from r23_cuda_hybrid_engine_v2 import Hybrid
P=pathlib.Path;D=P(__file__).parent;R=P.home()/".cache/mt-pruning"
def need(v,m):
 if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def met(a,b):
 a=a.cpu().double();b=b.cpu().double();need(a.shape==b.shape and bool(torch.isfinite(a).all() and torch.isfinite(b).all()),"finite shape")
 return dict(nmse=float((a-b).square().sum()/b.square().sum().clamp_min(1e-30)),bit_equal=bool(torch.equal(a,b)),equal_fraction=float((a==b).float().mean()),max_abs=float((a-b).abs().max()))
def overlap(a,b):
 a=a.cpu();b=b.cpu();need(a.shape==b.shape,"route shape");z=(a[:,:,None]==b[:,None,:]).any(-1).sum(-1)
 return dict(mean=float(z.double().mean()),exact_sets=int((z==a.shape[-1]).sum()),tokens=len(z))
def write(n,x):
 with (D/n).open("x") as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False)
need(torch.__version__=="2.5.1+cu121" and torch.cuda.is_available() and torch.cuda.get_device_capability()==(7,0),"V100")
torch.set_num_threads(8)
src=R/"hf_home/hub/models--tencent--Hy-MT3-30B-A3B/snapshots/08ae9acea91abe265c6236e757e49d8851e1ce06"
eng=Hybrid(E.Checkpoint(src),mode="fp32");c=eng.cfg
need(len(c.layers)==48 and c.top_k==8,"architecture")
def attn(l,h,cos,sin,causal):
 T=len(h);x=eng.norm(h,l.ln1)
 q=eng.linear(x,l.wq).view(T,c.n_heads,c.head_dim).transpose(0,1)
 k=eng.linear(x,l.wk).view(T,c.n_kv,c.head_dim).transpose(0,1)
 v=eng.linear(x,l.wv).view(T,c.n_kv,c.head_dim).transpose(0,1)
 qn=eng.norm(q,l.q_norm);kn=eng.norm(k,l.k_norm)
 qr=qn*cos+E.rotate_half(qn)*sin;kr=kn*cos+E.rotate_half(kn)*sin
 keys=kr.repeat_interleave(c.n_rep,dim=0);values=v.repeat_interleave(c.n_rep,dim=0)
 a=torch.matmul(qr.float(),keys.float().transpose(-1,-2)).bfloat16()*c.scaling
 a=F.softmax(a+causal,dim=-1,dtype=torch.float32).bfloat16()
 at=torch.matmul(a.float(),values.float()).bfloat16()
 y=h+eng.linear(at.transpose(0,1).reshape(T,c.attention_width),l.wo)
 return y,dict(ln1_out=x,q_norm=q,k_norm=k,q_norm_out=qn,k_norm_out=kn,ln2=y)
def cpu_route(l,x):
 return c._route(l,x.cpu(),"prefill")
def cuda_route(l,x):
 top,w=eng.route(l,x[None],torch.zeros((1,c.ckpt.num_experts),device="cuda"));return top[0],w[0]
def experts(l,x,top,w):
 top=top.cuda();w=w.cuda();flat=top.flatten()
 y=torch.empty((flat.numel(),c.hidden),device="cuda",dtype=torch.float32)
 for expert in flat.unique().tolist():
  sl=(flat==expert).nonzero().flatten();z=x[sl//c.top_k]
  y[sl]=eng.mlp(z,(l.gate[expert],l.up[expert],l.down[expert]),True).float()
 return (y.view(len(x),c.top_k,c.hidden)*w[:,:,None]).sum(1).bfloat16()+eng.mlp(x,l.shared)
reports=[]
for i in (0,2):
 p=D/f"cpu-layer-{i:02}.pt";x=torch.load(p,map_location="cpu",weights_only=True)
 need(x["teacher"]=="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94" and set(x["layers"])==set(range(48)),"witness")
 ids=x["ids"];T=len(ids);cos,sin=c._rope(torch.arange(T));cos=cos.cuda()[None];sin=sin.cuda()[None]
 causal=torch.full((T,T),-torch.inf,device="cuda").triu(diagonal=1)
 hn=c.embed[torch.tensor(ids)].cuda();hc=hn.clone();rows=[]
 start=time.monotonic()
 for l in c.layers:
  ref=x["layers"][l.idx];h=ref["ln1"].cuda();a,st=attn(l,h,cos,sin,causal)
  stage={k:met(v,ref[k]) for k,v in st.items()}
  z=eng.norm(ref["ln2"].cuda(),l.ln2);stage["ln2_out_anchored"]=met(z,ref["ln2_out"])
  xx=ref["ln2_out"].cuda()
  yn,an=attn(l,hn,cos,sin,causal);yc,ac=attn(l,hc,cos,sin,causal)
  xn=eng.norm(yn,l.ln2);xc=eng.norm(yc,l.ln2)
  if l.sparse:
   top,w=cuda_route(l,xx);ct,cw=cpu_route(l,xx)
   need(torch.equal(ct,ref["top"]) and torch.equal(cw,ref["weights"]),"CPU route replay")
   forced=experts(l,xx,ref["top"],ref["weights"]);natural=experts(l,xx,top,w)
   stage["moe_forced_anchored"]=met(forced,ref["moe"]);stage["moe_native_anchored"]=met(natural,ref["moe"])
   stage["router_anchored_overlap"]=overlap(top,ct)
   stage["router_weight_error"]=met(w.sort(-1).values,cw.sort(-1).values)
   nt,nw=cuda_route(l,xn);ct,cw=cpu_route(l,xc)
   hn=yn+experts(l,xn,nt,nw);hc=yc+experts(l,xc,ct,cw)
   stage["native_route_overlap"]=overlap(nt,ref["top"]);stage["cpu_router_route_overlap"]=overlap(ct,ref["top"])
   stage["layer_anchored"]=met(ref["ln2"].cuda()+forced,ref["output"])
  else:
   stage["layer_anchored"]=met(ref["ln2"].cuda()+eng.mlp(xx,l.dense),ref["output"])
   hn=yn+eng.mlp(xn,l.dense);hc=yc+eng.mlp(xc,l.dense)
  stage["free_native"]=met(hn,ref["output"]);stage["free_cpu_router"]=met(hc,ref["output"])
  rows.append(dict(layer=l.idx,metrics=stage))
  print(json.dumps(dict(status="LAYER_DIAGNOSTIC",id=x["id"],layer=l.idx,native=stage["free_native"]["nmse"],cpu_router=stage["free_cpu_router"]["nmse"],anchored=stage["layer_anchored"]["nmse"])),flush=True)
 positions=[T//2-1,T-1]
 logits={}
 for n,h in (("native",hn),("cpu_router",hc),("cpu_anchored",x["final"]["final"].cuda())):
  v=eng.linear(eng.norm(h[positions],c.final_norm),c.lm_head).cpu();q=met(v,x["logits"]);q["argmax_equal"]=bool(torch.equal(v.argmax(-1),x["logits"].argmax(-1)));logits[n]=q
 report=dict(id=x["id"],cpu_witness_sha256=sha(p.read_bytes()),layers=rows,logits=logits,seconds=time.monotonic()-start)
 write(f"drift-{i:02}.json",report);reports.append(report)
 print(json.dumps(dict(status="DRIFT_ROW_COMPLETE",id=x["id"],logits=logits,seconds=report["seconds"])),flush=True)
write("drift-report.json",dict(status="GPU_LAYER_DRIFT_DIAGNOSTIC_NOT_ORACLE",rows=reports,cache=eng.stats(),limits="Two fixed teacher prefill trajectories only. CPU-anchored local stages and native/CPU-router free propagation; no candidate quality or oracle admission."))
