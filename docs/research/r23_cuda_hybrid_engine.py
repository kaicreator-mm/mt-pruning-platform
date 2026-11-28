"""Bounded CUDA full-model prototype. CPU mmap weights; batched independent masks.
FP32 CUDA GEMMs and explicit BF16 rounding, not an exact CPU replacement."""
from collections import OrderedDict
import torch
import torch.nn.functional as F
import mtpruning.engine as E
GiB=1<<30
def need(v,m):
 if not v:raise ValueError(m)
class WeightCache:
 def __init__(self,budget):
  self.budget=budget;self.data=OrderedDict();self.bytes=0;self.hit=0;self.miss=0;self.transfer=0;self.evictions=0
 def get(self,t,fp32=True):
  key=(t.data_ptr(),tuple(t.shape),str(t.dtype),fp32)
  if key in self.data:
   self.hit+=1;self.data.move_to_end(key);return self.data[key]
  self.miss+=1;n=t.numel()*(4 if fp32 else t.element_size())
  need(n<=self.budget,"single tensor exceeds cache")
  while self.bytes+n>self.budget:
   _,v=self.data.popitem(last=False);self.bytes-=v.numel()*v.element_size();self.evictions+=1
  # Transfer BF16 source directly, expand on GPU (avoids large CPU copies).
  v=t.to(device="cuda",dtype=torch.float32 if fp32 else t.dtype)
  need(v.is_cuda,"cuda cache");self.data[key]=v;self.bytes+=n;self.transfer+=t.numel()*t.element_size();return v
 def stats(self):
  return dict(bytes=self.bytes,hits=self.hit,misses=self.miss,evictions=self.evictions,source_bytes=self.transfer)
class Hybrid:
 def __init__(self,checkpoint,static_budget=6*GiB,expert_budget=4*GiB):
  self.cpu=E.Engine(checkpoint,threads=8);self.static=WeightCache(static_budget);self.experts=WeightCache(expert_budget)
  self.cfg=self.cpu;self.kv=None;self.pos=0
  torch.backends.cuda.matmul.allow_tf32=False
 def new_sequence(self,B,capacity):
  c=self.cfg;need(B<=8 and 0<capacity<=256,"bounded batch/context")
  self.kv=[(torch.empty((B,c.n_kv,capacity,c.head_dim),device="cuda",dtype=torch.bfloat16),torch.empty((B,c.n_kv,capacity,c.head_dim),device="cuda",dtype=torch.bfloat16)) for _ in c.layers];self.pos=0
 def linear(self,x,w,expert=False):
  cache=self.experts if expert else self.static
  return F.linear(x.float(),cache.get(w)).bfloat16()
 def norm(self,x,w):
  return E.rms_norm(x,self.static.get(w,False),self.cfg.eps)
 def mlp(self,x,weights,expert=False):
  g,u,d=weights
  a=self.linear(x,g,expert);b=self.linear(x,u,expert)
  return self.linear((F.silu(a.float()).bfloat16()*b).bfloat16(),d,expert)
 def route(self,l,x,denied):
  c=self.cfg;scores=torch.sigmoid(F.linear(x.float(),self.static.get(l.router)))
  choice=scores+self.static.get(l.bias)
  top=(choice+denied[:,None,:]).topk(c.top_k,dim=-1,sorted=False).indices
  w=scores.gather(-1,top);w=w/(w.sum(-1,keepdim=True)+1e-20)*c.router_scaling
  return top,w
 @torch.inference_mode()
 def forward(self,ids,masks):
  c=self.cfg;B=len(masks);T=len(ids);start=self.pos;end=start+T
  need(self.kv is not None and self.kv[0][0].shape[0]==B and end<=self.kv[0][0].shape[2],"cache shape")
  ids=torch.tensor(ids,dtype=torch.long);h=c.embed[ids].to("cuda")[None,:,:].expand(B,-1,-1).contiguous()
  cos,sin=c._rope(torch.arange(start,end));cos=cos.cuda()[None,None,:,:];sin=sin.cuda()[None,None,:,:]
  causal=torch.full((T,end),-torch.inf,device="cuda").triu(diagonal=start+1) if T>1 else None
  routes={}
  for l in c.layers:
   x=self.norm(h,l.ln1)
   q=self.linear(x,l.wq).view(B,T,c.n_heads,c.head_dim).transpose(1,2)
   k=self.linear(x,l.wk).view(B,T,c.n_kv,c.head_dim).transpose(1,2)
   v=self.linear(x,l.wv).view(B,T,c.n_kv,c.head_dim).transpose(1,2)
   q=self.norm(q,l.q_norm);k=self.norm(k,l.k_norm)
   q=q*cos+E.rotate_half(q)*sin;k=k*cos+E.rotate_half(k)*sin
   keys,values=self.kv[l.idx];keys[:,:,start:end,:]=k;values[:,:,start:end,:]=v
   keys=keys[:,:,:end,:].repeat_interleave(c.n_rep,dim=1);values=values[:,:,:end,:].repeat_interleave(c.n_rep,dim=1)
   a=torch.matmul(q.float(),keys.float().transpose(-1,-2)).bfloat16()*c.scaling
   if causal is not None:a=a+causal
   a=F.softmax(a,dim=-1,dtype=torch.float32).bfloat16()
   attn=torch.matmul(a.float(),values.float()).bfloat16()
   h=h+self.linear(attn.transpose(1,2).reshape(B,T,c.attention_width),l.wo)
   x=self.norm(h,l.ln2)
   if l.sparse:
    denied=torch.zeros((B,c.ckpt.num_experts),device="cuda",dtype=torch.float32)
    for j,m in enumerate(masks):
     if m is not None:
      allowed=m[str(l.idx)];need(len(allowed)==96 and len(set(allowed))==96 and min(allowed)>=0 and max(allowed)<128,"mask")
      denied[j].fill_(-torch.inf);denied[j,allowed]=0
    top,w=self.route(l,x,denied);routes[l.idx]=top.detach().cpu()
    flat=top.flatten();xx=x.reshape(B*T,-1);y=torch.empty((flat.numel(),c.hidden),device="cuda",dtype=torch.float32)
    for expert in flat.unique().tolist():
     slots=(flat==expert).nonzero().flatten();z=xx[slots//c.top_k]
     y[slots]=self.mlp(z,(l.gate[expert],l.up[expert],l.down[expert]),True).float()
    routed=(y.view(B,T,c.top_k,c.hidden)*w[:,:,:,None]).sum(2).bfloat16()
    h=h+(routed+self.mlp(x,l.shared))
   else:h=h+self.mlp(x,l.dense)
  self.pos=end
  out=self.linear(self.norm(h,c.final_norm),c.lm_head)
  need(out.is_cuda and torch.isfinite(out).all(),"finite cuda output")
  return out.cpu(),routes
 def stats(self):
  return dict(static=self.static.stats(),experts=self.experts.stats(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
