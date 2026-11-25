"""Locally trained synthetic two-layer top-2 MoE; no natural language claims."""
import json, torch
from torch import nn
from trace import compare, stability
class Block(nn.Module):
    def __init__(self,n):
        super().__init__(); self.router=nn.Linear(12,n,bias=False); self.experts=nn.Parameter(torch.randn(n,12,12)*.15)
    def forward(self,x):
        v,i=self.router(x).topk(2); w=v.softmax(-1); y=torch.einsum('bi,bkij->bkj',x,self.experts[i]); return x+(y*w.unsqueeze(-1)).sum(1),i,w
class Net(nn.Module):
    def __init__(self,n=8):
        super().__init__(); self.blocks=nn.ModuleList([Block(n),Block(n)]); self.out=nn.Linear(12,4)
    def forward(self,x):
        traces=[]
        for b in self.blocks:
            x,i,w=b(x); traces.append((i,w))
        return self.out(x),traces

def run():
    torch.manual_seed(42); torch.set_num_threads(1); x=torch.randn(1024,12); teacher_matrix=torch.randn(12,4); y=(x@teacher_matrix).argmax(-1); m=Net(); opt=torch.optim.AdamW(m.parameters(),lr=.015)
    for _ in range(100):
        ix=torch.randint(0,640,(64,)); loss=nn.functional.cross_entropy(m(x[ix])[0],y[ix]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _,cal=m(x[640:832]); masks=[]; rows=[]; p=Net(4); p.out.load_state_dict(m.out.state_dict())
        for b,c,(i,w) in zip(m.blocks,p.blocks,cal):
            a=torch.zeros(len(i),8).scatter_add_(1,i,w); ids=sorted(a.sum(0).topk(4).indices.tolist()); masks.append(ids); rows.append(a.tolist()); c.router.weight.copy_(b.router.weight[ids]); c.experts.copy_(b.experts[ids])
        logits,t=m(x[832:]); small,s=p(x[832:]); result={'teacher_accuracy':float((logits.argmax(-1)==y[832:]).float().mean()),'pruned_accuracy':float((small.argmax(-1)==y[832:]).float().mean()),'teacher_params':sum(v.numel() for v in m.parameters()),'pruned_params':sum(v.numel() for v in p.parameters()),'layers':[]}
        for l,((i,_),(j,_),ids,r) in enumerate(zip(t,s,masks,rows)):
            result['layers'].append({'layer':l,'keep':ids,'drift':compare([{'key':str(k),'ids':v} for k,v in enumerate(i.tolist())],[{'key':str(k),'ids':v} for k,v in enumerate(j.tolist())],ids),'stability':stability(r,[8,32,96],4,[1,2,3,4,5])})
    print(json.dumps(result,indent=2)); return result
if __name__=='__main__': run()
