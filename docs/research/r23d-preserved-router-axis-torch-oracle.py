"""CPU-only PyTorch witness: compact FFN with unchanged router axis.
Synthetic tensors; no teacher model, no native llama.cpp equivalence claim.
"""
import json, torch
import torch.nn.functional as F

def run(x, router, bias, gate, up, down, ids, k, preserve_axis):
    n=router.shape[0]
    if preserve_axis:
        scores=torch.sigmoid(F.linear(x.float(),router));choice=scores+bias
        allowed=torch.zeros(n,dtype=torch.bool);allowed[ids]=True
        top=torch.topk(choice.masked_fill(~allowed,-float('inf')),k,dim=-1,sorted=False).indices
        weights=scores.gather(1,top)
        lut=torch.full((n,),-1,dtype=torch.long);lut[ids]=torch.arange(len(ids));local=lut[top]
        assert (local>=0).all()
    else:
        scores=torch.sigmoid(F.linear(x.float(),router[ids]));local=torch.topk(scores+bias[ids],k,dim=-1,sorted=False).indices
        top=ids[local];weights=scores.gather(1,local)
    weights=weights/(weights.sum(dim=-1,keepdim=True)+1e-20)*2.826
    out=torch.zeros_like(x)
    for row in range(x.shape[0]):
        for j in range(k):
            i=int(local[row,j]);a=F.silu(F.linear(x[row:row+1],gate[i]))*F.linear(x[row:row+1],up[i]);out[row]+=F.linear(a,down[i])[0]*weights[row,j]
    return top,weights,out

def teacher(x,router,bias,gate,up,down,ids,k):
    n=router.shape[0];scores=torch.sigmoid(F.linear(x.float(),router));allowed=torch.zeros(n,dtype=torch.bool);allowed[ids]=True
    top=torch.topk((scores+bias).masked_fill(~allowed,-float('inf')),k,dim=-1,sorted=False).indices
    w=scores.gather(1,top);w=w/(w.sum(dim=-1,keepdim=True)+1e-20)*2.826
    y=torch.zeros_like(x)
    for row in range(x.shape[0]):
        for j in range(k):
            i=int(top[row,j]);a=F.silu(F.linear(x[row:row+1],gate[i]))*F.linear(x[row:row+1],up[i]);y[row]+=F.linear(a,down[i])[0]*w[row,j]
    return top,w,y

def main():
    torch.set_num_threads(1);g=torch.Generator().manual_seed(2026);n,h,width,k=128,16,8,8
    def rand(*shape):return torch.randn(*shape,generator=g)
    compact_diffs=0;axis_checks=0;tie_cases=0;compact_tie_diffs=0
    for m in (64,80,96,112,128):
        ids=torch.randperm(n,generator=g)[:m].sort().values
        router=rand(n,h);bias=rand(n);gate=rand(n,width,h);up=rand(n,width,h);down=rand(n,h,width)
        for tied in (False,True):
            r=torch.zeros_like(router) if tied else router; b=torch.zeros_like(bias) if tied else bias
            x=rand(12,h)
            want=teacher(x,r,b,gate,up,down,ids,k)
            got=run(x,r,b,gate[ids],up[ids],down[ids],ids,k,True)
            assert all(torch.equal(a,b) for a,b in zip(want,got)),(m,tied)
            other=run(x,r,b,gate[ids],up[ids],down[ids],ids,k,False)
            different=int((want[0].sort(dim=-1).values!=other[0].sort(dim=-1).values).any(dim=-1).sum())
            axis_checks+=x.shape[0];compact_diffs+=different
            if tied:tie_cases+=x.shape[0];compact_tie_diffs+=different
    print(json.dumps(dict(status='PASS_CPU_PRESERVED_ROUTER_AXIS_SYNTHETIC',torch=str(torch.__version__),rows=axis_checks,tie_rows=tie_cases,compact_axis_topk_set_differences=compact_diffs,compact_axis_tie_set_differences=compact_tie_diffs,preserved_axis_exact=True,actual_model='NOT_RUN')))
if __name__=='__main__':main()
