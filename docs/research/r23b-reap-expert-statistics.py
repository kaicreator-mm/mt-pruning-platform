"""R23B additive routed-expert score statistics.

REAP conditional mean as in Lasby et al., ICLR 2026, arXiv:2510.13999.
Keep counts and sums, not averages of per-batch averages. Effective weights
are supplied by the exact model router; this module does not reroute tokens.
No claim of full model capture or pruning quality.
"""
import json,math
from dataclasses import dataclass
@dataclass
class ExpertStats:
    count:int=0
    weight:float=0.
    norm:float=0.
    weighted_norm:float=0.
    energy:float=0.
    def add(self,g,n):
        if not (math.isfinite(g) and g>=0 and math.isfinite(n) and n>=0):
            raise ValueError("invalid expert observation")
        self.count+=1;self.weight+=g;self.norm+=n;self.weighted_norm+=g*n;self.energy+=(g*n)**2
    def merge(self,other):
        for k in ("count","weight","norm","weighted_norm","energy"):
            setattr(self,k,getattr(self,k)+getattr(other,k))
    def report(self,tokens):
        if type(tokens) is not int or tokens<self.count or tokens<0:raise ValueError("token count")
        return dict(count=self.count,coverage=self.count/tokens if tokens else None,router_weight_sum=self.weight,activation_norm_sum=self.norm,weighted_norm_sum=self.weighted_norm,weighted_energy=self.energy,reap=self.weighted_norm/self.count if self.count else None,global_weighted_mean=self.weighted_norm/tokens if tokens else None,observed=self.count>0)
class Profile:
    def __init__(self,n,k,scale):
        if not (type(n) is type(k) is int and 0<k<=n and math.isfinite(scale) and scale>0):raise ValueError("router shape")
        self.n=n;self.k=k;self.scale=scale;self.tokens=0;self.stats=[ExpertStats() for _ in range(n)]
    def add(self,ids,weights,norms):
        if len(ids)!=self.k or len(weights)!=self.k or len(norms)!=self.k or len(set(ids))!=self.k or any(type(i) is not int or not 0<=i<self.n for i in ids):raise ValueError("top-k identities")
        if any(not math.isfinite(g) or g<0 or not math.isfinite(n) or n<0 for g,n in zip(weights,norms)):raise ValueError("nonfinite or negative")
        if not math.isclose(math.fsum(weights),self.scale,rel_tol=1e-5,abs_tol=1e-7):raise ValueError("effective routing normalization")
        for i,g,n in zip(ids,weights,norms):self.stats[i].add(g,n)
        self.tokens+=1
    def merge(self,o):
        if (self.n,self.k,self.scale)!=(o.n,o.k,o.scale):raise ValueError("incompatible router")
        for a,b in zip(self.stats,o.stats):a.merge(b)
        self.tokens+=o.tokens
    def report(self):
        if sum(s.count for s in self.stats)!=self.tokens*self.k:raise ValueError("count accounting")
        return {"tokens":self.tokens,"experts":[s.report(self.tokens) for s in self.stats]}
def observe_sparse(z,n,*,scale):
    """Adapt real sparse witness tensor layout, using teacher top-k only."""
    import torch
    scores=z["scores"];choice=z["choice"];values=z["values"];k=z["top_k"]
    if scores.ndim!=2 or scores.shape!=choice.shape or scores.shape[1]!=n or values.ndim!=3 or values.shape[:2]!=scores.shape:raise ValueError("witness shape")
    if not (torch.isfinite(scores).all() and torch.isfinite(choice).all()):raise ValueError("nonfinite router")
    if float(z["scale"])!=float(scale):raise ValueError("scale identity")
    top=torch.topk(choice,k,dim=-1,sorted=False).indices
    if k<n:
        sorted_choice=torch.sort(choice,dim=-1,descending=True).values
        if not bool((sorted_choice[:,k-1]>sorted_choice[:,k]).all()):raise ValueError("ambiguous boundary")
    g=scores.gather(1,top);g=g/(g.sum(dim=-1,keepdim=True)+1e-20)*scale
    selected=values[torch.arange(scores.shape[0])[:,None],top]
    if not bool(torch.isfinite(selected).all()):raise ValueError("uncomputed expert output")
    norms=torch.linalg.vector_norm(selected.double(),dim=-1)
    p=Profile(n,k,scale)
    for ids,weights,ns in zip(top.tolist(),g.tolist(),norms.tolist()):p.add(ids,weights,ns)
    return p
def selftest():
    p=Profile(4,2,1.);p.add([0,1],[.25,.75],[4.,2.]);p.add([0,2],[.5,.5],[6.,8.]);r=p.report()["experts"]
    assert r[0]["reap"]==2 and r[0]["weighted_energy"]==10 and r[1]["reap"]==1.5 and r[1]["global_weighted_mean"]==.75 and r[3]["reap"] is None
    a=Profile(4,2,1.);b=Profile(4,2,1.);a.add([0,1],[.25,.75],[4.,2.]);b.add([0,2],[.5,.5],[6.,8.]);a.merge(b);assert a.report()==p.report()
    def reject(f):
        try:f()
        except ValueError:return
        raise AssertionError("invalid accepted")
    reject(lambda:p.add([0,0],[.5,.5],[1.,1.]))
    reject(lambda:p.add([0,1],[.5,.4],[1.,1.]))
    reject(lambda:p.add([0,1],[.5,.5],[float("nan"),1.]))
    reject(lambda:p.merge(Profile(5,2,1.)))
    reject(lambda:p.add([0,4],[.5,.5],[1.,1.]))
    print(json.dumps(dict(status="PASS_SYNTHETIC_REAP_STATISTICS",positive=2,negative=5,real_capture="NOT_RUN")))
if __name__=="__main__":selftest()
