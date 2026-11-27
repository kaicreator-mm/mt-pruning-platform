"""Independent scalar router/expert oracle for ragged physical compaction.
Pure Python, synthetic only. Mirrors the documented sigmoid + bias selection,
normalized unbiased scores and gated FFN. Deterministic tie policy here is
original expert ID, not a claim about torch.topk or native llama.cpp ties.
"""
import math,json,random
def need(x,m):
    if not x:raise ValueError(m)
def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))
def sigmoid(x):return 1/(1+math.exp(-x))
def linear(x,w):return [dot(x,row) for row in w]
def silu(x):return x*sigmoid(x)
def validate(ids,n,k):
    need(type(k) is int and 1<=k<=len(ids)<=n and ids==sorted(set(ids)) and all(type(i) is int and 0<=i<n for i in ids),"expert IDs")
def compact(layer,ids,k):
    n=len(layer["router"]);validate(ids,n,k)
    need(len(layer["bias"])==len(layer["experts"])==n,"tensor inventory")
    return dict(origin_ids=ids.copy(),router=[layer["router"][i] for i in ids],bias=[layer["bias"][i] for i in ids],experts=[layer["experts"][i] for i in ids],shared=layer["shared"])
def select(scores,bias,local_ids,origin,k):
    need(len(scores)==len(bias)==len(origin) and len(set(origin))==len(origin),"router inventory")
    need(all(math.isfinite(v) for v in scores+bias),"finite scores")
    order=sorted(local_ids,key=lambda i:(-(scores[i]+bias[i]),origin[i]))
    need(len(order)>=k,"eligible experts")
    # Record top-k boundary ties separately: native unsorted topk need not
    # share this tie-breaking rule when physical axis width changes.
    ties=len(order)>k and scores[order[k-1]]+bias[order[k-1]]==scores[order[k]]+bias[order[k]]
    top=order[:k];mass=math.fsum(scores[i] for i in top);need(mass>0,"route mass")
    return top,[scores[i]/(mass+1e-20)*2.826 for i in top],ties
def ffn(x,e):
    g,u,d=e;h=[silu(a)*b for a,b in zip(linear(x,g),linear(x,u))]
    return linear(h,d)
def run(x,layer,k,allowed=None):
    n=len(layer["router"]);origin=layer.get("origin_ids",list(range(n)))
    scores=[sigmoid(v) for v in linear(x,layer["router"])]
    eligible=list(range(n)) if allowed is None else allowed
    top,w,tie=select(scores,layer["bias"],eligible,origin,k)
    y=[ffn(x,layer["experts"][i]) for i in top];shared=ffn(x,layer["shared"])
    return dict(ids=[origin[i] for i in top],weights=w,output=[shared[j]+math.fsum(weight*v[j] for weight,v in zip(w,y)) for j in range(len(x))],boundary_tie=tie)
def model(seed,n=8,h=4,width=3):
    rng=random.Random(seed)
    def mat(a,b):return [[rng.uniform(-.8,.8) for _ in range(b)] for _ in range(a)]
    def e():return (mat(width,h),mat(width,h),mat(h,width))
    return dict(router=mat(n,h),bias=[rng.uniform(-.2,.2) for _ in range(n)],experts=[e() for _ in range(n)],shared=e())
def selftest():
    layers=[model(23),model(24),model(25)]
    masks=[[0,2,4],[1,2,3,5,6],[0,1,2,3,4,5,6,7]]
    compacted=[compact(layer,ids,2) for layer,ids in zip(layers,masks)]
    rng=random.Random(2026);checks=0;ties=0
    for _ in range(24):
        x=[rng.uniform(-1,1) for _ in range(4)];a=x[:];b=x[:]
        for layer,small,ids in zip(layers,compacted,masks):
            old=run(a,layer,2,ids);new=run(b,small,2)
            need(old==new,"local routing/weights/output equivalence")
            need(set(new["ids"])<=set(ids),"excluded expert selected")
            a=[x+y for x,y in zip(a,old["output"])];b=[x+y for x,y in zip(b,new["output"])]
            ties+=int(old["boundary_tie"]);checks+=1
        need(a==b,"propagated residual equality")
    # All equal router scores + biases: tie recorded, original-ID policy explicit.
    t=model(3);t["router"]=[[0.]*4 for _ in range(8)];t["bias"]=[0.]*8
    ids=[1,3,5,7];a=run([1.]*4,t,2,ids);b=run([1.]*4,compact(t,ids,2),2)
    need(a==b and a["boundary_tie"] and a["ids"]==[1,3],"tie witness")
    def rejects(ids,k):
        try:compact(t,ids,k)
        except ValueError:return
        raise AssertionError("invalid mask accepted")
    for ids,k in (([1],2),([1,1],2),([3,1],2),([1,8],2)):rejects(ids,k)
    print(json.dumps(dict(status="PASS_SCALAR_RAGGED_ROUTING_ORACLE",propagated_inputs=24,layer_checks=checks,layer_counts=list(map(len,masks)),random_boundary_ties=ties,explicit_tie_witness=True,negative=4,production_topk_equivalence="NOT_PROVEN",actual_model="NOT_RUN")))
if __name__=="__main__":selftest()
