"""Same-budget pair interactions for R23C; local reconstruction, not causal language labels.
All queried masks must be declared before sparse capture. The four states
share identical hidden input, expert values, top-k policy and denominator.
"""
import math,json
def need(x,m):
    if not x:raise ValueError(m)
def family(base,drop,add,n=128,k=8):
    need(type(n) is type(k) is int and 0<k<=len(base)<n,"budget")
    need(base==sorted(set(base)) and all(type(i) is int and 0<=i<n for i in base),"base mask")
    need(len(drop)==len(add)==2 and len(set(drop))==len(set(add))==2 and all(type(i) is int and 0<=i<n for i in drop+add),"pair identities")
    need(set(drop)<=set(base) and not (set(add)&set(base)),"swap membership")
    s=set(base);a,b=drop;c,d=add
    result={"base":base.copy(),"swap_a":sorted(s-{a}|{c}),"swap_b":sorted(s-{b}|{d}),"swap_ab":sorted(s-{a,b}|{c,d})}
    need(all(len(v)==len(base) for v in result.values()) and len({tuple(v) for v in result.values()})==4,"same-budget family")
    return result
def measure(losses):
    need(set(losses)=={"base","swap_a","swap_b","swap_ab"} and all(math.isfinite(v) and v>=0 for v in losses.values()),"finite loss inventory")
    q=losses;b=q["base"];a=q["swap_a"]-b;c=q["swap_b"]-b;ab=q["swap_ab"]-b
    interaction=ab-a-c
    return {"delta_a":a,"delta_b":c,"delta_joint":ab,"interaction":interaction,"joint_better_than_base":ab<0,"joint_better_than_both_singles":q["swap_ab"]<min(q["swap_a"],q["swap_b"])}
def evaluate(w,s,z,masks):
    need(set(masks)=={"base","swap_a","swap_b","swap_ab"} and len({len(v) for v in masks.values()})==1,"family")
    out={n:s.errors(w,z,ids) for n,ids in masks.items()}
    return {"losses":out,"combined":measure({n:v["combined_error"] for n,v in out.items()}),"routed":measure({n:v["routed_error"] for n,v in out.items()})}
def shortlist(base,importance,n=128):
    # Deterministic, no holdout use. None means unobserved, not zero contribution.
    need(len(importance)==n and all(x is None or (math.isfinite(x) and x>=0) for x in importance),"importance")
    need(base==sorted(set(base)) and all(type(i) is int and 0<=i<n for i in base),"base")
    drop=sorted((i for i in base if importance[i] is not None),key=lambda i:(importance[i],i))[:2]
    add=sorted((i for i in range(n) if i not in base and importance[i] is not None),key=lambda i:(-importance[i],i))[:2]
    return None if len(drop)<2 or len(add)<2 else {"drop":drop,"add":add,"masks":family(base,drop,add,n=n,k=min(8,len(base)))}
def selftest():
    m=family([0,1,2,3],[0,1],[4,5],n=6,k=2)
    need(m["swap_ab"]==[2,3,4,5] and m["swap_a"]==[1,2,3,4],"mask oracle")
    q=measure({"base":.25,"swap_a":.125,"swap_b":.125,"swap_ab":.375})
    need(q["interaction"]==.375 and q["delta_joint"]==.125 and not q["joint_better_than_base"],"antagonistic swap oracle")
    q=measure({"base":.25,"swap_a":.375,"swap_b":.375,"swap_ab":.125})
    need(q["interaction"]==-.375 and q["joint_better_than_both_singles"],"synergistic swap oracle")
    q=shortlist([0,1,2,3],[1.,2.,3.,4.,5.,6.],6);need(q["drop"]==[0,1] and q["add"]==[5,4],"rank")
    need(shortlist([0,1,2,3],[1.,2.,3.,4.,None,6.],6) is None,"unobserved reject")
    def rejects(fn):
        try:fn()
        except ValueError:return
        raise AssertionError("invalid accepted")
    rejects(lambda:family([0,1,2,3],[0,0],[4,5],6,2))
    rejects(lambda:family([0,1,2,3],[0,1],[3,5],6,2))
    rejects(lambda:measure({"base":.1,"swap_a":.1,"swap_b":.1,"swap_ab":float("nan")}))
    print(json.dumps(dict(status="PASS_SYNTHETIC_SAME_BUDGET_INTERACTION",positive=5,negative=3,real_capture="NOT_RUN")))
if __name__=="__main__":selftest()
