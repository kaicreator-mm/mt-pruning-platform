"""R23 preparation: bounded same-hidden-state expert output error witness.
CPU, no training. The real collector is not dispatched by this module.
Actual end-to-end quality is a separate gate.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ENGINE_BLOB="2af43ca1710b029fdde0d1135ea9879fe8defe56"
MAX_ROWS=8
PRIMARY={"zh-my","my-zh"}
DIRECTIONS=PRIMARY|{"zh-en","en-zh","en-my","my-en"}
def need(ok,why):
    if not ok:raise ValueError(why)
def baseline_combine(path):
    b=path.read_bytes()
    need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==ENGINE_BLOB,"engine blob")
    tree=ast.parse(b);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="expert_combine"]
    need(len(nodes)==1,"baseline definition")
    env={"torch":torch,"F":F};exec(compile(ast.Module(body=nodes,type_ignores=[]),"pinned-engine","exec"),env)
    return env["expert_combine"]
def expert_fn(x,gate,up,down):
    return F.linear(F.silu(F.linear(x,gate))*F.linear(x,up),down)
def check_ids(ids,n,k):
    need(isinstance(ids,(list,tuple)) and len(ids)>=k and list(ids)==sorted(set(ids)) and all(type(i)==int and 0<=i<n for i in ids),"mask")
def mix(w,ids):
    """Same hidden-state alternate top-k, sigmoid scores + expert bias.
    All expert values are from the original teacher, no weight updates.
    Ties at selection boundary are rejected, not resolved by invented order.
    """
    scores,choice,values=w["scores"],w["choice"],w["values"]
    n=scores.shape[1];k=w["top_k"];check_ids(ids,n,k)
    kept=torch.tensor(ids,dtype=torch.long)
    c=choice[:,kept]
    if len(ids)>k:
        sorted_c=torch.sort(c,dim=1,descending=True).values
        need(bool((sorted_c[:,k-1]>sorted_c[:,k]).all()),"top-k boundary tie")
    top=kept[torch.topk(c,k,dim=1,sorted=False).indices]
    wt=scores.gather(1,top);wt=wt/(wt.sum(dim=1,keepdim=True)+1e-20)*w["scale"]
    chosen=values[torch.arange(scores.shape[0])[:,None],top]
    routed=(chosen*wt[:,:,None]).sum(dim=1).to(values.dtype)
    combined=routed+w["shared"]
    return top,wt,routed,combined
def normalized_error(actual,teacher):
    a=actual.float();b=teacher.float()
    need(a.shape==b.shape and bool(torch.isfinite(a).all() and torch.isfinite(b).all()),"output finite/shape")
    denom=float(b.square().sum())
    need(denom>1e-12,"zero teacher norm")
    return float((a-b).square().sum())/denom
@torch.inference_mode()
def witness(layer,x,rows,top_k,scale,actual,combine,*,sample_id,direction,phase,split):
    need(split=="calibration" and sample_id.startswith("calibration:"),"calibration only")
    need(direction in DIRECTIONS and phase in ("prefill","decode"),"group")
    need(len(rows)>0 and len(rows)<=MAX_ROWS and rows==sorted(set(rows)) and all(type(i)==int and 0<=i<len(x) for i in rows),"bounded row selection")
    need(layer.denied is None,"teacher must be unmasked")
    need(x.device.type=="cpu" and len(layer.gate)==len(layer.up)==len(layer.down),"CPU expert inventory")
    n=len(layer.gate);xx=x[rows]
    # Compute router over the original token batch to preserve its shape.
    scores=torch.sigmoid(F.linear(x.float(),layer.router))[rows]
    choice=scores+layer.bias;top=torch.topk(choice,top_k,dim=-1,sorted=False).indices
    wt=scores.gather(1,top);wt=wt/(wt.sum(dim=-1,keepdim=True)+1e-20)*scale
    baseline_routed=combine(xx,top,wt,layer.gate,layer.up,layer.down)
    shared=expert_fn(xx,*layer.shared)
    # Extra alternate experts evaluated only on selected rows; this is local
    # counterfactual output reconstruction, not full pruned model execution.
    values=torch.stack([expert_fn(xx,g,u,d) for g,u,d in zip(layer.gate,layer.up,layer.down)],dim=1)
    w={"id":sample_id,"direction":direction,"phase":phase,"layer":layer.idx,"split":split,"rows":rows,"scores":scores,"choice":choice,"values":values,"shared":shared,"top_k":top_k,"scale":scale,"teacher_routed":baseline_routed,"teacher_combined":actual[rows].clone()}
    need(bool(torch.isfinite(values).all() and torch.isfinite(scores).all()),"nonfinite expert output")
    _,_,reconstructed,combined=mix(w,list(range(n)))
    w["observer_routed_drift"]=normalized_error(reconstructed,baseline_routed)
    w["observer_combined_drift"]=normalized_error(combined,w["teacher_combined"])
    # These measurements may be nonzero at real GEMM shapes/rounding; caller
    # must assess drift before using any candidate loss, not silently ignore it.
    weighted=values[torch.arange(len(rows))[:,None],top].float()*wt[:,:,None]
    energy=torch.zeros(n,dtype=torch.float64)
    energy.scatter_add_(0,top.flatten(),weighted.square().sum(dim=-1).double().flatten())
    w["energy"]=energy
    return w
def errors(w,ids):
    _,_,routed,combined=mix(w,ids)
    return {"routed_error":normalized_error(routed,w["teacher_routed"]),"combined_error":normalized_error(combined,w["teacher_combined"])}
def score(witnesses,ids):
    need(bool(witnesses),"empty witness set")
    groups={}
    for w in witnesses:
        need(w["split"]=="calibration" and w["id"].startswith("calibration:"),"calibration only")
        e=errors(w,ids)
        groups.setdefault(w["direction"]+"/"+w["phase"],[]).append(e["combined_error"])
    means={g:sum(v)/len(v) for g,v in groups.items()}
    need(all(g in means for g in ("zh-my/prefill","zh-my/decode","my-zh/prefill","my-zh/decode")),"primary phase coverage")
    primary=[v for g,v in means.items() if g.split("/")[0] in PRIMARY]
    return {"group_mean_error":means,"primary_worst":max(primary),"primary_mean":sum(primary)/len(primary),"all_mean":sum(means.values())/len(means)}
def select_fit(fit,holdout,candidates,baseline_name):
    """Fit on calibration subset only; holdout is a report, never selection.
    All masks same budget, explicit layer-local alternatives.
    """
    need(bool(fit) and bool(holdout),"empty split")
    need(not ({w["id"] for w in fit}&{w["id"] for w in holdout}),"calibration overlap")
    need(len({w["layer"] for w in fit+holdout})==1,"single layer")
    need(baseline_name in candidates and len(candidates)>=2 and len({len(v) for v in candidates.values()})==1,"candidate inventory")
    f={name:score(fit,ids) for name,ids in candidates.items()}
    chosen=min(f,key=lambda name:(f[name]["primary_worst"],f[name]["primary_mean"],f[name]["all_mean"],name))
    return {"selected_fit_only":chosen,"fit":f,"holdout_baseline":score(holdout,candidates[baseline_name]),"holdout_selected":score(holdout,candidates[chosen]),"status":"LOCAL_OUTPUT_PROXY_ONLY"}
def selftest(combine):
    torch.manual_seed(19);torch.set_num_threads(1)
    n,h,d,k=6,8,5,2
    x=torch.randn(5,h).bfloat16()
    def matrices():return [torch.randn(d,h).bfloat16() for _ in range(n)]
    gate,up=matrices(),matrices();down=[torch.randn(h,d).bfloat16() for _ in range(n)]
    shared=(gate[0],up[0],down[0])
    layer=SimpleNamespace(idx=1,denied=None,router=torch.randn(n,h),bias=torch.arange(n).float()/17,gate=gate,up=up,down=down,shared=shared)
    scale=2.826
    scores=torch.sigmoid(F.linear(x.float(),layer.router));top=torch.topk(scores+layer.bias,k,dim=-1,sorted=False).indices
    wt=scores.gather(1,top);wt=wt/(wt.sum(dim=-1,keepdim=True)+1e-20)*scale
    actual=combine(x,top,wt,gate,up,down)+expert_fn(x,*shared)
    before=x.clone();original=actual.clone()
    w=witness(layer,x,list(range(len(x))),k,scale,actual,combine,sample_id="calibration:zh-my:1",direction="zh-my",phase="prefill",split="calibration")
    need(torch.equal(x,before) and torch.equal(actual,original),"observer mutated teacher")
    need(w["observer_routed_drift"]==0 and w["observer_combined_drift"]==0,"full reconstruction")
    checks=0
    import itertools
    for ids in itertools.combinations(range(n),4):
        t,wt,routed,combined=mix(w,list(ids))
        oracle=combine(x,t,wt,gate,up,down)
        need(torch.equal(oracle,routed) and torch.equal(oracle+expert_fn(x,*shared),combined),"masked output oracle")
        checks+=1
    # Construct counterexample: highest router mass prefers expert0,
    # but it has zero output, while expert1 has the actual teacher signal.
    toy={"id":"calibration:toy:1","split":"calibration","layer":1,"direction":"zh-my","phase":"prefill","scores":torch.tensor([[.95,.8,.7]]),"choice":torch.tensor([[.95,.8,.7]]),"values":torch.tensor([[[0.,0.],[4.,2.],[0.,0.]]]),"shared":torch.tensor([[1.,1.]]),"top_k":2,"scale":1.0}
    _,_,toy["teacher_routed"],toy["teacher_combined"]=mix(toy,[0,1,2])
    # Drop either expert0 or expert1. Renormalization makes either different,
    # but keeping expert1 matters more than keeping expert0.
    mass=errors(toy,[0,2]);functional=errors(toy,[1,2])
    need(functional["combined_error"]<mass["combined_error"],"output sensitivity counterexample")
    fit=[];hold=[]
    for dname in ("zh-my","my-zh","zh-en","en-zh","en-my","my-en"):
        for phase in ("prefill","decode"):
            fit.append(dict(toy,id="calibration:"+dname+":fit",direction=dname,phase=phase))
            hold.append(dict(toy,id="calibration:"+dname+":hold",direction=dname,phase=phase))
    result=select_fit(fit,hold,{"mass":[0,2],"functional":[1,2]},"mass")
    need(result["selected_fit_only"]=="functional","fit selection")
    def rejects(fn):
        try:fn()
        except ValueError:return
        raise AssertionError("expected rejection")
    rejects(lambda:mix(w,[0,0,1,2]))
    rejects(lambda:mix(w,[0]))
    rejects(lambda:mix(dict(toy,choice=torch.ones_like(toy["choice"])),[0,1,2]))
    rejects(lambda:select_fit(fit,fit,{"mass":[0,2],"functional":[1,2]},"mass"))
    rejects(lambda:witness(layer,x,[0],k,scale,actual,combine,sample_id="validation:zh-my:1",direction="zh-my",phase="prefill",split="validation"))
    rejects(lambda:witness(layer,x,[0,1,2,3,4,5,6,7,8],k,scale,actual,combine,sample_id="calibration:zh-my:1",direction="zh-my",phase="prefill",split="calibration"))
    print(json.dumps({"status":"PASS_SYNTHETIC_ONLY","torch":torch.__version__,"masked_oracle_checks":checks,"negative_checks":6,"observer_routed_drift":w["observer_routed_drift"],"observer_combined_drift":w["observer_combined_drift"],"toy_mass_error":mass["combined_error"],"toy_functional_error":functional["combined_error"],"fit_only_selection":result["selected_fit_only"],"real_capture":"NOT_RUN"}))
def main():
    p=argparse.ArgumentParser();p.add_argument("--engine",type=Path,required=True);p.add_argument("--selftest",action="store_true")
    a=p.parse_args();need(a.selftest,"no real collector dispatch");selftest(baseline_combine(a.engine))
if __name__=="__main__":main()

