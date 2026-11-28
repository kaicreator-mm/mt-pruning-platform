"""R23 sparse counterfactual expert evaluation for the fixed mask inventory.
Same hidden-state witness semantics, same original output oracle. Only evaluate
the union of experts selected by teacher and every declared candidate. Uncomputed
values are NaN and undeclared candidate queries are rejected.
No real-model run in this module.
"""
import argparse, json, math, tempfile, time
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import importlib.util

COLLECTOR_BLOB="02daf65bb7fe0526b2a700d17013054f7b0eea46"
def need(v,m):
    if not v:raise ValueError(m)
def load_collector(path):
    import hashlib
    b=path.read_bytes();need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==COLLECTOR_BLOB,"collector pin")
    s=importlib.util.spec_from_file_location("collector",path);c=importlib.util.module_from_spec(s);s.loader.exec_module(c);c.verify_code();return c
def selected(choice,ids,k,w):
    n=choice.shape[1];w.check_ids(ids,n,k)
    kept=torch.tensor(ids,dtype=torch.long);c=choice[:,kept]
    if len(ids)>k:
        v=torch.sort(c,dim=1,descending=True).values
        need(bool((v[:,k-1]>v[:,k]).all()),"top-k boundary tie")
    return kept[torch.topk(c,k,dim=1,sorted=False).indices]
@torch.inference_mode()
def sparse_witness(w,layer,x,rows,top_k,scale,actual,combine,masks,*,sample_id,direction,phase,split):
    need(split=="calibration" and sample_id.startswith("calibration:"),"calibration only")
    need(direction in w.DIRECTIONS and phase in ("prefill","decode"),"group")
    need(0<len(rows)<=w.MAX_ROWS and rows==sorted(set(rows)) and all(type(i)==int and 0<=i<len(x) for i in rows),"bounded rows")
    need(layer.denied is None and x.device.type=="cpu" and len(layer.gate)==len(layer.up)==len(layer.down),"teacher inventory")
    n=len(layer.gate);xx=x[rows]
    need(0<top_k<=n and masks and len({len(v) for v in masks})==1,"mask budget")
    allowed={tuple(v) for v in masks}
    for v in masks:w.check_ids(v,n,top_k)
    scores=torch.sigmoid(F.linear(x.float(),layer.router))[rows]
    choice=scores+layer.bias
    need(bool(torch.isfinite(scores).all() and torch.isfinite(choice).all()),"router finite")
    top=torch.topk(choice,top_k,dim=-1,sorted=False).indices
    wt=scores.gather(1,top);wt=wt/(wt.sum(dim=-1,keepdim=True)+1e-20)*scale
    baseline_routed=combine(xx,top,wt,layer.gate,layer.up,layer.down)
    shared=w.expert_fn(xx,*layer.shared)
    required=set(top.flatten().tolist())
    for ids in [list(range(n)),*masks]:
        required.update(selected(choice,ids,top_k,w).flatten().tolist())
    # Each required expert is evaluated on the whole selected row batch,
    # exactly the same GEMM shape as the exhaustive witness. This matters
    # for BF16 rounding and is not a per-token shape shortcut.
    ids=sorted(required)
    first=w.expert_fn(xx,layer.gate[ids[0]],layer.up[ids[0]],layer.down[ids[0]])
    values=torch.full((len(xx),n,first.shape[-1]),float("nan"),dtype=first.dtype)
    values[:,ids[0]]=first
    for j in ids[1:]:values[:,j]=w.expert_fn(xx,layer.gate[j],layer.up[j],layer.down[j])
    need(bool(torch.isfinite(values[:,ids]).all()),"selected expert output finite")
    z={"id":sample_id,"direction":direction,"phase":phase,"layer":layer.idx,"split":split,"rows":rows,"scores":scores,"choice":choice,"values":values,"shared":shared,"top_k":top_k,"scale":scale,"teacher_routed":baseline_routed,"teacher_combined":actual[rows].clone(),"allowed":allowed,"computed":ids,"nominal_token_expert_evaluations":len(xx)*n,"actual_token_expert_evaluations":len(xx)*len(ids)}
    _,_,routed,combined=w.mix(z,list(range(n)))
    z["observer_routed_drift"]=w.normalized_error(routed,baseline_routed)
    z["observer_combined_drift"]=w.normalized_error(combined,z["teacher_combined"])
    weighted=values[torch.arange(len(rows))[:,None],top].float()*wt[:,:,None]
    energy=torch.zeros(n,dtype=torch.float64)
    energy.scatter_add_(0,top.flatten(),weighted.square().sum(dim=-1).double().flatten())
    z["energy"]=energy
    return z
def errors(w,z,ids):
    need(tuple(ids) in z["allowed"],"candidate not declared before capture")
    need(set(selected(z["choice"],ids,z["top_k"],w).flatten().tolist())<=set(z["computed"]),"missing computed expert")
    return w.errors(z,ids)
def adapter(w,candidates):
    counters=[]
    def witness(layer,*a,**kw):
        masks=[m[layer.idx] for m in candidates.values()]
        z=sparse_witness(w,layer,*a,masks=masks,**kw)
        counters.append({"layer":layer.idx,"phase":z["phase"],"nominal":z["nominal_token_expert_evaluations"],"actual":z["actual_token_expert_evaluations"]})
        return z
    return SimpleNamespace(check_ids=w.check_ids,witness=witness,errors=lambda z,ids:errors(w,z,ids),counters=counters)
def capture(c,w,engine,record,prompt,candidates):
    a=adapter(w,candidates)
    result=c.capture(a,engine,record,prompt,candidates)
    nominal=sum(x["nominal"] for x in a.counters);actual=sum(x["actual"] for x in a.counters)
    need(nominal==result.pop("token_expert_evaluations") and 0<actual<=nominal,"counter accounting")
    result.update(nominal_exhaustive_token_expert_evaluations=nominal,actual_counterfactual_token_expert_evaluations=actual,expert_compute_counts=a.counters)
    return result
def selftest(c,w):
    import random
    torch.set_num_threads(1)
    checks=0;nominal=0;actual=0;groups=[]
    start=time.monotonic()
    for seed in range(12):
        torch.manual_seed(seed+41);rng=random.Random(seed+41)
        n,h,d,k=128,24,16,8
        x=torch.randn(7,h).bfloat16()
        gate=[torch.randn(d,h).bfloat16() for _ in range(n)]
        up=[torch.randn(d,h).bfloat16() for _ in range(n)]
        down=[torch.randn(h,d).bfloat16() for _ in range(n)]
        layer=SimpleNamespace(idx=1,denied=None,router=torch.randn(n,h),bias=torch.arange(n).float()/127,gate=gate,up=up,down=down,shared=(gate[0],up[0],down[0]))
        scale=2.826;scores=torch.sigmoid(F.linear(x.float(),layer.router));top=torch.topk(scores+layer.bias,k,dim=-1,sorted=False).indices
        wt=scores.gather(1,top);wt=wt/(wt.sum(dim=-1,keepdim=True)+1e-20)*scale
        teacher=c.engine_module.expert_combine(x,top,wt,gate,up,down)+w.expert_fn(x,*layer.shared)
        masks=[sorted(rng.sample(range(n),96)) for _ in range(5)]
        for rows in ([0],[1,6]):
            kw=dict(sample_id="calibration:zh-my:synthetic",direction="zh-my",phase="decode" if len(rows)==1 else "prefill",split="calibration")
            ref=w.witness(layer,x,rows,k,scale,teacher,c.engine_module.expert_combine,**kw)
            z=sparse_witness(w,layer,x,rows,k,scale,teacher,c.engine_module.expert_combine,masks,**kw)
            need(z["observer_routed_drift"]==ref["observer_routed_drift"] and z["observer_combined_drift"]==ref["observer_combined_drift"] and torch.equal(z["energy"],ref["energy"]),"oracle drift/energy")
            need(torch.equal(z["values"][:,z["computed"]],ref["values"][:,z["computed"]]),"computed expert bytes")
            for ids in masks:
                need(errors(w,z,ids)==w.errors(ref,ids),"candidate oracle")
                checks+=1
            nominal+=z["nominal_token_expert_evaluations"];actual+=z["actual_token_expert_evaluations"]
            groups.append({"rows":len(rows),"expert_union":len(z["computed"])})
    def rejects(fn):
        try:fn()
        except ValueError:return
        raise AssertionError("expected rejection")
    rejects(lambda:errors(w,z,list(range(96))))
    rejects(lambda:sparse_witness(w,layer,x,[0],k,scale,teacher,c.engine_module.expert_combine,masks,**dict(kw,split="validation")))
    rejects(lambda:sparse_witness(w,layer,x,[0],k,scale,teacher,c.engine_module.expert_combine,[masks[0],masks[1][:-1]],**kw))
    rejects(lambda:selected(torch.ones(1,128),list(range(128)),8,w))
    # Full engine observer differential: exactly identical logits/loss records.
    with tempfile.TemporaryDirectory() as d:
        p=Path(d);model=c.tiny_checkpoint(p/"tiny",p/"none",vocab_size=256,seed=19)
        engine=c.engine_module.Engine(c.engine_module.Checkpoint(model),threads=1)
        prompt=[4,8,15,16,23,42];prefix=c.greedy_prefix(engine,prompt,8)
        rec={"id":"calibration:zh-my:synthetic","direction":"zh-my","line":0,"split":"calibration","prompt_tokens":len(prompt),"prompt_ids_sha256":c.canonical(prompt),"teacher_prefix_tokens":prefix,"teacher_prefix_sha256":c.canonical(prefix),"prefill_rows":[2,5],"decode_steps_1based":[1,8]}
        ms={"a":{1:[0,2,4,6],2:[1,2,5,7]},"b":{1:[1,3,5,7],2:[0,3,4,6]}}
        full=c.capture(w,engine,rec,prompt,ms);sparse=capture(c,w,engine,rec,prompt,ms)
        need(full["teacher_logits"]==sparse["teacher_logits"] and full["observations"]==sparse["observations"] and full["token_expert_evaluations"]==sparse["nominal_exhaustive_token_expert_evaluations"],"full engine exact differential")
        need("_moe" not in engine.__dict__,"observer restored")
        print(json.dumps({"status":"PASS_SYNTHETIC_SPARSE_EXACT_DIFFERENTIAL","torch":torch.__version__,"candidate_checks":checks,"negative_checks":4,"nominal":nominal,"actual":actual,"reduction_fraction":1-actual/nominal,"synthetic_union_counts":groups,"engine_logit_checks":len(sparse["teacher_logits"]),"engine_observations":len(sparse["observations"]),"engine_nominal":full["token_expert_evaluations"],"engine_actual":sparse["actual_counterfactual_token_expert_evaluations"],"elapsed_seconds":time.monotonic()-start,"real_capture":"NOT_RUN"},allow_nan=False))
def main():
    p=argparse.ArgumentParser();p.add_argument("--collector",type=Path,required=True);p.add_argument("--witness",type=Path,required=True);p.add_argument("--selftest",action="store_true");a=p.parse_args();need(a.selftest,"synthetic only");c=load_collector(a.collector);selftest(c,c.load_witness(a.witness))
if __name__=="__main__":main()

