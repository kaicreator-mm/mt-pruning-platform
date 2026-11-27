"""R23E propagated-state diagnostic (no physical pruning or translation claim).
Pinned Engine.set_mask restricts routing on actual downstream states.
Record teacher and candidate at common historical inputs, never call generate.
"""
import json,hashlib,math,pathlib,inspect,argparse
P=pathlib.Path
ENGINE_BLOB="2af43ca1710b029fdde0d1135ea9879fe8defe56"
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def normerr(a,b):
    import torch
    a=a.double();b=b.double();need(a.shape==b.shape and bool(torch.isfinite(a).all() and torch.isfinite(b).all()),"finite shape")
    denom=float(b.square().sum());need(denom>1e-12,"zero teacher norm")
    return float((a-b).square().sum())/denom
def compare_routes(a,b):
    import torch
    need(a.shape==b.shape and a.ndim==2,"route shape")
    overlap=(a[:,:,None]==b[:,None,:]).any(dim=2).sum(dim=1)
    return dict(tokens=int(a.shape[0]),mean_overlap=float(overlap.double().mean()),exact_sets=int((overlap==a.shape[1]).sum()))
def observe(c,e,prompt,prefix,mask=None):
    torch=c.torch;need(len(prefix)==8 and len(prompt)>=2,"fixed trajectory")
    need(e.ckpt.kept is None and all(l.denied is None for l in e.layers) and e.on_route is None and e.audit is None and "_moe" not in e.__dict__,"clean teacher")
    layers=e.ckpt.sparse_layers
    if mask is not None:
        need(set(mask)==set(layers) and all(v==sorted(set(v)) and e.top_k<=len(v)<=e.ckpt.num_experts and all(type(i) is int and 0<=i<e.ckpt.num_experts for i in v) for v in mask.values()),"mask")
    orig=e._moe;records={};routes={};logits=[];state={}
    def route(l,phase,ids,weights):
        key=(l,phase,state["step"]);need(key not in routes and phase==state["phase"],"route coverage")
        if mask is not None:need(bool(torch.isin(ids,torch.tensor(mask[l])).all()),"excluded expert routed")
        routes[key]=ids[state["rows"]].clone()
    def moe(layer,x,phase):
        key=(layer.idx,phase,state["step"]);need(key not in records,"duplicate layer")
        out=orig(layer,x,phase)
        records[key]=(x[state["rows"]].clone(),out[state["rows"]].clone())
        return out
    try:
        e.set_mask(mask);e.on_route=route;e._moe=moe
        cache=e.new_cache(len(prompt)+9)
        with torch.inference_mode():
            for j,ids in enumerate([prompt]+[[t] for t in prefix]):
                phase="prefill" if j==0 else "decode";rows=[len(prompt)//2-1,len(prompt)-1] if j==0 else [0]
                state.update(phase=phase,step=j,rows=rows)
                z=e.forward(ids,cache,phase)
                logits.append(dict(step=j,sha256=c.logits_sha(z),next=int(z[-1].float().argmax()),selected=z[rows].clone()))
    finally:
        del e._moe;e.on_route=None;e.set_mask(None)
    expected={(l,"prefill" if j==0 else "decode",j) for l in layers for j in range(9)}
    need(set(records)==set(routes)==expected,"full propagation coverage")
    return dict(records=records,routes=routes,logits=logits)
def loghash(x):return [dict(step=z["step"],sha256=z["sha256"],next=z["next"]) for z in x["logits"]]
def compare(t,x):
    import torch
    need(set(t["records"])==set(x["records"]) and len(t["logits"])==len(x["logits"])==9,"coverage")
    layer=[]
    for key,(h,y) in sorted(t["records"].items()):
        a,b=x["records"][key];r=compare_routes(t["routes"][key],x["routes"][key])
        layer.append(dict(layer=key[0],phase=key[1],step=key[2],hidden_input_nmse=normerr(a,h),moe_output_nmse=normerr(b,y),**r))
    log=[]
    for a,b in zip(t["logits"],x["logits"]):
        p=a["selected"].double().log_softmax(-1);q=b["selected"].double().log_softmax(-1)
        kl=float((p.exp()*(p-q)).sum(dim=-1).mean())
        need(math.isfinite(kl) and kl>=-1e-12,"KL finite")
        log.append(dict(step=a["step"],nmse=normerr(b["selected"],a["selected"]),teacher_to_masked_kl=max(0.,kl),teacher_argmax=a["next"],masked_argmax=b["next"]))
    return dict(layer=layer,logits=log,teacher_hashes=loghash(t),masked_hashes=loghash(x))
def run(c,e,prompt,prefix,masks):
    need(set(masks)=={"original96","candidate96"},"exact two masks")
    t=observe(c,e,prompt,prefix);a=observe(c,e,prompt,prefix,masks["original96"]);b=observe(c,e,prompt,prefix,masks["candidate96"]);z=observe(c,e,prompt,prefix)
    need(loghash(t)==loghash(z),"same-process teacher restoration")
    return dict(status="PROPAGATED_FIXED_TRAJECTORY_DIAGNOSTIC_NOT_QUALITY",teacher_restored=True,original96=compare(t,a),candidate96=compare(t,b),scope="common fixed input sequence; actual propagated states, not free-running translation; selected prefill rows plus every decode step")
def selftest(c):
    import tempfile
    c.torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as d:
        p=P(d);model=c.tiny_checkpoint(p/"tiny",p/"none",vocab_size=256,seed=19);e=c.engine_module.Engine(c.engine_module.Checkpoint(model),threads=1)
        prompt=[4,8,15,16,23,42];prefix=c.greedy_prefix(e,prompt,8)
        a={l:list(range(8)) for l in e.ckpt.sparse_layers};b={l:[0,2,4,6] for l in e.ckpt.sparse_layers}
        x=run(c,e,prompt,prefix,{"original96":a,"candidate96":b})
        need(all(q["nmse"]==0 and q["teacher_to_masked_kl"]==0 for q in x["original96"]["logits"]),"full-mask identity")
        need(len(x["candidate96"]["layer"])==18 and any(q["nmse"]>0 for q in x["candidate96"]["logits"]),"propagation witness")
        need(all(l.denied is None for l in e.layers) and "_moe" not in e.__dict__,"restoration")
        try:observe(c,e,prompt,prefix,{l:[0] for l in e.ckpt.sparse_layers})
        except ValueError:pass
        else:raise AssertionError("invalid mask")
    print(json.dumps(dict(status="PASS_SYNTHETIC_PROPAGATED_STATE_WITNESS",layer_records=18,logits=9,real_capture="NOT_RUN")))
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--collector",type=P,required=True);a.add_argument("--selftest",action="store_true");q=a.parse_args()
    need(q.selftest,"preparation only")
    import importlib.util
    need(blob(q.collector.read_bytes())=="02daf65bb7fe0526b2a700d17013054f7b0eea46","collector pin")
    sp=importlib.util.spec_from_file_location("collector",q.collector);c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c);c.verify_code();selftest(c)
