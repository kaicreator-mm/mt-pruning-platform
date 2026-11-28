"""R23 full-engine observer integration; preparation and finite pilot only.

Keep original Engine._moe return value unchanged. Capture only selected
rows; release all expert output tensors after reducing each observation.
Never use validation/test text to choose experts.
"""
import argparse, hashlib, importlib.util, json, math, tempfile, time
from collections import Counter
from pathlib import Path
import torch
import mtpruning.engine as engine_module
from mtpruning.testing import tiny_checkpoint

PINS={"engine.py":"2af43ca1710b029fdde0d1135ea9879fe8defe56","testing.py":"e8979ac3cd42b739dd93448f3384c188c08568c4","prune.py":"770f55b8fc20f0e78cd8b32673829da6e1a9c461","identity.py":"0a2375314d599af492e6bffc8e24287150262f58"}
WITNESS_BLOB="b972e153a299b7fb39e580dfacfc08aaa8fe942c"
def need(x,m):
    if not x:raise ValueError(m)
def blob(path):
    b=Path(path).read_bytes()
    return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def load_witness(path):
    need(blob(path)==WITNESS_BLOB,"witness pin")
    spec=importlib.util.spec_from_file_location("r23_witness",path)
    w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w);return w
def verify_code():
    p=Path(engine_module.__file__).resolve().parent
    for name,sha in PINS.items():need(blob(p/name)==sha,"engine package pin "+name)
    return dict(PINS)
def logits_sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def forward_sequence(engine,prompt,prefix,callback=None):
    cache=engine.new_cache(len(prompt)+len(prefix)+1)
    seen=[]
    with torch.inference_mode():
        if callback:callback("prefill",0)
        logits=engine.forward(prompt,cache,"prefill")
        seen.append({"phase":"prefill","step":0,"sha256":logits_sha(logits),"next":int(logits[-1].float().argmax())})
        for j,token in enumerate(prefix,1):
            need(token==seen[-1]["next"],"teacher prefix diverged before step "+str(j))
            if callback:callback("decode",j)
            logits=engine.forward([token],cache,"decode")
            seen.append({"phase":"decode","step":j,"sha256":logits_sha(logits),"next":int(logits[-1].float().argmax())})
    return seen
def greedy_prefix(engine,prompt,n):
    cache=engine.new_cache(len(prompt)+n+1);logits=engine.forward(prompt,cache,"prefill")
    out=[]
    for _ in range(n):
        t=int(logits[-1].float().argmax());out.append(t);logits=engine.forward([t],cache,"decode")
    return out
def capture(w,engine,record,prompt,candidates,*,drift_limit=1e-4):
    need(record["split"]=="calibration" and record["id"].startswith("calibration:"),"calibration only")
    need(math.isfinite(drift_limit) and 0<=drift_limit<=1e-4,"bounded observer drift limit")
    need(engine.ckpt.kept is None and all(l.denied is None for l in engine.layers),"unmasked teacher")
    layers=set(engine.ckpt.sparse_layers);n=engine.ckpt.num_experts;k=engine.top_k
    need(candidates and len({len(ids) for masks in candidates.values() for ids in masks.values()})==1,"same budget")
    for name,masks in candidates.items():
        need(set(masks)==layers,"mask layer inventory "+name)
        for ids in masks.values():w.check_ids(ids,n,k)
    prefix=record["teacher_prefix_tokens"]
    need(len(prefix)==8 and record["decode_steps_1based"]==[1,8],"bounded prefix")
    need(canonical(prompt)==record["prompt_ids_sha256"] and canonical(prefix)==record["teacher_prefix_sha256"],"token pins")
    need(len(prompt)==record["prompt_tokens"] and record["prefill_rows"]==[len(prompt)//2-1,len(prompt)-1],"prefill row plan")
    need(len(prompt)>=2 and all(type(t)==int and 0<=t<engine.ckpt.config["vocab_size"] for t in prompt+prefix),"token IDs")
    # One unobserved teacher pass to bind actual logits and forced prefix.
    start=time.monotonic();baseline=forward_sequence(engine,prompt,prefix);plain_sec=time.monotonic()-start
    original=engine._moe;had_instance="_moe" in engine.__dict__;prior=engine.__dict__.get("_moe")
    state={"phase":None,"step":None};rows=[];visited=set()
    def set_step(phase,step):state.update(phase=phase,step=step)
    def observed(layer,x,phase):
        actual=original(layer,x,phase)
        need(phase==state["phase"],"observer phase")
        step=state["step"]
        if phase=="decode" and step not in record["decode_steps_1based"]:return actual
        if layer.idx not in layers:raise ValueError("unexpected layer")
        key=(layer.idx,phase,step);need(key not in visited,"duplicate observer visit");visited.add(key)
        selected=record["prefill_rows"] if phase=="prefill" else [0]
        z=w.witness(layer,x,selected,k,engine.router_scaling,actual,engine_module.expert_combine,sample_id=record["id"],direction=record["direction"],phase=phase,split="calibration")
        # No output arrays kept across layer calls; explicit drift bound.
        need(max(z["observer_routed_drift"],z["observer_combined_drift"])<=drift_limit,"observer reconstruction drift at "+repr(key))
        losses={name:w.errors(z,masks[layer.idx]) for name,masks in candidates.items()}
        rows.append({"layer":layer.idx,"phase":phase,"step":step,"row_count":len(selected),"routed_drift":z["observer_routed_drift"],"combined_drift":z["observer_combined_drift"],"energy":z["energy"].tolist(),"losses":losses})
        return actual
    start=time.monotonic()
    try:
        engine._moe=observed
        current=forward_sequence(engine,prompt,prefix,set_step)
    finally:
        if had_instance:engine._moe=prior
        else:del engine._moe
    observed_sec=time.monotonic()-start
    need(current==baseline,"observer changed logits or teacher prefix")
    expect={(l,p,j) for l in layers for p,j in (("prefill",0),("decode",1),("decode",8))}
    need(visited==expect,"observer coverage")
    return {"status":"LOCAL_OUTPUT_PROXY_ONLY","id":record["id"],"direction":record["direction"],"line":record["line"],"teacher_logits":baseline,"logit_equivalence":True,"plain_seconds":plain_sec,"observed_seconds":observed_sec,"candidate_masks":{name:canonical({str(l):ids for l,ids in masks.items()}) for name,masks in candidates.items()},"observations":rows,"max_combined_drift":max(x["combined_drift"] for x in rows),"max_routed_drift":max(x["routed_drift"] for x in rows),"token_expert_evaluations":sum(x["row_count"] for x in rows)*n}
def pilot_inputs(plan,corpus,baseline,original,guarded):
    # Exact frozen inputs, independently check line disjointness and every row binding.
    need(blob(plan)=="3c92bd0a944637ee7eb8d36d271417396b313026","pilot blob")
    b=corpus.read_bytes();r=baseline.read_bytes()
    need(hashlib.sha256(b).hexdigest()=="9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4","corpus")
    need(hashlib.sha256(r).hexdigest()=="37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8","baseline")
    cs={x["id"]:x for x in map(json.loads,b.splitlines()) if x["split"]=="calibration"}
    _,*br=map(json.loads,r.splitlines());rr={x["id"]:x for x in br if x["split"]=="calibration"}
    need(len(cs)==len(rr)==144 and set(cs)==set(rr),"calibration coverage")
    data=json.loads(plan.read_bytes());groups=data["plan"];need(set(groups)=={"fit","holdout"},"pilot subsets")
    all_rows=groups["fit"]+groups["holdout"]
    need(len(all_rows)==len({x["line"] for x in all_rows})==len({x["id"] for x in all_rows})==24,"aligned line isolation")
    dirs={"zh-my","my-zh","zh-en","en-zh","en-my","my-en"}
    for subset,rows in groups.items():
        need(Counter(x["direction"] for x in rows)==dict.fromkeys(dirs,2),"all direction coverage")
        for q in rows:
            c=cs[q["id"]];t=rr[q["id"]]
            need(q["line"]==c["line"]==t["line"] and q["direction"]==c["direction"]==t["direction"],"sample binding")
            need(q["prompt_tokens"]==len(c["prompt_ids"])==t["prompt_tokens"] and q["prompt_ids_sha256"]==canonical(c["prompt_ids"]),"prompt binding")
            need(t["finish"]=="eos" and q["teacher_prefix_tokens"]==t["tokens"][:8] and q["teacher_prefix_sha256"]==canonical(t["tokens"][:8]),"prefix binding")
    need(blob(original)=="d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed","original selection blob")
    need(blob(guarded)=="150616a10fbf503f815fb2f7fc281a81dcc28fea","guarded selection blob")
    a=json.loads(original.read_bytes())["candidates"]["96"]["kept"];b=json.loads(guarded.read_bytes())["plans"]["global_guarded96"]["kept"]
    need(canonical(a)=="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7" and canonical(b)=="4589d613512ac0a07f02be500e8b2fabe3f4beb78ddd171e14fd26cd8078960c","mask digests")
    need(set(a)==set(b)=={str(i) for i in range(1,48)} and all(len(v)==96 and v==sorted(set(v)) and all(type(i)==int and 0<=i<128 for i in v) for v in list(a.values())+list(b.values())),"real mask shape")
    first=groups["fit"][0];return dict(first,split="calibration"),cs[first["id"]]["prompt_ids"],{"original96":{int(l):ids for l,ids in a.items()},"guarded96":{int(l):ids for l,ids in b.items()}}
def selftest(w):
    torch.set_num_threads(1);torch.manual_seed(19)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d);model=tiny_checkpoint(p/"tiny",p/"none",vocab_size=256,seed=19)
        engine=engine_module.Engine(engine_module.Checkpoint(model),threads=1)
        prompt=[4,8,15,16,23,42];prefix=greedy_prefix(engine,prompt,8)
        rec={"id":"calibration:zh-my:synthetic","direction":"zh-my","line":0,"split":"calibration","prompt_tokens":len(prompt),"prompt_ids_sha256":canonical(prompt),"teacher_prefix_tokens":prefix,"teacher_prefix_sha256":canonical(prefix),"prefill_rows":[2,5],"decode_steps_1based":[1,8]}
        masks={"a":{1:[0,2,4,6],2:[1,2,5,7]},"b":{1:[1,3,5,7],2:[0,3,4,6]}}
        got=capture(w,engine,rec,prompt,masks)
        need(len(got["observations"])==6 and got["token_expert_evaluations"]==64,"synthetic coverage")
        need("_moe" not in engine.__dict__,"observer restored")
        need(forward_sequence(engine,prompt,prefix)==got["teacher_logits"],"post-restore logits")
        need(all(set(x["losses"])=={"a","b"} for x in got["observations"]),"mask comparison")
        def reject(fn):
            try:fn()
            except ValueError:return
            raise AssertionError("expected rejection")
        reject(lambda:capture(w,engine,dict(rec,split="validation"),prompt,masks))
        reject(lambda:capture(w,engine,dict(rec,prompt_ids_sha256="0"*64),prompt,masks))
        reject(lambda:capture(w,engine,rec,prompt,{"bad":{1:[0,2,4,6]}}))
        reject(lambda:capture(w,engine,rec,prompt,{"a":masks["a"],"bad":{1:[0,1],2:[0,1]}}))
        reject(lambda:capture(w,engine,rec,prompt,masks,drift_limit=1.0))
        bad=prefix.copy();bad[0]=(bad[0]+1)%256
        reject(lambda:capture(w,engine,dict(rec,teacher_prefix_tokens=bad,teacher_prefix_sha256=canonical(bad)),prompt,masks))
        print(json.dumps({"status":"PASS_SYNTHETIC_ENGINE_INTEGRATION","torch":torch.__version__,"forward_logit_checks":len(got["teacher_logits"]),"observations":len(got["observations"]),"token_expert_evaluations":got["token_expert_evaluations"],"negative_checks":6,"max_routed_drift":got["max_routed_drift"],"max_combined_drift":got["max_combined_drift"],"observer_restored":True,"real_capture":"NOT_RUN"}))
def main():
    a=argparse.ArgumentParser();a.add_argument("--witness",type=Path,required=True);a.add_argument("--selftest",action="store_true");a.add_argument("--validate-inputs",nargs=5,type=Path)
    q=a.parse_args();verify_code();w=load_witness(q.witness)
    if q.selftest:selftest(w)
    if q.validate_inputs:
        rec,prompt,masks=pilot_inputs(*q.validate_inputs)
        print(json.dumps({"status":"REAL_PILOT_INPUTS_VERIFIED_NOT_EXECUTED","first_fit_id":rec["id"],"prompt_tokens":len(prompt),"teacher_prefix_tokens":len(rec["teacher_prefix_tokens"]),"masks":{n:canonical({str(l):ids for l,ids in v.items()}) for n,v in masks.items()},"fit_holdout_aligned_lines":24,"layers":47}))
    need(q.selftest or q.validate_inputs,"no real capture CLI until R21 admission")
if __name__=="__main__":main()

