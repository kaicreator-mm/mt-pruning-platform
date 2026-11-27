"""R23 sparse real pilot with same-hidden-state exhaustive oracle.

Default only preflight; first real sample only with explicit R21 admission.
For first sparse layer, compare all three observed positions against exhaustive
expert outputs. No candidate selection or extra real-model generation.
"""
import argparse, json, resource, tempfile, time
from pathlib import Path
from types import SimpleNamespace
import torch
import hashlib,importlib.util
P=Path
DRIVER_BLOB="0735238e15f13fd75e3f6939fc31aa834d8b5854"
SPARSE_BLOB="1c38f6e928ff653fbbef584b298ff65e194900fe"
def need(x,m):
    if not x:raise ValueError(m)
def load(p,pin,name):
    b=p.read_bytes();need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==pin,name+" blob")
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def capture(c,w,s,engine,rec,prompt,masks):
    adapter=s.adapter(w,masks)
    sparse=adapter.witness;counts=[];first=min(engine.ckpt.sparse_layers)
    def checked(layer,*a,**kw):
        z=sparse(layer,*a,**kw)
        if layer.idx==first:
            full=w.witness(layer,*a,**kw)
            need(torch.equal(z["values"][:,z["computed"]],full["values"][:,z["computed"]]),"real expert bytes")
            need(z["observer_routed_drift"]==full["observer_routed_drift"] and z["observer_combined_drift"]==full["observer_combined_drift"] and torch.equal(z["energy"],full["energy"]),"real reconstruction/energy")
            for ids in masks.values():
                need(s.errors(w,z,ids[layer.idx])==w.errors(full,ids[layer.idx]),"real mask errors")
            counts.append({"layer":layer.idx,"phase":z["phase"],"rows":len(z["rows"]),"candidate_comparisons":len(masks),"exhaustive_extra_evaluations":len(z["rows"])*len(layer.gate)})
        return z
    adapter.witness=checked
    out=c.capture(adapter,engine,rec,prompt,masks)
    nominal=sum(x["nominal"] for x in adapter.counters);actual=sum(x["actual"] for x in adapter.counters)
    need(nominal==out["token_expert_evaluations"] and 0<actual<=nominal,"compute counts")
    need(len(counts)==3 and sum(x["rows"] for x in counts)==4,"oracle coverage")
    out.update(sparse_witness_blob=SPARSE_BLOB,nominal_exhaustive_evaluations=nominal,sparse_counterfactual_evaluations=actual,extra_oracle_evaluations=sum(x["exhaustive_extra_evaluations"] for x in counts),exhaustive_oracle=counts,expert_compute_counts=adapter.counters)
    return out
def selftest(c,w,s):
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as d:
        p=P(d);model=c.tiny_checkpoint(p/"tiny",p/"none",vocab_size=256,seed=19)
        engine=c.engine_module.Engine(c.engine_module.Checkpoint(model),threads=1)
        prompt=[4,8,15,16,23,42];prefix=c.greedy_prefix(engine,prompt,8)
        rec={"id":"calibration:zh-my:synthetic","direction":"zh-my","line":0,"split":"calibration","prompt_tokens":len(prompt),"prompt_ids_sha256":c.canonical(prompt),"teacher_prefix_tokens":prefix,"teacher_prefix_sha256":c.canonical(prefix),"prefill_rows":[2,5],"decode_steps_1based":[1,8]}
        masks={"a":{1:[0,2,4,6],2:[1,2,5,7]},"b":{1:[1,3,5,7],2:[0,3,4,6]}}
        base=c.capture(w,engine,rec,prompt,masks);got=capture(c,w,s,engine,rec,prompt,masks)
        need(got["observations"]==base["observations"] and got["teacher_logits"]==base["teacher_logits"],"complete observer oracle")
        need("_moe" not in engine.__dict__,"restore")
        need(got["extra_oracle_evaluations"]==32 and sum(x["candidate_comparisons"] for x in got["exhaustive_oracle"])==6,"oracle accounting")
        print(json.dumps({"status":"PASS_SPARSE_PILOT_INTEGRATION_SYNTHETIC","logit_checks":len(got["teacher_logits"]),"observations":len(got["observations"]),"nominal":got["nominal_exhaustive_evaluations"],"sparse":got["sparse_counterfactual_evaluations"],"extra_oracle":got["extra_oracle_evaluations"],"oracle_candidate_checks":6,"real_capture":"NOT_RUN"}),flush=True)
def main():
    p=argparse.ArgumentParser()
    for n in ("driver","collector","witness","selector","sparse","plan","original","guarded","p04"):p.add_argument("--"+n,type=P,required=True)
    p.add_argument("--selftest",action="store_true");p.add_argument("--execute",action="store_true");p.add_argument("--admission",type=P);p.add_argument("--admission-sha256")
    q=p.parse_args();d=load(q.driver,DRIVER_BLOB,"driver");s=load(q.sparse,SPARSE_BLOB,"sparse")
    c=d.load(q.collector,d.COLLECTOR_BLOB,"collector");selector=d.load(q.selector,d.SELECTOR_BLOB,"selector");c.verify_code();w=c.load_witness(q.witness)
    rec,prompt,_=c.pilot_inputs(q.plan,d.ROOT/"run-d01b-r19/corpus/corpus.jsonl",d.ROOT/"run-d01b-r19/baseline/records.jsonl",q.original,q.guarded)
    need(rec["id"]==d.FIRST,"first sample")
    masks=selector.masks_from(q.original,q.guarded,q.p04);pins={n:selector.canon(m) for n,m in masks.items()}
    if q.selftest:selftest(c,w,s)
    result=d.preflight();result.update(sample_id=d.FIRST,candidate_masks=pins,nominal_exhaustive_evaluations=24064,extra_exhaustive_oracle_evaluations=512,sparse_evaluations="MEASURE_ON_REAL_CAPTURE")
    print(json.dumps(result),flush=True)
    if not q.execute:return
    need(not result["blockers"] and q.admission is not None and q.admission_sha256 is not None,"blocked or missing admission")
    raw=q.admission.read_bytes();need(d.sha(raw)==q.admission_sha256,"admission digest")
    a=json.loads(raw);d.validate_admission(a,d.FIRST,pins)
    need(a.get("sparse_witness_blob")==SPARSE_BLOB and a.get("pilot_mode")=="SPARSE_WITH_FIRST_LAYER_EXHAUSTIVE_ORACLE","sparse admission")
    need(not d.preflight()["blockers"],"resource changed")
    torch.set_num_threads(8);torch.set_num_interop_threads(1);started=time.monotonic()
    ident=d.checkpoint_identity(d.SOURCE,cache_dir=None,fresh=True,exact_inventory=True)
    need(ident["teacher"]==d.TEACHER and ident["kind"]=="pinned-snapshot" and ident["num_experts"]==128,"actual teacher bytes")
    need(not d.preflight()["blockers"],"resource changed after identity")
    engine=c.engine_module.Engine(c.engine_module.Checkpoint(d.SOURCE),threads=8)
    got=capture(c,w,s,engine,rec,prompt,{n:{int(l):ids for l,ids in m.items()} for n,m in masks.items()})
    need(got["nominal_exhaustive_evaluations"]==24064 and got["extra_oracle_evaluations"]==512 and len(got["observations"])==141,"real coverage")
    got.update(teacher_digest=ident["teacher"],admission_sha256=q.admission_sha256,r20_exact_head=a["r20_exact_head"],engine_head=d.ENGINE_HEAD,elapsed_with_identity_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,torch=torch.__version__)
    print(json.dumps(got,allow_nan=False,separators=(",",":")),flush=True)
if __name__=="__main__":main()

