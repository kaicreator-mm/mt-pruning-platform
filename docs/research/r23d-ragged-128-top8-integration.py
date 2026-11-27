"""Actual 128-wide routing, top8 and all five budgets on tiny random HY-V3.
Uses original pinned engine and ragged adapter, no real teacher.
"""
import argparse,hashlib,importlib.util,json,pathlib,tempfile,torch
P=pathlib.Path
ADAPTER="56ceecad1fd2db0d33399840e2231c718c7e6ab0"
def need(x,m):
    if not x:raise ValueError(m)
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("--adapter",type=P,required=True);q=p.parse_args()
    need(blob(q.adapter.read_bytes())==ADAPTER,"adapter pin")
    spec=importlib.util.spec_from_file_location("ragged",q.adapter);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    from mtpruning.testing import tiny_checkpoint
    from safetensors.torch import save_file
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as d:
        root=P(d);src=tiny_checkpoint(root/"source",root/"none",vocab_size=256,seed=101,num_experts=128,top_k=8,layers=6)
        budgets=(64,80,96,112,128);g=torch.Generator().manual_seed(2026)
        mask={str(i):torch.randperm(128,generator=g)[:k].sort().values.tolist() for i,k in enumerate(budgets,1)}
        dest=root/"compact";m=r.build_fixture(src,dest,mask);mh=r.sha(r.read(dest/"ragged_manifest.json"))
        ck=r.RaggedCheckpoint(dest,mh);orig=r.eng.Checkpoint(src)
        a=r.eng.Engine(orig,1);a.set_mask({int(l):v for l,v in mask.items()});b=r.RaggedEngine(ck,1)
        need(sum(map(len,mask.values()))==480 and sum(len(l.gate) for l in b.layers if l.sparse)==480,"actual compact FFN count")
        old_keys=r.eng.expected_keys(ck.config,ck.sparse)
        need(len(old_keys)-len(ck.source_map)==(640-480)*3,"removed tensor inventory")
        need(all(torch.equal(ck.tensor(k),orig.tensor(v)) for k,v in ck.source_map.items()),"physical tensors")
        checks=0;routes=0;gens=0
        for tie in (False,True):
            if tie:
                for e in (a,b):
                    for l in e.layers:
                        if l.sparse:l.router=torch.zeros_like(l.router);l.bias=torch.zeros_like(l.bias)
            for prompt in ([4,8,15,16,23,42],[17,7,31,9,3,2,5,8]):
                x,y=r.compare_fixed(a,b,prompt,[11,12,13,14,15,16,17,18]);checks+=x;routes+=y
                x=a.generate(prompt,8,trace=True);y=b.generate(prompt,8,trace=True)
                need(x.tokens==y.tokens and x.finish==y.finish and all(torch.equal(v,y.trace[k]) for k,v in x.trace.items()),"exact greedy trace")
                gens+=1
        def rejected(f):
            try:f()
            except (ValueError,OSError):return
            raise AssertionError("invalid fixture accepted")
        # Rehash intentionally malformed payloads: digest checks alone must not authorize wrong shapes or extra keys.
        for kind in ("router","ffn","extra"):
            bad=root/kind;bad.mkdir()
            (bad/"config.json").write_bytes(r.read(dest/"config.json"))
            tensors={k:ck.tensor(k).clone() for k in ck.source_map}
            if kind=="router":tensors["model.layers.1.mlp.router.gate.weight"]=tensors["model.layers.1.mlp.router.gate.weight"][:-1].contiguous()
            if kind=="ffn":tensors["model.layers.1.mlp.experts.0.gate_proj.weight"]=tensors["model.layers.1.mlp.experts.0.gate_proj.weight"][:-1].contiguous()
            if kind=="extra":tensors["unexpected.weight"]=torch.zeros(1)
            save_file(tensors,str(bad/"compact.safetensors"))
            z=dict(m,payload_sha256=r.fsha(bad/"compact.safetensors"),tensor_count=len(tensors));raw=r.canon(z)+b"\n";(bad/"ragged_manifest.json").write_bytes(raw)
            rejected(lambda:r.RaggedCheckpoint(bad,r.sha(raw)))
        print(json.dumps(dict(status="PASS_CPU_RAGGED_128_TOP8",torch=str(torch.__version__),layer_counts=budgets,physical_expert_instances=480,removed_expert_instances=160,removed_ffn_tensors=480,physical_tensor_checks=len(ck.source_map),fixed_logit_checks=checks,route_hook_checks=routes,greedy_trace_checks=gens,rehash_invalid_payload_rejections=3,actual_teacher="NOT_RUN",native_llama_cpp="NOT_RUN")))
if __name__=="__main__":main()
