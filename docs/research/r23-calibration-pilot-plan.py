"""R23 bounded real-capture pilot plan, selection on R19 calibration only."""
import hashlib,json,pathlib,collections
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
p=pathlib.Path.home()/".cache/mt-pruning/run-d01b-r19"
c=(p/"corpus/corpus.jsonl").read_bytes();r=(p/"baseline/records.jsonl").read_bytes()
need(sha(c)=="9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4","corpus")
need(sha(r)=="37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8","records")
cs={x["id"]:x for x in map(json.loads,c.splitlines()) if x["split"]=="calibration"}
header,*rows=map(json.loads,r.splitlines());rr={x["id"]:x for x in rows if x["split"]=="calibration"}
need(len(cs)==len(rr)==144 and set(cs)==set(rr),"coverage")
dirs=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
# Cross-direction aligned sentences cannot leak between fit/holdout.
# A deterministic hash order with globally unique source line IDs avoids
# overlap even within a subset. No translation quality or outputs scored.
salt="mt-r23-local-output-pilot-v1"
used=set();plan={"fit":[],"holdout":[]}
for subset in ("fit","holdout"):
    for d in dirs:
        pool=[]
        for i,x in cs.items():
            y=rr[i]
            need(x["direction"]==y["direction"] and x["line"]==y["line"] and len(x["prompt_ids"])==y["prompt_tokens"],"identity")
            if x["direction"]==d and y["finish"]=="eos" and len(y["tokens"])>=8 and len(x["prompt_ids"])>=2 and x["line"] not in used:
                pool.append((sha((salt+":"+subset+":"+i).encode()),i))
        need(len(pool)>=2,"insufficient eligible "+d)
        chosen=[]
        for _,i in sorted(pool):
            x=cs[i];y=rr[i]
            if x["line"] in used:continue
            used.add(x["line"]);chosen.append(i)
            # Prefill observer samples 2 rows in actual full prompt; teacher
            # forced decode only through first 8 known teacher tokens.
            # One selected row at steps 1 and 8, each expert evaluated locally.
            plan[subset].append({"id":i,"direction":d,"line":x["line"],"prompt_tokens":len(x["prompt_ids"]),"prompt_ids_sha256":sha(json.dumps(x["prompt_ids"],separators=(",",":")).encode()),"prefill_rows":[len(x["prompt_ids"])//2-1,len(x["prompt_ids"])-1],"decode_steps_1based":[1,8],"teacher_prefix_tokens":y["tokens"][:8],"teacher_prefix_sha256":sha(json.dumps(y["tokens"][:8],separators=(",",":")).encode())})
            if len(chosen)==2:break
        need(len(chosen)==2,"selection coverage")
need(len(used)==24 and all(len(v)==12 for v in plan.values()),"globally disjoint")
need(not ({x["line"] for x in plan["fit"]}&{x["line"] for x in plan["holdout"]}),"line isolation")
out={"status":"FROZEN_CALIBRATION_PILOT_PLAN_NOT_DISPATCHED","source_head":"3212303e1e02994a5124db8c85195b91325c7a7c","corpus_sha256":sha(c),"records_sha256":sha(r),"salt":salt,"sample_count":24,"aligned_line_count":24,"rows_per_layer_per_sample":4,"sparse_layers":47,"expert_count":128,"planned_local_token_expert_evaluations":24*4*47*128,"plan":plan,"limits":["Only calibration. Existing teacher prefix is reused for bounded teacher-forced input, not new generation.","Run one sample first; compare observer drift, runtime/memory before remaining pilot. Do not auto-run if GPU/IO busy.","The pilot is not sufficient proof of robust quality or full calibration; fit/holdout group coverage is a minimum."]} 
print(json.dumps(out,separators=(",",":")))

