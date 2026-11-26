"""R21 matched v2 baseline scoring, read-only, no test or candidate fitting."""
import ast,hashlib,json,pathlib,re,unicodedata,collections,math
import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
p=pathlib.Path.home()/".cache/mt-pruning/r20-r1"
src=(pathlib.Path.home()/"mt-pruning-platform/mtpruning/evaluate.py").read_bytes()
need(hashlib.sha1(b"blob "+str(len(src)).encode()+b"\0"+src).hexdigest()=="ac122a7f46f0381cba5a2b019c42e391c670d968","metric blob")
names={"digit_runs","script_share","failure_flags","chrf_pp","segment_stats","score_from_stats","paired_bootstrap"}
nodes=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_DIGITS" for t in n.targets)]
need(len(nodes)==8 and sacrebleu.__version__=="2.6.0" and np.__version__=="2.5.5","pinned metric")
env={"re":re,"unicodedata":unicodedata,"np":np,"CHRF":CHRF};exec(compile(ast.Module(body=nodes,type_ignores=[]),"pinned-metric","exec"),env)
metric=env["chrf_pp"]();c=(p.parent/"run-d01b-r19/corpus/corpus.jsonl").read_bytes()
need(sha(c)=="9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4","corpus")
ctx={x["id"]:x for x in map(json.loads,c.splitlines()) if x["split"]=="validation"};need(len(ctx)==120,"contexts")
b=(p.parent/"run-d01b-r19/baseline/records.jsonl").read_bytes()
need(sha(b)=="37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8","bf16 records")
_,*rows=map(json.loads,b.splitlines());bf={r["id"]:r for r in rows if r["split"]=="validation"};need(set(bf)==set(ctx),"bf16 IDs")
def load(name,expected_sha=None):
    root=p/"runs"/name;raw=(root/"runs.jsonl").read_bytes()
    if expected_sha:need(sha(raw)==expected_sha,"exact record hash "+name)
    rs=list(map(json.loads,raw.splitlines()));need(len(rs)==120 and {r["id"] for r in rs}==set(ctx),"run coverage")
    for r in rs:
        i=r["id"];f=root/(i.replace(":","__")+".output.txt");data=f.read_bytes();s=data.decode("utf-8")
        need(r["exit_code"]==0 and r["direction"]==ctx[i]["direction"] and str(f)==r["output_file"] and sha(data)==r["output_sha256"] and s==r["raw_output"] and "Assistant:\n" in s and s.split("Assistant:\n",1)[1].strip()==r["text"],"actual output "+i)
    return {r["id"]:r for r in rs},sha(raw)
v1,h1=load("full-f16-k128","ad49d71d6ad5da68512445f71d9df865922b5a07759cd9a9b2a6c40173648437")
v2,h2=load("full-f16-k128-v2","2b977c4b18a697b24057b140f6c9ef3f7298ba0b959bdd1900822cd8e5d169db")
for r in v2.values():need(r["finish"]=="eos" and r["stop_type"]=="eos" and r["tokens"][-1]==120025 and 120025 not in r["tokens"][:-1] and len(r["tokens"])==r["predicted_tokens"]==r["server_timings"]["predicted_n"],"v2 EOS")
def flags(r,normalized):
    i=r["id"];x=ctx[i];ts=r["tokens"][:-1] if normalized else r["tokens"]
    return env["failure_flags"](x["source"],x["reference"],r["text"],x["tgt_lang"],r["finish"],ts)
results={}
for d in ("zh-my","my-zh","zh-en","en-zh","en-my","my-en"):
    ids=sorted(i for i,x in ctx.items() if x["direction"]==d);need(len(ids)==20,"direction")
    refs=[ctx[i]["reference"] for i in ids]
    stats={name:env["segment_stats"](metric,[rs[i]["text"] for i in ids],refs) for name,rs in (("bf16",bf),("fp16_v1",v1),("fp16_v2",v2))}
    scores={name:env["score_from_stats"](metric,rr) for name,rr in stats.items()}
    base_flags=collections.Counter(x for i in ids for x in flags(bf[i],False))
    norm_flags=collections.Counter(x for i in ids for x in flags(v2[i],True))
    raw_flags=collections.Counter(x for i in ids for x in flags(v2[i],False))
    results[d]={"n":len(ids),"scores":scores,"v2_vs_v1":env["paired_bootstrap"](metric,stats["fp16_v1"],stats["fp16_v2"]),"v2_vs_bf16":env["paired_bootstrap"](metric,stats["bf16"],stats["fp16_v2"]),"v2_same_text_v1":sum(v2[i]["text"]==v1[i]["text"] for i in ids),"v2_same_text_bf16":sum(v2[i]["text"]==bf[i]["text"] for i in ids),"bf16_flags":dict(base_flags),"v2_reference_normalized_flags":dict(norm_flags),"v2_raw_token_flags":dict(raw_flags)}
wall=sum(r["wall_seconds"] for r in v2.values())
t=[r["server_timings"] for r in v2.values()]
print(json.dumps({"status":"V2_BASELINE_BRIDGE_DESCRIPTIVE_ONLY","metric_signature":str(metric.get_signature()),"v1_sha256":h1,"v2_sha256":h2,"results":results,"same_text_v1":sum(x["v2_same_text_v1"] for x in results.values()),"same_text_bf16":sum(x["v2_same_text_bf16"] for x in results.values()),"timing":{"wall_seconds":wall,"prompt_ms":sum(x["prompt_ms"] for x in t),"prompt_tokens":sum(x["prompt_n"] for x in t),"decode_ms":sum(x["predicted_ms"] for x in t),"decode_tokens":sum(x["predicted_n"] for x in t),"cache_tokens":sum(x.get("cache_n",0) for x in t),"sampled_device_vram_peak_mib":max(r["peak_vram_mib"] for r in v2.values())},"limitations":["FP16 v1 mixed direct/server, v2 all server; do not infer pure precision or backend effect.","V2 raw token records include terminal EOS; reference-style diagnostic removes verified final EOS without rewriting raw.","Baseline bridge not a pruning candidate or mechanism qualification. No final-test scoring."]},separators=(",",":"),allow_nan=False))

