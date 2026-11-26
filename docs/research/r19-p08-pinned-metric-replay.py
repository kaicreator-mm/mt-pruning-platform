"""P08: pinned metric replay of existing R19 validation, no inference."""
import ast,hashlib,json,pathlib,re,sys,unicodedata,platform
from collections import Counter
import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF
root=pathlib.Path(sys.argv[1])
def need(v,msg):
    if not v: raise ValueError(msg)
def raw(n):return (root/n).read_bytes()
def read(n):return json.loads(raw(n))
def lines(n):return [json.loads(x) for x in raw(n).splitlines() if x]
pins={"corpus.jsonl":"5fc8e3586a486e89ca6e0e6644d6c79ad1e4de72","manifest.json":"05322b3a96e06d2a7676dcc39b6efccd15093bfb","baseline.jsonl":"8b2b9a3394c52973201115913374a8a8e1e29749","physical.jsonl":"7593b21a0395650fca1094d2b2900a8f6a4b94c4","masked.jsonl":"1423d039695659dea7d4ea074907bb258d7a825c","evaluate.py":"ac122a7f46f0381cba5a2b019c42e391c670d968"}
for n,v in pins.items():
    b=raw(n);need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==v,n+" Git blob mismatch")
need(sacrebleu.__version__=="2.6.0" and np.__version__=="2.5.5","metric dependency pin")
manifest=read("manifest.json")
need(hashlib.sha256(raw("corpus.jsonl")).hexdigest()==manifest["corpus_sha256"]=="9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4","corpus digest")
dirs=manifest["directions"]
expected={f"validation:{d}:{i}" for d in dirs for i in manifest["indices"][d.split("-")[0]]["validation"]}
ctx={x["id"]:x for x in lines("corpus.jsonl") if x["split"]=="validation"}
need(len(ctx)==len(expected)==120 and set(ctx)==expected,"validation corpus IDs")
common=("teacher","corpus","prompt","runtime","code","decoding","mode")
def load(n,k,mask,count,anchor=None):
    h,*rows=lines(n)
    need(h["type"]=="run" and h["mode"]=="real" and h["synthetic"] is False and h["num_experts"]==k and h["masked_keep"]==mask and h["corpus_sha256"]==manifest["corpus_sha256"] and h["max_new_tokens"]==640 and h["decoding"]=="greedy (argmax of bf16 logits)" and "test" not in h["splits"],"header "+n)
    need(all(x["split"]!="test" for x in rows),"test output present")
    if anchor:need(all(h["identity"][key]==anchor["identity"][key] for key in common),"shared identity "+n)
    vs=[x for x in rows if x["split"]=="validation"];m={x["id"]:x for x in vs}
    need(len(m)==len(vs)==count and set(m)<=expected,"coverage "+n)
    for x in vs:
        c=ctx[x["id"]]
        need(x["direction"]==c["direction"] and x["line"]==c["line"] and x["prompt_tokens"]==len(c["prompt_ids"]) and x["new_tokens"]==len(x["tokens"]) and x["new_tokens"]<=640 and x["identity_key"]==h["identity_keys"]["validation"] and x["finish"] in ("eos","length"),"context "+n)
    return h,m
bh,b=load("baseline.jsonl",128,None,120)
ph,p=load("physical.jsonl",96,None,48,bh)
mh,m=load("masked.jsonl",128,96,48,bh)
need(set(p)==set(m)=={i for i,x in b.items() if x["traced"]},"fixed witness sets")
# Execute the actual pinned metric/flag functions, without importing torch or
# changing them. AST extraction is intentional and limited to named definitions.
names={"digit_runs","script_share","failure_flags","chrf_pp","chrf","segment_stats","score_from_stats","paired_bootstrap"}
tree=ast.parse(raw("evaluate.py"));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_DIGITS" for t in n.targets)]
need({n.name for n in nodes if isinstance(n,ast.FunctionDef)}==names and len(nodes)==len(names)+1,"function extraction")
env={"re":re,"unicodedata":unicodedata,"np":np,"CHRF":CHRF}
exec(compile(ast.Module(body=nodes,type_ignores=[]),"pinned-evaluate-functions","exec"),env)
metric=env["chrf_pp"]()
need(abs(metric.corpus_score(["abc"],[["abc"]]).score-100)<1e-10,"identity score")
need(metric.corpus_score(["abc"],[["xyz"]]).score==0,"disjoint score")
need(env["digit_runs"]("၁၂ ３４")==["12","34"],"digit folding")
flag_keys=("empty","truncated","untranslated_copy","wrong_script","digits_differ_from_reference","length_ratio_outlier","degenerate_repetition")
def result(rs,ids):
    hyp=[rs[i]["text"] for i in ids];ref=[ctx[i]["reference"] for i in ids]
    stat=env["segment_stats"](metric,hyp,ref)
    v=metric.corpus_score(hyp,[ref]).score
    need(abs(v-env["score_from_stats"](metric,stat))<1e-9,"corpus statistics replay")
    flags={i:env["failure_flags"](ctx[i]["source"],ctx[i]["reference"],rs[i]["text"],ctx[i]["direction"].split("-")[1],rs[i]["finish"],rs[i]["tokens"]) for i in ids}
    counts={k:sum(k in f for f in flags.values()) for k in flag_keys}
    return {"n":len(ids),"chrfpp":v,"chrf":env["chrf"]().corpus_score(hyp,[ref]).score,"flags":counts,"flag_ids":{k:[i for i,f in flags.items() if k in f] for k in flag_keys}},stat
out={}
for d in dirs:
    ids=sorted(i for i in expected if ctx[i]["direction"]==d);sub=[i for i in ids if i in p]
    need(len(ids)==20 and len(sub)==8,"direction coverage")
    full,_=result(b,ids);bs,bst=result(b,sub);ps,pst=result(p,sub);ms,mst=result(m,sub)
    boot=env["paired_bootstrap"](metric,bst,pst,2000,42);bootm=env["paired_bootstrap"](metric,bst,mst,2000,42)
    bootpm=env["paired_bootstrap"](metric,mst,pst,2000,42)
    delta=ps["chrfpp"]-bs["chrfpp"]
    need(abs(boot["delta"]-delta)<1e-9,"delta")
    out[d]={"baseline20":full,"baseline8":bs,"physical8":ps,"masked8":ms,"physical_delta":delta,"physical_bootstrap":boot,"masked_bootstrap":bootm,"physical_minus_masked_bootstrap":bootpm,"exact_text_equal_to_baseline":sum(p[i]["text"]==b[i]["text"] for i in sub),"physical_vs_masked_exact_text":sum(p[i]["text"]==m[i]["text"] for i in sub)}
print(json.dumps({"status":"PINNED_METRIC_REPLAY_EXPLORATORY_SUBSET_ONLY","versions":{"python":platform.python_version(),"sacrebleu":sacrebleu.__version__,"numpy":np.__version__},"metric_signature":str(metric.get_signature()),"input_blobs":pins,"shared_identity_keys":list(common),"results":out},allow_nan=False,separators=(",",":")))

