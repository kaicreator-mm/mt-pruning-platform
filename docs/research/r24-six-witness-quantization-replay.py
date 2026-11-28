"""R24 preparatory replay of existing six-witness FP16/Q4 records.
One witness per direction; descriptive only, not a quality gate.
"""
import ast,hashlib,json,re,pathlib
import sacrebleu
from sacrebleu.metrics import CHRF
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
p=pathlib.Path.home()/".cache/mt-pruning/r20-r1"
source=(pathlib.Path.home()/"mt-pruning-platform/mtpruning/evaluate.py").read_bytes()
need(hashlib.sha1(b"blob "+str(len(source)).encode()+b"\0"+source).hexdigest()=="ac122a7f46f0381cba5a2b019c42e391c670d968","metric blob")
nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ("digit_runs","chrf_pp") or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_DIGITS" for t in n.targets)]
need(len(nodes)==3 and sacrebleu.__version__=="2.6.0","metric source/version")
env={"re":re,"CHRF":CHRF};exec(compile(ast.Module(body=nodes,type_ignores=[]),"pinned-metric","exec"),env)
metric=env["chrf_pp"]()
corpus=(p.parent/"run-d01b-r19/corpus/corpus.jsonl").read_bytes()
need(sha(corpus)=="9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4","corpus")
ctx={x["id"]:x for x in map(json.loads,corpus.splitlines()) if x["split"]=="validation"}
w=json.loads((p/"witnesses/manifest.json").read_bytes())["witnesses"];ws={x["id"]:x for x in w}
need(len(w)==len(ws)==6 and set(ws)<=set(ctx),"witnesses")
for i,x in ws.items():need(x["prompt_ids"]==ctx[i]["prompt_ids"] and sha(pathlib.Path(x["prompt_file"]).read_bytes())==x["prompt_sha256"],"prompt "+i)
groups={};receipts={}
for name,cn in [("f16-hybrid-ncmoe40","f16-ncmoe40-config.json"),("q4-cpu","q4-cpu-config.json"),("q4-full-gpu","q4-full-gpu-config.json")]:
    root=p/"runs"/name;conf=json.loads((p/cn).read_bytes());b=(root/"runs.jsonl").read_bytes();rows=list(map(json.loads,b.splitlines()))
    need(len(rows)==6 and {r["id"] for r in rows}==set(ws),"coverage")
    rdict={}
    for r in rows:
        i=r["id"];need(r["exit_code"]==0 and r["config"]==name and r["model"]==conf["model"] and r["direction"]==ctx[i]["direction"],"row identity")
        f=root/(i.replace(":","__")+".output.txt");need(str(f)==r["output_file"],"output path")
        data=f.read_bytes();s=data.decode("utf-8");need(sha(data)==r["output_sha256"] and s==r["raw_output"] and "Assistant:\n" in s and s.split("Assistant:\n",1)[1].strip()==r["text"],"actual output")
        cmd=r["command"];need("--server-base" not in cmd,"unexpected server mode")
        for opt,val in {"--ctx-size":"2048","--batch-size":"512","--threads":"16","--cache-type-k":"f16","--cache-type-v":"f16","--temp":"0","--top-k":"1","--top-p":"1","--seed":"0","--n-predict":"640"}.items():need(cmd.count(opt)==1 and cmd[cmd.index(opt)+1]==val,"command")
        need(cmd[cmd.index("--file")+1]==ws[i]["prompt_file"] and cmd[cmd.index("--model")+1]==conf["model"],"command model/prompt")
        need(isinstance(r["prompt_eval"],dict) and isinstance(r["decode_eval"],dict),"timings")
        rdict[i]=r
    groups[name]=rdict
    receipts[name]={"sha256":sha(b),"n":len(rows),"wall_seconds":sum(r["wall_seconds"] for r in rows),"prompt_ms":sum(r["prompt_eval"]["milliseconds"] for r in rows),"prompt_tokens":sum(r["prompt_eval"]["tokens"] for r in rows),"decode_ms":sum(r["decode_eval"]["milliseconds"] for r in rows),"decode_tokens":sum(r["decode_eval"]["tokens"] for r in rows),"max_device_vram_mib":max(r["peak_vram_mib"] for r in rows),"finish_missing":sum(r.get("finish") is None for r in rows),"actual_tokens_missing":sum(not isinstance(r.get("tokens"),list) for r in rows)}
pairs=[]
for i in sorted(ws):
    ref=ctx[i]["reference"];x={"id":i,"direction":ctx[i]["direction"],"scores":{}}
    for n,rs in groups.items():
        t=rs[i]["text"];x["scores"][n]={"chrfpp":metric.corpus_score([t],[[ref]]).score,"digit_flag":env["digit_runs"](t)!=env["digit_runs"](ref)}
    x["q4_cpu_gpu_identical"]=groups["q4-cpu"][i]["text"]==groups["q4-full-gpu"][i]["text"]
    x["f16_q4_gpu_identical"]=groups["f16-hybrid-ncmoe40"][i]["text"]==groups["q4-full-gpu"][i]["text"]
    pairs.append(x)
a,b,c=[receipts[n] for n in ("f16-hybrid-ncmoe40","q4-cpu","q4-full-gpu")]
print(json.dumps({"status":"SIX_WITNESS_REPLAY_ONLY","metric_signature":str(metric.get_signature()),"corpus_sha256":sha(corpus),"runs":receipts,"pairs":pairs,"speed":{"q4_cpu_to_gpu_wall":b["wall_seconds"]/c["wall_seconds"],"f16_hybrid_to_q4_gpu_wall":a["wall_seconds"]/c["wall_seconds"],"q4_cpu_decode_tps":b["decode_tokens"]*1000/b["decode_ms"],"q4_gpu_decode_tps":c["decode_tokens"]*1000/c["decode_ms"],"f16_hybrid_decode_tps":a["decode_tokens"]*1000/a["decode_ms"]},"limits":["One witness per direction, not gate or representative estimate.","FP16 hybrid vs Q4 GPU changes precision and placement; Q4 CPU vs GPU holds model path/precision but backend differs; timing includes model loads.","Missing actual generated token arrays and finish evidence; no formal diagnostics/qualification."]},ensure_ascii=False,separators=(",",":")))

