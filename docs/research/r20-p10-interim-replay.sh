#!/usr/bin/env bash
# Exact successful P10 replay command. Run only on authorized aliyun-sg with existing loopback SSH tunnel; V100 read-only.
set -eu
D=$(mktemp -d /tmp/mt-p10.XXXXXX)
trap 'rm -rf "$D"' EXIT
ssh -p 22022 -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 cc@localhost 'nice -n 19 python3 -B -' > "$D/snapshot.json" <<'REMOTE'
import json,hashlib,pathlib,datetime
p=pathlib.Path.home()/".cache/mt-pruning/r20-r1"
def sha(b):return hashlib.sha256(b).hexdigest()
def need(v,msg):
    if not v:raise ValueError(msg)
m=json.loads((p/"validation120/manifest.json").read_bytes())
need(m["source_head"]=="3212303e1e02994a5124db8c85195b91325c7a7c" and m["llama_source_commit"]=="9c003a97ab34a69cc49243f9b02014038205ffae","manifest identity")
w={x["id"]:x for x in m["witnesses"]};need(len(w)==len(m["witnesses"])==120,"witness set")
for x in w.values():
    need(sha(pathlib.Path(x["prompt_file"]).read_bytes())==x["prompt_sha256"],"prompt hash")
out={"captured":datetime.datetime.now(datetime.timezone.utc).isoformat(),"manifest_sha256":sha((p/"validation120/manifest.json").read_bytes()),"witnesses":[{"id":i,"prompt_ids":x["prompt_ids"],"prompt_sha256":x["prompt_sha256"]} for i,x in w.items()],"runs":{},"stage_receipts":{}}
for k in (128,96,64):
    root=p/f"runs/full-f16-k{k}";b=(root/"runs.jsonl").read_bytes()
    # An actively appended partial line is excluded, not treated as completed.
    lines=b.splitlines(keepends=True);partial=bool(lines and not lines[-1].endswith(b"\n"))
    if partial:lines=lines[:-1]
    rows=[json.loads(s) for s in lines if s.strip()]
    need(len({x["id"] for x in rows})==len(rows) and set(x["id"] for x in rows)<=set(w),"coverage")
    slim=[]
    for x in rows:
        i=x["id"];need(x["config"]==f"full-f16-k{k}" and x["model"]==str(p.parent/f"gguf-r20-r1/k{k}-f16.gguf") and x["direction"]==w[i]["direction"],"row identity")
        f=root/(i.replace(":","__")+".output.txt")
        need(str(f)==x["output_file"],"output path")
        data=f.read_bytes();s=data.decode("utf-8")
        need(sha(data)==x["output_sha256"] and s==x["raw_output"],"output digest")
        need("Assistant:\n" in s and s.split("Assistant:\n",1)[1].strip()==x["text"],"text extraction")
        need(x["exit_code"]==0,"nonzero exit")
        slim.append({key:x.get(key) for key in ("id","direction","text","output_sha256","wall_seconds","peak_vram_mib","maximum_rss_kib","prompt_eval","decode_eval","server_timings","stop_type","predicted_tokens","finish","tokens","command")})
    out["runs"][str(k)]={"rows":slim,"raw_prefix_sha256":sha(b"".join(lines)),"partial_line":partial,"enriched_exists":(root/"runs.enriched.jsonl").exists(),"artifact_bytes":(p.parent/f"gguf-r20-r1/k{k}-f16.gguf").stat().st_size}
for n in ("diagnosis_summary.json","k128-f16-verification.json","k96-f16-verification.json","k64-f16-verification.json","k96-q4-verification.json","f16-hybrid-vs-q4-gpu.json"):
    b=(p/n).read_bytes();out["stage_receipts"][n]={"sha256":sha(b),"receipt":json.loads(b)}
print(json.dumps(out,ensure_ascii=False,separators=(",",":")))

REMOTE
BASE='https://raw.githubusercontent.com/kaicreator-mm/mt-pruning-platform/3212303e1e02994a5124db8c85195b91325c7a7c'
curl -fsSL --retry 2 --retry-delay 1 --max-time 45 "$BASE/evidence/d01b-r19/corpus/corpus.jsonl" -o "$D/corpus.jsonl"
curl -fsSL --retry 2 --retry-delay 1 --max-time 45 "$BASE/mtpruning/evaluate.py" -o "$D/evaluate.py"
timeout 50 python3 -m pip install --disable-pip-version-check --no-cache-dir --target "$D/deps" 'sacrebleu==2.6.0' 'numpy==2.5.5' > "$D/install.log" 2>&1 || { tail -20 "$D/install.log"; exit 1; }
cat > "$D/replay.py" <<'PY'
import ast,json,hashlib,pathlib,sys,re,unicodedata,statistics,math
import sacrebleu,numpy as np
from sacrebleu.metrics import CHRF
p=pathlib.Path(sys.argv[1]);snap=json.loads((p/"snapshot.json").read_bytes())
def need(x,msg):
    if not x:raise ValueError(msg)
def blob(n,v):
    b=(p/n).read_bytes();need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==v,n);return b
ctxbytes=blob("corpus.jsonl","5fc8e3586a486e89ca6e0e6644d6c79ad1e4de72")
need(hashlib.sha256(ctxbytes).hexdigest()=="9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4","corpus sha")
ctx={x["id"]:x for x in map(json.loads,ctxbytes.splitlines()) if x["split"]=="validation"}
need(len(ctx)==120 and {x["id"] for x in snap["witnesses"]}==set(ctx),"validation IDs")
need(all(x["prompt_ids"]==ctx[x["id"]]["prompt_ids"] for x in snap["witnesses"]),"manifest token equality")
src=blob("evaluate.py","ac122a7f46f0381cba5a2b019c42e391c670d968")
tree=ast.parse(src);names={"digit_runs","chrf_pp","segment_stats","score_from_stats","paired_bootstrap"}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_DIGITS" for t in n.targets)]
need(len(nodes)==6,"definitions");env={"re":re,"np":np,"CHRF":CHRF};exec(compile(ast.Module(body=nodes,type_ignores=[]),"pinned-metric","exec"),env)
need(sacrebleu.__version__=="2.6.0" and np.__version__=="2.5.5","versions")
metric=env["chrf_pp"]();runs={k:{x["id"]:x for x in a["rows"]} for k,a in snap["runs"].items()}
dirs=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
opts={"--ctx-size":"2048","--batch-size":"512","--threads":"16","--cache-type-k":"f16","--cache-type-v":"f16","--temp":"0","--top-k":"1","--top-p":"1","--seed":"0","--n-predict":"640","--n-gpu-layers":"99","--n-cpu-moe":"40"}
for k,rs in runs.items():
    for x in rs.values():
        cmd=x["command"];need(all(cmd.count(o)==1 and cmd[cmd.index(o)+1]==v for o,v in opts.items()),"command options")
def score(rs,ids):
    h=[rs[i]["text"] for i in ids];ref=[ctx[i]["reference"] for i in ids]
    return metric.corpus_score(h,[ref]).score
results={}
for k,rs in runs.items():
    results[k]={"n":len(rs),"wall_seconds":sum(x["wall_seconds"] for x in rs.values()),"max_sampled_device_vram_mib":max(x["peak_vram_mib"] for x in rs.values()),"rows_without_finish":sum(x["finish"] is None for x in rs.values()),"rows_without_token_ids":sum(x["tokens"] is None for x in rs.values()),"rows_without_stop_type":sum(x["stop_type"] is None for x in rs.values()),"directions":{}}
    for d in dirs:
        ids=sorted(i for i in rs if i in runs["128"] and ctx[i]["direction"]==d)
        if not ids:continue
        base=score(runs["128"],ids);v=score(rs,ids)
        digits=lambda rr:sum(env["digit_runs"](rr[i]["text"])!=env["digit_runs"](ctx[i]["reference"]) for i in ids)
        results[k]["directions"][d]={"n":len(ids),"baseline_chrfpp":base,"chrfpp":v,"delta":v-base,"baseline_digit_flags":digits(runs["128"]),"digit_flags":digits(rs),"empty":sum(not rs[i]["text"].strip() for i in ids),"wall_seconds":sum(rs[i]["wall_seconds"] for i in ids)}
out={"status":"READ_ONLY_INTERIM_REPLAY_NOT_QUALIFICATION","captured":snap["captured"],"manifest_sha256":snap["manifest_sha256"],"metric_signature":str(metric.get_signature()),"snapshot_sha256":hashlib.sha256((p/"snapshot.json").read_bytes()).hexdigest(),"runs":{k:{key:v for key,v in a.items() if key!="rows"} for k,a in snap["runs"].items()},"results":results,"stage_receipts":snap["stage_receipts"]}
print(json.dumps(out,separators=(",",":"),allow_nan=False))

PY
python3 -m py_compile "$D/replay.py"
PYTHONPATH="$D/deps" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 15 timeout 30 python3 "$D/replay.py" "$D"
