"""R21 read-only validation of R20 v2 output and raw server token evidence.
No generation or writes. Verify completed journal prefix while writer may continue.
"""
import argparse,collections,hashlib,json,re
from pathlib import Path

MANIFEST_SHA="b4fd031ae37e5fff2ac50d7cd7b098e5c2a7c4a11f23a7420c983a3eb80098d9"
TOK=re.compile(r"slot\s+process_toke:\s+id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|\s+n_gen\s*=\s*(\d+),.*?next token:\s*(\d+)")
END=re.compile(r"slot\s+release:\s+id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|\s+stop processing: n_tokens\s*=\s*(\d+), truncated\s*=\s*(\d+)")
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def raw_tokens(log):
    pending={};blocks=[];finished=set()
    for line in log.splitlines():
        m=TOK.search(line)
        if m:
            slot,task,index,token=map(int,m.groups());key=(slot,task)
            need(key not in finished,"token after release")
            need(not pending or key in pending,"overlapping generation streams")
            arr=pending.setdefault(key,[])
            need(index==len(arr)+1,"nonsequential generation index")
            arr.append(token)
        elif "next token:" in line and "process_toke" in line:
            raise ValueError("unparsed token event")
        m=END.search(line)
        if m:
            slot,task,context,truncated=map(int,m.groups());key=(slot,task)
            need(key not in finished and key in pending,"unmatched release")
            blocks.append({"slot":slot,"task":task,"tokens":pending.pop(key),"context":context,"truncated":truncated})
            finished.add(key)
    need(not pending,"unreleased token events")
    return blocks
def check(root,config,keep):
    conf=json.loads(config.read_bytes());manifest=Path(conf["witness_manifest"]).read_bytes()
    need(sha(manifest)==MANIFEST_SHA,"frozen manifest")
    witnesses=json.loads(manifest)["witnesses"];need(len(witnesses)==120,"manifest count")
    expect=[x["id"] for x in witnesses];need(len(set(expect))==120,"manifest uniqueness")
    raw=(root/"runs.jsonl").read_bytes();lines=raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):lines=lines[:-1]
    raw=b"".join(lines)
    records=[json.loads(l) for l in raw.splitlines() if l.strip()]
    need(0<len(records)<=120 and [x["id"] for x in records]==expect[:len(records)],"record prefix order/coverage")
    complete=(root/"driver.exit").exists() and (root/"driver.exit").read_text().strip()=="EXIT_CODE=0" and len(records)==120
    log=Path(conf["server_log"]).read_bytes();endings=list(re.finditer(END.pattern.encode(),log))
    need(len(endings)>=len(records),"release prefix coverage")
    log=log[:endings[len(records)-1].end()];blocks=raw_tokens(log.decode("utf-8",errors="replace"))
    need(len(blocks)==len(records),"request block coverage")
    # The sequential runner uses sequential requests (slots may change); preserve task IDs
    # and fail closed if concurrent or incomplete token streams are observed.
    need(bool(blocks),"no request blocks")
    need(all(a["task"]<b["task"] for a,b in zip(blocks,blocks[1:])),"task order")
    receipt=json.loads((config.parent/f"k{keep}-f16-verification.json").read_bytes())
    eos=receipt["fields"]["eos_token_id"];vocab=receipt["fields"]["tokens"]["count"]
    need(type(eos)==int and 0<=eos<vocab,"EOS metadata")
    totals=collections.Counter();directions=collections.Counter();notes=[];pairs=[]
    for r,w,b in zip(records,witnesses,blocks):
        i=r["id"];need(r["direction"]==w["direction"] and r["config"]==conf["name"] and r["model"]==conf["model"] and r["exit_code"]==0,"row identity "+i)
        prompt=Path(w["prompt_file"]).read_bytes();need(sha(prompt)==w["prompt_sha256"],"prompt bytes "+i)
        out=Path(r["output_file"]);need(out==root/(i.replace(":","__")+".output.txt"),"output path "+i)
        data=out.read_bytes();text=data.decode("utf-8")
        need(sha(data)==r["output_sha256"] and text==r["raw_output"] and "Assistant:\n" in text and text.split("Assistant:\n",1)[1].strip()==r["text"],"output content "+i)
        cmd=r["command"];need("--server-base" in cmd and cmd[cmd.index("--server-base")+1]==conf["server_base"] and cmd[cmd.index("--file")+1]==w["prompt_file"],"execution mode "+i)
        for opt,val in {"--ctx-size":"2048","--batch-size":"512","--threads":"16","--cache-type-k":"f16","--cache-type-v":"f16","--temp":"0","--top-k":"1","--top-p":"1","--seed":"0","--n-predict":"640","--n-gpu-layers":"99","--n-cpu-moe":"40"}.items():
            need(cmd.count(opt)==1 and cmd[cmd.index(opt)+1]==val,"command "+i+" "+opt)
        ts=r["tokens"];timing=r["server_timings"]
        need(isinstance(ts,list) and ts==b["tokens"] and all(type(t)==int and 0<=t<vocab for t in ts),"actual token log "+i)
        need(bool(ts) and len(ts)==r["predicted_tokens"]==timing["predicted_n"],"token count "+i)
        need(r["finish"] in ("eos","length") and r["stop_type"] in ("eos","limit"),"unknown finish "+i)
        if r["finish"]=="eos":
            need(r["stop_type"]=="eos" and ts[-1]==eos and eos not in ts[:-1],"EOS witness "+i)
            # R19 reference generator excludes terminal EOS from returned tokens.
            # Report normalized count; do not overwrite raw record.
            normalized=ts[:-1]
        else:
            need(r["stop_type"]=="limit" and eos not in ts,"length witness "+i)
            normalized=ts
            if len(ts)!=640:notes.append({"id":i,"issue":"length_count_not_640","count":len(ts)})
        need(isinstance(r["processed_prompt"],str) and bool(r["processed_prompt"]),"processed prompt "+i)
        totals["raw_tokens"]+=len(ts);totals["normalized_tokens"]+=len(normalized);totals["finish_"+r["finish"]]+=1;totals["server_truncated"]+=bool(b["truncated"]);totals["empty"]+=not bool(r["text"].strip());directions[r["direction"]]+=1
        pairs.append({"id":i,"task":b["task"],"token_sha256":sha(json.dumps(ts,separators=(",",":")).encode()),"finish":r["finish"],"server_truncated":b["truncated"]})
    need(all(v<=20 for v in directions.values()),"direction prefix coverage")
    return records,{"status":"PREFIX_RAW_OUTPUT_TOKEN_REPLAY_NOT_QUALIFICATION","driver_completed":complete,"keep":keep,"rows":len(records),"manifest_sha256":sha(manifest),"records_sha256":sha(raw),"server_log_snapshot_sha256":sha(log),"config_sha256":sha(config.read_bytes()),"raw_log_task_count":len(blocks),"slots":sorted({b["slot"] for b in blocks}),"directions":dict(directions),"totals":dict(totals),"notes":notes,"request_token_bindings":pairs,"limitations":["Independent raw log order/token replay only; sequential task order association, not a cryptographic request-ID binding.","Rendered prompt token equivalence and model full hash/quality/stock checks not completed.","Server truncated flag is distinct from generation length limit. EOS removed only for normalized count, raw tokens retained."]}

import ast,unicodedata
import numpy as np,sacrebleu
from sacrebleu.metrics import CHRF
p=Path.home()/".cache/mt-pruning/r20-r1"
src=(Path.home()/"mt-pruning-platform/mtpruning/evaluate.py").read_bytes()
need(hashlib.sha1(b"blob "+str(len(src)).encode()+b"\0"+src).hexdigest()=="ac122a7f46f0381cba5a2b019c42e391c670d968","metric blob")
names={"digit_runs","script_share","failure_flags","chrf_pp","segment_stats","score_from_stats","paired_bootstrap"}
nodes=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_DIGITS" for t in n.targets)]
need(len(nodes)==8 and sacrebleu.__version__=="2.6.0" and np.__version__=="2.5.5","metric versions")
env={"re":re,"unicodedata":unicodedata,"np":np,"CHRF":CHRF};exec(compile(ast.Module(body=nodes,type_ignores=[]),"pinned-metric","exec"),env)
metric=env["chrf_pp"]();raw=(p.parent/"run-d01b-r19/corpus/corpus.jsonl").read_bytes()
need(sha(raw)=="9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4","corpus")
ctx={x["id"]:x for x in map(json.loads,raw.splitlines()) if x["split"]=="validation"}
need(len(ctx)==120,"context inventory")
runs={};receipts={}
for k in (128,96):
    rows,receipt=check(p/f"runs/full-f16-k{k}-v2",p/f"full-k{k}-v2-config.json",k)
    if k==128:need(receipt["records_sha256"]=="2b977c4b18a697b24057b140f6c9ef3f7298ba0b959bdd1900822cd8e5d169db" and receipt["driver_completed"],"baseline identity")
    runs[k]={r["id"]:r for r in rows};receipts[str(k)]=receipt
manifest=json.loads((p/"validation120/manifest.json").read_bytes())
need(all(x["prompt_ids"]==ctx[x["id"]]["prompt_ids"] for x in manifest["witnesses"]),"manifest prompt IDs")
def flags(r):
    x=ctx[r["id"]];ts=r["tokens"][:-1] if r["finish"]=="eos" else r["tokens"]
    return env["failure_flags"](x["source"],x["reference"],r["text"],x["tgt_lang"],r["finish"],ts)
out={}
for d in ("zh-my","my-zh","zh-en","en-zh","en-my","my-en"):
    ids=sorted(i for i in runs[96] if ctx[i]["direction"]==d)
    if not ids:continue
    ref=[ctx[i]["reference"] for i in ids];s={k:env["segment_stats"](metric,[rs[i]["text"] for i in ids],ref) for k,rs in runs.items()}
    scores={str(k):env["score_from_stats"](metric,rows) for k,rows in s.items()}
    f={str(k):dict(collections.Counter(x for i in ids for x in flags(rs[i]))) for k,rs in runs.items()}
    delta=env["paired_bootstrap"](metric,s[128],s[96])
    out[d]={"n":len(ids),"scores":scores,"delta":delta,"flags":f,"same_text":sum(runs[128][i]["text"]==runs[96][i]["text"] for i in ids),"full_direction_observed":len(ids)==20,"descriptive_gate":len(ids)==20 and delta["delta"]>=-1 and all(f["96"].get(g,0)<=f["128"].get(g,0) for g in ("empty","truncated","digits_differ_from_reference"))}
print(json.dumps({"status":"MATCHED_V2_PREFIX_REPLAY_NOT_QUALIFICATION","metric_signature":str(metric.get_signature()),"receipts":receipts,"directions":out,"limits":["Only completed prefix; full direction score descriptive, not canonical qualification.","No validation fitting. No inference or final-test scoring.","EOS removed only for normalized diagnostics, raw evidence retained."]},separators=(",",":"),allow_nan=False))

