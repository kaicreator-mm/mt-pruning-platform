"""R21 read-only validation of R20 v2 output and raw server token evidence.
No generation, no score, no writes. Use only on completed sequential runs.
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
    raw=(root/"runs.jsonl").read_bytes();need(raw.endswith(b"\n"),"complete record stream")
    records=[json.loads(l) for l in raw.splitlines() if l.strip()]
    need((root/"driver.exit").read_text().strip()=="EXIT_CODE=0","completed driver")
    need(len(records)==120 and [x["id"] for x in records]==expect,"record order/coverage")
    log=Path(conf["server_log"]).read_bytes();blocks=raw_tokens(log.decode("utf-8",errors="replace"))
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
    need(set(directions)=={"zh-my","my-zh","zh-en","en-zh","en-my","my-en"} and set(directions.values())=={20},"direction coverage")
    return {"status":"RAW_OUTPUT_TOKEN_REPLAY_PASS_NOT_QUALIFICATION","keep":keep,"rows":len(records),"manifest_sha256":sha(manifest),"records_sha256":sha(raw),"server_log_snapshot_sha256":sha(log),"config_sha256":sha(config.read_bytes()),"raw_log_task_count":len(blocks),"slots":sorted({b["slot"] for b in blocks}),"directions":dict(directions),"totals":dict(totals),"notes":notes,"request_token_bindings":pairs,"limitations":["Independent raw log order/token replay only; sequential task order association, not a cryptographic request-ID binding.","Rendered prompt token equivalence and model full hash/quality/stock checks not completed.","Server truncated flag is distinct from generation length limit. EOS removed only for normalized count, raw tokens retained."]}
def main():
    a=argparse.ArgumentParser();a.add_argument("--root",type=Path,required=True);a.add_argument("--config",type=Path,required=True);a.add_argument("--keep",type=int,required=True);args=a.parse_args();print(json.dumps(check(args.root,args.config,args.keep),ensure_ascii=False,separators=(",",":")))
if __name__=="__main__":main()

