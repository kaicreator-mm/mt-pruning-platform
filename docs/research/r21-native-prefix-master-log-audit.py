"""Independent read-only native full/pilot prefix audit.
No model generation or validation-based mask fitting. No full weight rehash.
"""
import argparse,collections,json,pathlib,hashlib,math
P=pathlib.Path
ROOT=P.home()/".cache/mt-pruning"
def need(v,m):
    if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(p):
    b=p.read_bytes();return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def audit(score,proto,driver,k,phase,wanted):
    root=ROOT/"native-greedy-runs"/f"{k}-{phase}"
    if not (root/"runs.jsonl").exists():return None,{}
    planbytes=(root/"run-plan.json").read_bytes();plan=json.loads(planbytes)
    admission=(ROOT/"native-greedy-admissions"/f"{k}-{phase}.json").read_bytes()
    need(sha(admission)==plan["admission_sha256"],"admission bytes")
    a=json.loads(admission);driver.validate_admission(a,plan)
    need(a["handoff_comment"]==7211531504 and a["admission_comment"]==(7211579974 if phase=="pilot" else 7211680160),"admission provenance")
    model,size,digest=driver.model_identity(str(k))
    need(plan["name"]==str(k) and plan["phase"]==phase and plan["model"]==str(model) and plan["model_bytes"]==size and plan["actual_model_sha256"]==plan["expected_model_sha256"]==digest and plan["model_full_hash_rechecked"] is True,"model plan")
    need(plan["status"]=="RESOURCE_PREFLIGHT_ONLY" and not plan["blockers"] and plan["runtime_file_hashes"]==driver.BINS and plan["sample_ids"]==[x["id"] for x in wanted] and plan["destination"]==str(root),"plan identity")
    props=json.loads((root/"props.json").read_bytes());need(props["model_path"]==str(model),"model props")
    raw=(root/"runs.jsonl").read_bytes();lines=raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):lines=lines[:-1]
    raw=b"".join(lines);rows=[json.loads(l) for l in lines]
    need(len(rows)<=len(wanted),"record count")
    # Read after snapshotting journal; writer has already saved each row's log.
    master=(root/"server.log").read_bytes()
    cursor=0;seen={};matches=[];tasks=[];tot=collections.Counter()
    for i,(row,x) in enumerate(zip(rows,wanted)):
        v=score.verify_row(proto,root,row,x,i,digest,(str(model),props.get("model_alias")))
        chunk=(root/row["request_directory"]/"server.log").read_bytes()
        pos=master.find(chunk,cursor);need(pos>=cursor,"log capture not found in sequential master log "+row["id"])
        cursor=pos+len(chunk);tasks.append(v["task"])
        need(row["repeat"]==(row["id"] in seen),"repeat marker")
        if row["repeat"]:need((row["raw_tokens"],row["text"])==(seen[row["id"]]["raw_tokens"],seen[row["id"]]["text"]),"pilot repeat drift")
        else:seen[row["id"]]=row
        tot["rows"]+=1;tot["eos"]+=row["finish"]=="eos";tot["length"]+=row["finish"]=="length";tot["empty"]+=not bool(row["text"])
        tot["raw_tokens"]+=len(row["raw_tokens"]);tot["prompt_tokens"]+=row["timings"]["prompt_n"]
        tot["prefill_ms"]+=row["timings"]["prompt_ms"];tot["decode_ms"]+=row["timings"]["predicted_ms"];tot["wall_seconds"]+=row["wall_seconds"]
        matches.append({"id":row["id"],"task":v["task"],"master_log_offset":pos,"master_log_bytes":len(chunk),"log_sha256":sha(chunk),"response_sha256":row["response_sha256"]})
    need(all(a<b for a,b in zip(tasks,tasks[1:])),"task order")
    complete=False
    if (root/"complete.json").exists() and len(rows)==len(wanted):
        receipt=json.loads((root/"complete.json").read_bytes())
        need(receipt["status"]=="NATIVE_GREEDY_RECORDS_COMPLETE_NOT_QUALITY" and receipt["journal_sha256"]==sha(raw) and receipt["requests"]==len(rows) and receipt["unique_ids"]==len(seen),"complete receipt")
        complete=True
    return {"status":"PREFIX_VERIFIED_NOT_QUALIFICATION","complete":complete,"rows":len(rows),"planned_rows":len(wanted),"unique":len(seen),"journal_prefix_sha256":sha(raw),"admission_sha256":sha(admission),"run_plan_sha256":sha(planbytes),"master_log_snapshot_sha256":sha(master),"matched_master_log_slices":len(matches),"last_verified_master_byte":cursor,"totals":dict(tot),"bindings":matches},seen
def bridge(e,ctx,bf,run):
    metric=e["chrf_pp"]();out={}
    for d in ("zh-my","my-zh","zh-en","en-zh","en-my","my-en"):
        ids=sorted(i for i in run if ctx[i]["direction"]==d)
        if not ids:continue
        ref=[ctx[i]["reference"] for i in ids]
        stats=[e["segment_stats"](metric,[r[i]["text"] for i in ids],ref) for r in (bf,run)]
        flags=[]
        for records in (bf,run):
            flag=collections.Counter()
            for i in ids:
                x=ctx[i];r=records[i];flag.update(e["failure_flags"](x["source"],x["reference"],r["text"],x["tgt_lang"],r["finish"],r["tokens"]))
            flags.append(dict(flag))
        out[d]={"n":len(ids),"complete_direction":len(ids)==20,"bf16":e["score_from_stats"](metric,stats[0]),"native":e["score_from_stats"](metric,stats[1]),"exact_text":sum(bf[i]["text"]==run[i]["text"] for i in ids),"exact_tokens":sum(bf[i]["tokens"]==run[i]["tokens"] for i in ids),"flags_bf16":flags[0],"flags_native":flags[1]}
    return out
def main():
    a=argparse.ArgumentParser();a.add_argument("--dir",type=P,required=True);q=a.parse_args();p=q.dir
    need(blob(p/"score.py")=="ee010f8e92e649dfca9b930c8d19c414d99e26e2","score pin")
    import importlib.util
    sp=importlib.util.spec_from_file_location("score",p/"score.py");s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
    proto=s.load(p/"protocol.py",s.PROTOCOL_BLOB,"proto");driver=s.load(p/"driver.py",s.DRIVER_BLOB,"driver");e=s.evaluator()
    full,smoke=driver.inputs();pilot=smoke+[x for x in smoke if x["direction"] in ("zh-my","my-zh")]
    ctx={x["id"]:x for x in map(json.loads,(ROOT/"run-d01b-r19/corpus/corpus.jsonl").read_bytes().splitlines()) if x["split"]=="validation"}
    raw=(ROOT/"run-d01b-r19/baseline/records.jsonl").read_bytes();need(sha(raw)=="37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8","BF16 records")
    _,*rows=map(json.loads,raw.splitlines());bf={x["id"]:x for x in rows if x["split"]=="validation"};need(set(bf)==set(ctx),"BF16 identity")
    receipts={};repro={};bridges={};runs={}
    for k in (128,96):
        pr,ps=audit(s,proto,driver,k,"pilot",pilot);need(pr and pr["complete"],"pilot complete")
        fr,fs=audit(s,proto,driver,k,"full",full)
        receipts[str(k)]={"pilot":pr,"full":fr};runs[k]=fs
        common=sorted(set(ps)&set(fs));repro[str(k)]={"matched":len(common),"same_raw_tokens":sum(ps[i]["raw_tokens"]==fs[i]["raw_tokens"] for i in common),"same_text":sum(ps[i]["text"]==fs[i]["text"] for i in common),"differences":[i for i in common if (ps[i]["raw_tokens"],ps[i]["text"])!=(fs[i]["raw_tokens"],fs[i]["text"])]}
        if fs:bridges[str(k)]=bridge(e,ctx,bf,fs)
    # Only score complete matched groups; partial group stays explicitly marked.
    overlap=set(runs[128])&set(runs[96]);paired={}
    if overlap:
        metric=e["chrf_pp"]()
        for d in ("zh-my","my-zh","zh-en","en-zh","en-my","my-en"):
            ids=sorted(i for i in overlap if ctx[i]["direction"]==d)
            if ids:
                refs=[ctx[i]["reference"] for i in ids]
                scores=[metric.corpus_score([runs[k][i]["text"] for i in ids],[refs]).score for k in (128,96)]
                paired[d]={"n":len(ids),"complete_direction":len(ids)==20,"128":scores[0],"96":scores[1],"delta":scores[1]-scores[0]}
    print(json.dumps({"status":"INDEPENDENT_PREFIX_AUDIT_NOT_QUALIFICATION","receipts":receipts,"pilot_full_reproducibility":repro,"bf16_bridge_descriptive":bridges,"matched_native_prefix_descriptive":paired,"limitations":["Full model hashes were checked by execution driver, not re-read here.","BF16 reference differs in runtime and precision; bridge is descriptive, not canonical gate.","Active journal prefix only; no test split scoring or mask fitting."]},separators=(",",":"),allow_nan=False))
if __name__=="__main__":main()

