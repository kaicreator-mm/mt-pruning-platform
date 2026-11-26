"""Read-only scorer for the new native greedy protocol.

Verify each raw request, response and server-log slice; pilot is not a gate.
No model launch or final-test evaluation.
"""
import argparse,ast,collections,hashlib,importlib.util,json,pathlib,re,unicodedata
import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF
P=pathlib.Path
ROOT=P.home()/".cache/mt-pruning"
PROTOCOL_BLOB="85b3bbabeafa4b6bf318a2f8fb25e44900d5626f"
DRIVER_BLOB="0cdca8d43df86722775aad3d391001c78872926f"
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def load(path,pin,name):
    b=path.read_bytes();need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==pin,name+" blob")
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def evaluator():
    b=(P.home()/"mt-pruning-platform/mtpruning/evaluate.py").read_bytes()
    need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()=="ac122a7f46f0381cba5a2b019c42e391c670d968","metric blob")
    names={"digit_runs","script_share","failure_flags","chrf_pp","segment_stats","score_from_stats","paired_bootstrap"}
    nodes=[n for n in ast.parse(b).body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_DIGITS" for t in n.targets)]
    need(len(nodes)==8 and sacrebleu.__version__=="2.6.0" and np.__version__=="2.5.5","metric versions")
    e={"re":re,"unicodedata":unicodedata,"np":np,"CHRF":CHRF};exec(compile(ast.Module(body=nodes,type_ignores=[]),"pinned-metric","exec"),e);return e
def verify_row(proto,root,r,x,index,model_digest,model_names):
    need(r["id"]==x["id"] and r["direction"]==x["direction"] and r["model_sha256"]==model_digest,"row identity")
    need(r["request_directory"]==f"request-{index:03d}","capture path")
    d=root/r["request_directory"];need(d.is_dir() and not d.is_symlink(),"capture directory")
    blobs={}
    for name,field in (("request.json","request_sha256"),("response.json","response_sha256"),("server.log","server_log_sha256")):
        p=d/name;need(p.is_file() and not p.is_symlink(),"raw regular file")
        b=p.read_bytes();need(sha(b)==r[field],"raw SHA "+name);blobs[name]=b
    q=json.loads(blobs["request.json"]);resp=json.loads(blobs["response.json"])
    need(q==proto.body(x["prompt_ids"]),"exact native request")
    need(resp["model"] in model_names,"response model")
    z=proto.validate(q,resp,blobs["server.log"].decode("utf-8",errors="replace"))
    need(all(r[k]==z[k] for k in ("text","tokens","raw_tokens","finish","timings","task","slot")),"journal/raw mismatch")
    return z
def read_run(proto,driver,root,full,smoke):
    p=(root/"run-plan.json").read_bytes();plan=json.loads(p)
    c=json.loads((root/"complete.json").read_bytes());raw=(root/"runs.jsonl").read_bytes()
    need(plan["status"]=="RESOURCE_PREFLIGHT_ONLY" and plan["blockers"]==[] and plan["model_full_hash_rechecked"] is True and plan["actual_model_sha256"]==plan["expected_model_sha256"],"execution admission")
    need(plan["runtime_file_hashes"]==driver.BINS and str(root)==plan["destination"],"run identity")
    phase=plan["phase"];need(phase in ("pilot","full"),"phase")
    wanted=full if phase=="full" else smoke+[x for x in smoke if x["direction"] in ("zh-my","my-zh")]
    need(plan["manifest_sha256"]==(driver.FULL_MANIFEST_SHA if phase=="full" else driver.WITNESS_SHA),"manifest pin")
    need(plan["sample_ids"]==[x["id"] for x in wanted],"frozen run order")
    need(c["status"]=="NATIVE_GREEDY_RECORDS_COMPLETE_NOT_QUALITY" and c["journal_sha256"]==sha(raw) and c["requests"]==len(wanted) and c["unique_ids"]==len({x["id"] for x in wanted}),"complete receipt")
    need(raw.endswith(b"\n"),"journal completeness")
    rows=[json.loads(l) for l in raw.splitlines()];need(len(rows)==len(wanted),"record coverage")
    props=json.loads((root/"props.json").read_bytes());need(props["model_path"]==plan["model"],"server model")
    cmd=plan["server_command"]
    need(cmd[0]==str(driver.R20/"build/bin/llama-server"),"server binary path")
    for opt,val in {"--model":plan["model"],"--n-gpu-layers":"99","--n-cpu-moe":"40","--ctx-size":"2048","--batch-size":"512","--threads":"16","--parallel":"1","--cache-type-k":"f16","--cache-type-v":"f16","--host":"127.0.0.1","--temp":"0","--top-k":"1","--top-p":"1","--seed":"0","--n-predict":"640"}.items():need(cmd.count(opt)==1 and cmd[cmd.index(opt)+1]==val,"server command "+opt)
    seen={};tasks=[]
    for i,(r,x) in enumerate(zip(rows,wanted)):
        v=verify_row(proto,root,r,x,i,plan["expected_model_sha256"],(plan["model"],props.get("model_alias")))
        need(r["repeat"]==(r["id"] in seen),"repeat marker")
        if r["repeat"]:need((r["tokens"],r["text"])==(seen[r["id"]]["tokens"],seen[r["id"]]["text"]),"repeat changed")
        else:seen[r["id"]]=r
        tasks.append(v["task"])
    need(all(a<b for a,b in zip(tasks,tasks[1:])),"task sequence")
    return plan,seen,{"journal_sha256":sha(raw),"run_plan_sha256":sha(p),"requests":len(rows),"unique":len(seen)}
def compare(e,ctx,a,b,phase):
    m=e["chrf_pp"]();result={}
    need(set(a)==set(b) and set(a)<=set(ctx),"matched validation IDs")
    for d in ("zh-my","my-zh","zh-en","en-zh","en-my","my-en"):
        ids=sorted(i for i in a if ctx[i]["direction"]==d)
        need(len(ids)==(20 if phase=="full" else 1),"direction sample count")
        refs=[ctx[i]["reference"] for i in ids]
        stats=[e["segment_stats"](m,[rs[i]["text"] for i in ids],refs) for rs in (a,b)]
        scores=[e["score_from_stats"](m,s) for s in stats]
        flags=[]
        for rs in (a,b):
            f=collections.Counter()
            for i in ids:
                x=ctx[i];r=rs[i];f.update(e["failure_flags"](x["source"],x["reference"],r["text"],x["tgt_lang"],r["finish"],r["tokens"]))
            flags.append(dict(f))
        delta=scores[1]-scores[0]
        result[d]={"n":len(ids),"baseline":scores[0],"candidate":scores[1],"delta":delta,"flags_baseline":flags[0],"flags_candidate":flags[1],"bootstrap":e["paired_bootstrap"](m,*stats) if phase=="full" else None,"quality_gate":(delta>=-1 and all(flags[1].get(k,0)<=flags[0].get(k,0) for k in ("empty","truncated","digits_differ_from_reference"))) if phase=="full" else None}
    return {"status":"FULL_DESCRIPTIVE_QUALITY_PASS_NOT_FINAL_QUALIFICATION" if phase=="full" and all(x["quality_gate"] for x in result.values()) else "FULL_QUALITY_GATE_FAIL" if phase=="full" else "PILOT_ONLY","directions":result,"metric_signature":str(m.get_signature())}
def selftest(proto,e):
    import tempfile,copy
    # Synthetic row exercises raw-byte and actual effective-settings rejection.
    q=proto.body([120000,123,456]);ts=[42,120025]
    settings={k:v for k,v in q.items() if k not in ("prompt","cache_prompt","return_tokens")}
    resp={"stop":True,"truncated":False,"generation_settings":settings,"tokens":ts,"tokens_predicted":2,"tokens_evaluated":3,"timings":{"predicted_n":2,"prompt_n":3,"cache_n":0,"prompt_ms":1.0,"predicted_ms":2.0},"stop_type":"eos","content":"测试","stopping_word":"","model":"synthetic"}
    log="sampler params: \nrepeat\n dry\n top_k = 1, top_p = 1.000, min_p = 0.000, temp = 0.000\n"+"".join(f"slot process_toke: id 0 | task 7 | n_gen = {j}, next token: {t}\n" for j,t in enumerate(ts,1))+"slot release: id 0 | task 7 | stop processing: n_tokens = 4, truncated = 0\n"
    with tempfile.TemporaryDirectory() as d:
        p=P(d);cap=p/"request-000";cap.mkdir()
        data={"request.json":json.dumps(q).encode(),"response.json":json.dumps(resp).encode(),"server.log":log.encode()}
        for n,b in data.items():(cap/n).write_bytes(b)
        x={"id":"validation:my-zh:synthetic","direction":"my-zh","prompt_ids":q["prompt"]}
        z=proto.validate(q,resp,log)
        r={**x,**z,"model_sha256":"1"*64,"request_directory":cap.name,"request_sha256":sha(data["request.json"]),"response_sha256":sha(data["response.json"]),"server_log_sha256":sha(data["server.log"])}
        need(verify_row(proto,p,r,x,0,"1"*64,("synthetic",))["tokens"]==[42],"normalized EOS")
        def rejects(fn):
            try:fn()
            except (ValueError,KeyError):return
            raise AssertionError("expected rejection")
        rejects(lambda:verify_row(proto,p,dict(r,tokens=[43]),x,0,"1"*64,("synthetic",)))
        rejects(lambda:verify_row(proto,p,r,dict(x,prompt_ids=[1]),0,"1"*64,("synthetic",)))
        rejects(lambda:verify_row(proto,p,r,x,0,"2"*64,("synthetic",)))
        bad=copy.deepcopy(resp);bad["generation_settings"]["temperature"]=.8;b=json.dumps(bad).encode();(cap/"response.json").write_bytes(b)
        rejects(lambda:verify_row(proto,p,dict(r,response_sha256=sha(b)),x,0,"1"*64,("synthetic",)))
    ctx={};a={};b={}
    for d,txt,lang in (("zh-my","မြန်မာ","my"),("my-zh","中文","zh"),("zh-en","hello world","en"),("en-zh","中文","zh"),("en-my","မြန်မာ","my"),("my-en","hello world","en")):
        i="validation:"+d+":synthetic";ctx[i]={"id":i,"direction":d,"source":"synthetic source","reference":txt,"tgt_lang":lang}
        a[i]={"text":txt,"finish":"eos","tokens":[42]};b[i]=dict(a[i])
    got=compare(e,ctx,a,b,"pilot");need(got["status"]=="PILOT_ONLY" and all(x["delta"]==0 and x["quality_gate"] is None for x in got["directions"].values()),"pilot never qualifies")
    print(json.dumps({"status":"PASS_SYNTHETIC_REPLAY_AND_PINNED_SCORER","raw_verification":1,"negative_checks":4,"pilot_group_scores":6,"metric_signature":got["metric_signature"]}))
def main():
    a=argparse.ArgumentParser();a.add_argument("--protocol",type=P,required=True);a.add_argument("--driver",type=P,required=True);a.add_argument("--selftest",action="store_true");a.add_argument("--compare",nargs=2,type=P)
    q=a.parse_args();proto=load(q.protocol,PROTOCOL_BLOB,"proto");driver=load(q.driver,DRIVER_BLOB,"driver");e=evaluator()
    if q.selftest:selftest(proto,e)
    if q.compare:
        full,smoke=driver.inputs();raw=(ROOT/"run-d01b-r19/corpus/corpus.jsonl").read_bytes();need(sha(raw)==driver.CORPUS_SHA,"corpus")
        ctx={x["id"]:x for x in map(json.loads,raw.splitlines()) if x["split"]=="validation"}
        p,a,ra=read_run(proto,driver,q.compare[0],full,smoke);q2,b,rb=read_run(proto,driver,q.compare[1],full,smoke)
        need(p["name"]=="128" and q2["name"]!="128" and p["phase"]==q2["phase"],"baseline/candidate")
        got=compare(e,ctx,a,b,p["phase"]);got.update(baseline=ra,candidate=rb);print(json.dumps(got,allow_nan=False,separators=(",",":")))
    need(q.selftest or q.compare,"explicit comparison")
if __name__=="__main__":main()

