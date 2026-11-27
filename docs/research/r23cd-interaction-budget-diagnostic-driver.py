"""R23C/D combined same-state diagnostic: four swap masks + nested budget curves.
Bounded fit-only capture; no candidate admission, no holdout read or physical build.
Adapt exactly two uniform-budget guards on pinned local observer sources.
"""
import argparse,hashlib,json,pathlib,math,os,time,fcntl,importlib.util,sys
P=pathlib.Path;ROOT=P.home()/".cache/mt-pruning"
PLAN_SHA="a9da78015d010484c16d6c8af9000a15d46946db00677c19f4fac4a0d562e41e"
PINS={"batch":"08abbbfbd930e532d41602dd29a5c70f2bc474e3","verifier":"e56b25b3fd92ccbd91455628606925a02af49450","driver":"0735238e15f13fd75e3f6939fc31aa834d8b5854","collector":"02daf65bb7fe0526b2a700d17013054f7b0eea46","pilot":"19c8c851b1647cc0f495d6a7f1122219c42a0cdb","sparse":"1c38f6e928ff653fbbef584b298ff65e194900fe","interaction":"0bc99fcec770371e076c625a9ec9a540aaf65ba3"}
NAMES=("original96","swap_a","swap_b","swap_ab","k64","k80","k112","k128")
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en");GROUPS=tuple(d+"/"+p for d in DIRS for p in ("prefill","decode"))
def need(x,m):
    if not x:raise ValueError(m)
def sha(x):return hashlib.sha256(x).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(raw):return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
def load(path,name,adapt=False):
    raw=path.read_bytes();need(blob(raw)==PINS[name],"source "+name)
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m
    if adapt:
        # Keep all individual mask/shape/top-k/finite checks and exhaustive oracle.
        a,b={"collector":('candidates and len({len(ids) for masks in candidates.values() for ids in masks.values()})==1','candidates and all(k<=len(ids)<=n for masks in candidates.values() for ids in masks.values())'),"sparse":('masks and len({len(v) for v in masks})==1','masks and all(top_k<=len(v)<=n for v in masks)')}[name]
        src=raw.decode();need(src.count(a)==1,"exact guard")
        exec(compile(src.replace(a,b),str(path),"exec"),m.__dict__)
    else:sp.loader.exec_module(m)
    return m
def report(rows,captures,interaction,masks,layers):
    need(set(captures)=={r["id"] for r in rows} and len(rows)==12,"fit scope")
    expected={n:sha(canon(ms)) for n,ms in masks.items()}
    data={l:{n:[[] for _ in GROUPS] for n in NAMES} for l in layers}
    for r in rows:
        x=captures[r["id"]];need(x["id"]==r["id"] and x["line"]==r["line"] and x["direction"]==r["direction"] and x["logit_equivalence"] is True and x["candidate_masks"]==expected,"sample identity")
        obs=x["observations"];seen={(o["layer"],o["phase"],o["step"]):o for o in obs}
        need(len(obs)==len(seen)==3*len(layers) and set(seen)=={(l,p,j) for l in layers for p,j in (("prefill",0),("decode",1),("decode",8))},"observations")
        for o in obs:
            need(set(o["losses"])==set(NAMES) and all(math.isfinite(o[k]) and 0<=o[k]<=1e-4 for k in ("routed_drift","combined_drift")) and o["row_count"]==(2 if o["phase"]=="prefill" else 1),"observer")
            need(all(set(v)=={"routed_error","combined_error"} and all(math.isfinite(z) and z>=0 for z in v.values()) for v in o["losses"].values()),"finite errors")
            need(o["losses"]["k128"]["combined_error"]==o["combined_drift"] and o["losses"]["k128"]["routed_error"]==o["routed_drift"],"full inventory oracle")
        for l in layers:
            for n in NAMES:
                for p,steps in (("prefill",[0]),("decode",[1,8])):
                    j=GROUPS.index(r["direction"]+"/"+p)
                    data[l][n][j].append(math.fsum(seen[l,p,t]["losses"][n]["combined_error"] for t in steps)/len(steps))
    need(all(len(z)==2 for cell in data.values() for a in cell.values() for z in a),"all groups")
    avg={str(l):{n:[math.fsum(z)/2 for z in a] for n,a in cell.items()} for l,cell in data.items()}
    effects={}
    for l,cell in avg.items():
        effects[l]={g:interaction.measure({"base":cell["original96"][i],**{n:cell[n][i] for n in ("swap_a","swap_b","swap_ab")}}) for i,g in enumerate(GROUPS)}
    means={n:dict(zip(GROUPS,[math.fsum(cell[n][i] for cell in avg.values())/len(avg) for i in range(12)])) for n in NAMES}
    return dict(status="FIT_ONLY_INTERACTION_BUDGET_DIAGNOSTIC_NOT_ADMISSION",groups=GROUPS,means=means,layer_group_combined_error=avg,interaction_by_layer_group=effects,scope="same-state original-centered curves; local proxy only; no mask choice or holdout evaluation")
def selftest(c,b,pilot,s,w):
    torch=c.torch;torch.set_num_threads(1)
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        p=P(t);m=c.tiny_checkpoint(p/"tiny",p/"none",vocab_size=256,seed=19);e=c.engine_module.Engine(c.engine_module.Checkpoint(m),threads=1)
        prompt=[4,8,15,16,23,42];prefix=c.greedy_prefix(e,prompt,8)
        r={"id":"calibration:zh-my:synthetic","direction":"zh-my","line":0,"split":"calibration","prompt_tokens":len(prompt),"prompt_ids_sha256":c.canonical(prompt),"teacher_prefix_tokens":prefix,"teacher_prefix_sha256":c.canonical(prefix),"prefill_rows":[2,5],"decode_steps_1based":[1,8]}
        masks={f"k{k}":{1:list(range(k)),2:list(range(8-k,8))} for k in (2,4,6,8)}
        exact=c.capture(w,e,r,prompt,masks);got=pilot.capture(c,w,s,e,r,prompt,masks)
        need(exact["observations"]==got["observations"] and exact["teacher_logits"]==got["teacher_logits"],"heterogeneous sparse exhaustive oracle")
        need(len(got["observations"])==6 and got["extra_oracle_evaluations"]==32 and got["logit_equivalence"] is True,"coverage")
        need(all(o["losses"]["k8"]["combined_error"]==o["combined_drift"] for o in got["observations"]),"full expert oracle")
        need("_moe" not in e.__dict__,"restoration")
        print(json.dumps(dict(status="PASS_SYNTHETIC_MIXED_BUDGET_EXACT_DIFFERENTIAL",budgets=[2,4,6,8],observations=6,logits=9,real_capture="NOT_RUN")),flush=True)
def main():
    a=argparse.ArgumentParser();a.add_argument("--inputs",type=P,required=True);a.add_argument("--admission",type=P);a.add_argument("--execute",action="store_true");a.add_argument("--selftest",action="store_true");q=a.parse_args()
    d=q.inputs;b,v,drv,c,pilot,s,interaction=[load(d/(n+".py"),n,n in ("collector","sparse")) for n in ("batch","verifier","driver","collector","pilot","sparse","interaction")]
    c.verify_code();w=c.load_witness(d/"witness.py")
    if q.selftest:selftest(c,b,pilot,s,w);return
    need(q.admission is not None,"admission path");raw=q.admission.read_bytes();ad=v.strict(raw)
    need(raw==canon(ad) and ad["status"]=="R23CD_FIT_DIAGNOSTIC_ADMITTED" and ad["plan_sha256"]==PLAN_SHA and ad["teacher"]==v.TEACHER and ad["engine_head"]==v.ENGINE and ad["fit_plan_blob"]==v.PLAN and ad["threads"]==8,"admission")
    planraw=(ROOT/"r23-results/fit-interaction-budget-plan-v1.json").read_bytes();need(sha(planraw)==PLAN_SHA,"mask plan")
    plan=v.strict(planraw);masks={n:x["kept"] for n,x in plan["masks"].items()}
    need(set(masks)==set(NAMES) and plan["supported_layers"]==47 and all(sha(canon(masks[n]))==x["sha256"] for n,x in plan["masks"].items()),"plan inventory")
    layers=list(range(1,48));need(all(set(m)==set(map(str,layers)) and all(len(ids)==(96 if n.startswith("swap") or n=="original96" else int(n[1:])) and ids==sorted(set(ids)) and all(type(i) is int and 0<=i<128 for i in ids) for ids in m.values()) for n,m in masks.items()),"budget shapes")
    corpus=ROOT/"run-d01b-r19/corpus/corpus.jsonl";baseline=ROOT/"run-d01b-r19/baseline/records.jsonl"
    c.pilot_inputs(d/"plan.json",corpus,baseline,d/"original.json",d/"guarded.json");rows=v.frozen(d/"plan.json")[0]["fit"]
    need(ad["fit_ids"]==[r["id"] for r in rows] and set(plan["fit_ids"])==set(ad["fit_ids"]),"fit ids")
    ctx={x["id"]:x for x in map(json.loads,corpus.read_bytes().splitlines()) if x["split"]=="calibration" and x["id"] in ad["fit_ids"]}
    pre=drv.preflight();print(json.dumps(dict(pre,fit_samples=12,mask_count=8)),flush=True)
    if not q.execute:return
    need(not pre["blockers"],"resources")
    dst=ROOT/"r23-results/interaction-budget-v1"
    with (ROOT/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);need(not drv.preflight()["blockers"] and not dst.exists(),"exclusive resources/path")
        dst.mkdir(mode=0o700);b.publish(dst/"admission.json",ad)
        torch=c.torch;torch.set_num_threads(8);torch.set_num_interop_threads(1)
        ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True);need(ident["teacher"]==v.TEACHER and ident["num_experts"]==128 and not drv.preflight()["blockers"],"teacher")
        e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=8)
        c.forward_sequence=lambda e,prompt,prefix,callback=None:b.forced_forward(c,e,prompt,prefix,callback)
        print(json.dumps(dict(status="TEACHER_READY",teacher=ident["teacher"],admission_sha256=sha(raw))),flush=True)
        ms={n:{int(l):ids for l,ids in m.items()} for n,m in masks.items()};results={}
        for i,r in enumerate(rows):
            need(not drv.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"GPU busy")
            t=time.monotonic();x=b.annotate(pilot.capture(c,w,s,e,dict(r,split="calibration"),ctx[r["id"]]["prompt_ids"],ms),r)
            need(x["extra_oracle_evaluations"]==512 and len(x["observations"])==141,"oracle")
            x.update(teacher_digest=ident["teacher"],engine_head=v.ENGINE,admission_sha256=sha(raw),index=i)
            results[r["id"]]=x;h=b.publish(dst/f"fit-{i:02d}.json",x)
            print(json.dumps(dict(status="CD_FIT_CAPTURED",id=r["id"],completed=i+1,receipt_sha256=h,seconds=time.monotonic()-t)),flush=True)
        out=report(rows,results,interaction,masks,layers);h=b.publish(dst/"report.json",out)
        print(json.dumps(dict(status=out["status"],receipt_sha256=h,means=out["means"])),flush=True)
if __name__=="__main__":main()
