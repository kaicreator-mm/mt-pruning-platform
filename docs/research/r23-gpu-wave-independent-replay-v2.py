import json,hashlib,tarfile,sys,math,pathlib
def need(x,m):
 if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
p=pathlib.Path(sys.argv[1]);need(sha(p.read_bytes())==sys.argv[2],"transport archive")
with tarfile.open(p) as t:
 F={}
 for m in t.getmembers():
  need(not m.name.startswith("/") and ".." not in pathlib.PurePosixPath(m.name).parts and (m.isfile() or m.isdir()),"safe")
  if m.isfile():
   need(m.name not in F and m.size<5_000_000,"inventory");F[m.name]=t.extractfile(m).read()
def j(n):return json.loads(F[n])
W=int(sys.argv[3]);need(W in (1,2),"wave");R="r23-pipeline-v2/";O=f"r23-results/r23h-gpu-proposed-wave{W}-primary-v2/";G="r23g-recovery-ssd-38ed4798/"
need(F[R+f"wave{W}.exit"].strip()==b"0" and not F[R+f"wave{W}.stderr"],"terminal")
need(sha(F[R+"r23-gpu-proposed-primary-v2.py"])=="25965fb82fbc4f9958b59b168958aa317c065107cec863f972e223984a5b50a0","source")
a=j(O+"admission.json");ah=sha(F[O+"admission.json"]);need(a["status"]=="R23H_GPU_PROPOSED_PRIMARY_BATCH_FIT_ONLY" and a["holdout"]=="FORBIDDEN" and a["teacher"]=="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94","admission")
plan=j(R+f"wave{W}-plan.json");need(sha(F[R+f"wave{W}-plan.json"])==a["search_plan_sha256"] and sha(F[R+"replay.json"])==a["independent_parent_replay_sha256"],"source")
base=next(x for x in j("r23-results/r23h-all8-one-swap-prefill-v1/plan.json")["candidates"] if x["name"]=="l42_remove63_add120")["kept"]
need(sha(canon(base))==plan["base_mask_sha256"],"base")
names={}
rep=j(R+"gpu-search-report.json")
need(sha(F[R+"gpu-search-report.json"])==plan["search_report_sha256"] and rep["evaluated"]==33792 and rep["status"]=="GPU_LOCAL_PROXY_SEARCH_COMPLETE_NOT_ADMISSION" and plan["wave"]==W,"GPU source")
chosen=[];seen=set()
for v in sorted(rep["rank"],key=lambda x:(x["score"],x["layer"],x["remove"],x["add"])):
 if v["layer"] not in seen:chosen.append(v);seen.add(v["layer"])
need(seen=={25,26,27,28,32,34,35,36,39,41,44},"layers")
exp=[("gpu_l%d_%d_%d"%(v["layer"],v["remove"],v["add"]),[v]) for v in chosen]+[("gpu_joint_best2",chosen[:2]),("gpu_joint_best3",chosen[:3])]
exp=exp[:6] if W==1 else exp[6:]
for n,items in exp:
 mask={k:v[:] for k,v in base.items()}
 for v in items:
  k=str(v["layer"]);rem=v["remove"];add=v["add"];need(rem in mask[k] and add not in mask[k],"swap");mask[k]=sorted((set(mask[k])-{rem})|{add})
 need(set(mask)==set(map(str,range(1,48))) and all(len(v)==96 and v==sorted(set(v)) and all(type(i) is int and 0<=i<128 for i in v) for v in mask.values()),"mask")
 names[n]=sha(canon(mask));got=next(x for x in plan["candidates"] if x["name"]==n)
 need(got["kept"]==mask and got["mask_sha256"]==names[n],"mask bytes")
need(names==a["candidates"]=={x["name"]:x["mask_sha256"] for x in plan["candidates"]},"masks")
g=j(G+"recovery-receipt.json");need(sha(F[G+"recovery-receipt.json"])==a["previous_recovery_sha256"],"recovery")
exp=j(G+"expanded.json");pp=j(G+"prop-plan.json");rows=pp["rows"]+[r for r in exp["fit"] if r["direction"]=="zh-my" and r["id"] not in {z["id"] for z in pp["rows"]}]
sel=[(i,r) for i,r in enumerate(rows) if r["direction"] in ("zh-my","my-zh")]
need(len(sel)==10 and a["ids"]==[r["id"] for i,r in sel] and a["indices"]==[i for i,r in sel] and not ({r["line"] for i,r in sel}&{r["line"] for r in exp["holdout"]}),"scope")
events=[json.loads(l) for l in F[R+f"wave{W}.log"].splitlines() if l.startswith(b"{")];caps=[x for x in events if x["status"]=="R23H_GPU_PROPOSED_PRIMARY_CAPTURED"]
need(len(caps)==10,"capture")
allrs={n:[] for n in names}
for (i,r),e in zip(sel,caps):
 old="r23-results/r23g-l38-propagation-v1/"+f"{i:02}.json";need(sha(F[old])==g["receipt_sha256"][i],"baseline pin")
 need(e["index"]==i and e["id"]==r["id"] and set(e["receipt_sha256"])==set(names),"event")
 for n,h in names.items():
  fn=O+f"{n}-{i:02}.json";x=j(fn)
  need(sha(F[fn])==e["receipt_sha256"][n] and x["id"]==r["id"] and x["index"]==i and x["direction"]==r["direction"] and x["line"]==r["line"] and x["consumed_prefix"]==r["teacher_prefix_tokens"] and x["teacher"]==a["teacher"] and x["teacher_restored"] is True and x["mask_sha256"]==h and x["admission_sha256"]==ah,"receipt")
  need(x["original96"]==j(old)["original96"] and x["original96"]["teacher_hashes"]==x["candidate96"]["teacher_hashes"],"teacher/baseline")
  for nm in ("original96","candidate96"):
   z=x[nm];ls=z["layer"];need(len(ls)==423 and {(v["layer"],v["step"],v["phase"]) for v in ls}=={(l,s,"prefill" if s==0 else "decode") for l in range(1,48) for s in range(9)},"layers")
   need(all(all(math.isfinite(v[k]) and v[k]>=0 for k in ("hidden_input_nmse","moe_output_nmse","mean_overlap")) and v["mean_overlap"]<=8 and v["tokens"]==(2 if v["step"]==0 else 1) and 0<=v["exact_sets"]<=v["tokens"] for v in ls),"layer values")
   for k in ("teacher_hashes","masked_hashes"):need(len(z[k])==9 and all(v["step"]==s and len(v["sha256"])==64 and type(v["next"]) is int for s,v in enumerate(z[k])),"token hashes")
   need(len(z["logits"])==9 and all(v["step"]==s and all(math.isfinite(v[k]) and v[k]>=0 for k in ("nmse","teacher_to_masked_kl")) and v["teacher_argmax"]==z["teacher_hashes"][s]["next"] and v["masked_argmax"]==z["masked_hashes"][s]["next"] for s,v in enumerate(z["logits"])),"logits")
  allrs[n].append(x)
groups={};viol={}
for n,rs in allrs.items():
 groups[n]={}
 for d in ("zh-my","my-zh"):
  for phase in ("prefill","decode"):
   k=d+"/"+phase;groups[n][k]={}
   for nm in ("original96","candidate96"):
    z=[v for x in rs if x["direction"]==d for v in x[nm]["logits"] if (v["step"]==0)==(phase=="prefill")];need(len(z)==(8 if d=="zh-my" else 2)*(1 if phase=="prefill" else 8),"coverage")
    groups[n][k][nm]={"kl":math.fsum(v["teacher_to_masked_kl"] for v in z)/len(z),"nmse":math.fsum(v["nmse"] for v in z)/len(z),"argmax_mismatch":sum(v["teacher_argmax"]!=v["masked_argmax"] for v in z),"positions":len(z)}
 viol[n]=[g+"/"+k for g in groups[n] for k in ("kl","nmse","argmax_mismatch") if groups[n][g]["candidate96"][k]>groups[n][g]["original96"][k]+1e-12]
q=j(O+"report.json");rh=sha(F[O+"report.json"]);eligible=sorted((n for n in groups if not viol[n]),key=lambda n:(math.fsum(groups[n][g]["candidate96"][k]/max(groups[n][g]["original96"][k],1e-12) for g in groups[n] for k in ("kl","nmse"))/8,n));need(q["groups"]==groups and q["violations"]==viol and q["primary_gate_pass"]==eligible and q["eligible_for_secondary_fit_only"]==[] and q["selected"] is None,"decision")
need(len([x for x in events if x.get("status")==q["status"] and x.get("report_sha256")==rh])==1,"terminal binding")
print(json.dumps(dict(status="R23H_GPU_PROPOSED_INDEPENDENT_REPLAY_PASS",archive_sha256=sha(p.read_bytes()),report_sha256=rh,candidates=names,groups=groups,violations=viol,captures=len(names)*10,wave=W,primary_gate_pass=eligible),sort_keys=True,indent=2))
