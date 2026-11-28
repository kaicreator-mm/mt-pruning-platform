import tarfile,hashlib,json,math,sys,pathlib
def need(x,m):
 if not x: raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
raw=pathlib.Path(sys.argv[1]).read_bytes()
need(sha(raw)=="0a7a80a240dc865b270de92b7dafc93e6bd9ebe8e6363f924d44a67fbc80b805","archive pin")
with tarfile.open(sys.argv[1]) as t:
 members=t.getmembers();need(len(members)==50,"inventory");F={}
 for m in members:
  need(not m.name.startswith("/") and ".." not in pathlib.PurePosixPath(m.name).parts and (m.isfile() or m.isdir()),"safe member")
  if m.isfile():
   need(m.name not in F and m.size<5_000_000,"unique bounded file");F[m.name]=t.extractfile(m).read()
def j(p):return json.loads(F[p])
R="r23h-all8-one-swap-primary-run-v1/"
O="r23-results/r23h-all8-one-swap-primary-v1/"
S="r23-results/r23h-all8-one-swap-prefill-v1/"
G="r23g-recovery-ssd-38ed4798/"
N="l42_remove63_add120"
need(F[R+"exit-code"].strip()==b"0" and not F[R+"run.stderr"],"terminal")
need(sha(F[R+"primary.py"])=="68fcafed0d00363e874e207c9572bc1a6a42b88c9356de14d1381f34ebb5e2c6","source")
a=j(O+"admission.json");ah=sha(F[O+"admission.json"])
need(ah=="2265d5104f83dae18cb49b7e0bb2a5e413e95c64b7b5e2ff0c31ab265744af09","admission")
for p,k in [(R+"replay.json","independent_screen_replay_sha256"),(G+"recovery-receipt.json","previous_recovery_sha256"),(S+"plan.json","screen_plan_sha256"),(S+"report.json","screen_report_sha256")]:
 need(sha(F[p])==a[k],k)
need(j(R+"replay.json")["selected"]==N and a["holdout"]=="FORBIDDEN","screen selection")
p=j(S+"plan.json");c=next(x for x in p["candidates"] if x["name"]==N)
need(sha(canon(c["kept"]))==c["mask_sha256"]==a["candidates"][N] and set(c["kept"])==set(map(str,range(1,48))) and all(len(v)==96 and v==sorted(set(v)) and all(type(i) is int and 0<=i<128 for i in v) for v in c["kept"].values()),"mask")
g=j(G+"recovery-receipt.json");expanded=j(G+"expanded.json");pp=j(G+"prop-plan.json")
rows=pp["rows"]+[r for r in expanded["fit"] if r["direction"]=="zh-my" and r["id"] not in {z["id"] for z in pp["rows"]}]
selected=[(i,r) for i,r in enumerate(rows) if r["direction"] in ("zh-my","my-zh")]
need(len(selected)==10 and a["indices"]==[i for i,r in selected] and a["ids"]==[r["id"] for i,r in selected] and not ({r["line"] for i,r in selected}&{r["line"] for r in expanded["holdout"]}),"scope")
ev=[json.loads(l) for l in F[R+"run.log"].splitlines() if l.startswith(b"{")]
cap=[e for e in ev if e.get("status")=="R23H_ALL8_ONE_SWAP_PRIMARY_CAPTURED"]
need(len(cap)==10,"capture count")
sc={j(S+f"{i:02}.json")["id"]:j(S+f"{i:02}.json") for i in range(16)}
rr=[]
for (i,r),e in zip(selected,cap):
 fn=O+f"{N}-{i:02}.json";x=j(fn)
 need(sha(F[fn])==e["receipt_sha256"][N] and e["index"]==x["index"]==i and e["id"]==x["id"]==r["id"] and x["direction"]==r["direction"] and x["line"]==r["line"] and x["consumed_prefix"]==r["teacher_prefix_tokens"] and x["teacher"]==a["teacher"] and x["teacher_restored"] is True and x["mask_sha256"]==c["mask_sha256"] and x["admission_sha256"]==ah,"capture identity")
 old="r23-results/r23g-l38-propagation-v1/"+f"{i:02}.json"
 need(sha(F[old])==g["receipt_sha256"][i] and x["original96"]==j(old)["original96"],"baseline")
 for name in ("original96","candidate96"):
  z=x[name];ls=z["layer"];want={(l,s,"prefill" if s==0 else "decode") for l in range(1,48) for s in range(9)}
  need(len(ls)==423 and {(v["layer"],v["step"],v["phase"]) for v in ls}==want,"layer coverage")
  need(all(all(math.isfinite(v[k]) and v[k]>=0 for k in ("hidden_input_nmse","moe_output_nmse","mean_overlap")) and v["mean_overlap"]<=8 and v["tokens"]==(2 if v["step"]==0 else 1) and type(v["exact_sets"]) is int and 0<=v["exact_sets"]<=v["tokens"] for v in ls),"layer values")
  for k in ("teacher_hashes","masked_hashes"):
   need(len(z[k])==9 and all(v["step"]==s and len(v["sha256"])==64 and type(v["next"]) is int and v["next"]>=0 for s,v in enumerate(z[k])),"hash coverage")
  need(len(z["logits"])==9 and all(v["step"]==s and all(math.isfinite(v[k]) and v[k]>=0 for k in ("nmse","teacher_to_masked_kl")) and v["teacher_argmax"]==z["teacher_hashes"][s]["next"] and v["masked_argmax"]==z["masked_hashes"][s]["next"] for s,v in enumerate(z["logits"])),"logits")
 need(x["original96"]["teacher_hashes"]==x["candidate96"]["teacher_hashes"],"teacher match")
 s=sc[r["id"]];m=s["metrics"][N];v=x["candidate96"]["logits"][0]
 need(x["candidate96"]["masked_hashes"][0]["sha256"]==m["logits_sha256"] and x["candidate96"]["teacher_hashes"][0]["sha256"]==s["teacher_sha256"] and abs(v["teacher_to_masked_kl"]-m["kl"])<1e-12 and abs(v["nmse"]-m["nmse"])<1e-12 and int(v["teacher_argmax"]!=v["masked_argmax"])==m["argmax_mismatch"],"screen replay")
 rr.append(x)
groups={}
for d in ("zh-my","my-zh"):
 for phase in ("prefill","decode"):
  k=d+"/"+phase;groups[k]={}
  for n in ("original96","candidate96"):
   z=[v for x in rr if x["direction"]==d for v in x[n]["logits"] if (v["step"]==0)==(phase=="prefill")]
   need(len(z)==(8 if d=="zh-my" else 2)*(1 if phase=="prefill" else 8),"coverage")
   groups[k][n]={"kl":math.fsum(v["teacher_to_masked_kl"] for v in z)/len(z),"nmse":math.fsum(v["nmse"] for v in z)/len(z),"argmax_mismatch":sum(v["teacher_argmax"]!=v["masked_argmax"] for v in z),"positions":len(z)}
viol=[g+"/"+k for g in groups for k in ("kl","nmse","argmax_mismatch") if groups[g]["candidate96"][k]>groups[g]["original96"][k]+1e-12]
report=j(O+"report.json");rh=sha(F[O+"report.json"]);done=[e for e in ev if e.get("status")==report["status"]]
need(len(done)==1 and done[0]["report_sha256"]==rh=="0760d92157809e5e88a25af5188066060764f21470d5f581986bbf26676d2495" and report["admission_sha256"]==ah and report["groups"]=={N:groups} and report["violations"]=={N:viol} and report["eligible_for_secondary_fit_only"]==[] and report["selected"] is None and viol==["zh-my/decode/nmse"],"report decision")
# Layer associations, descriptive only (not causal).
deltas=[]
for l in range(1,48):
 z=[v["moe_output_nmse"]-u["moe_output_nmse"] for x in rr if x["direction"]=="zh-my" for u,v in zip(x["original96"]["layer"],x["candidate96"]["layer"]) if u["layer"]==v["layer"]==l and u["phase"]==v["phase"]=="decode" and u["step"]==v["step"]]
 need(len(z)==64,"layer association coverage")
 deltas.append([l,math.fsum(z)/len(z)])
print(json.dumps({"status":"R23H_ONE_SWAP_PRIMARY_INDEPENDENT_REPLAY_PASS_REJECT_CANDIDATE","archive_sha256":sha(raw),"report_sha256":rh,"samples":10,"groups":groups,"violations":viol,"eligible_for_secondary_fit_only":[],"layer_decode_nmse_delta_top":sorted(deltas,key=lambda x:-x[1])[:12],"limits":"Fixed fit; layer association not causal; no Fresh or quality admission."},sort_keys=True,indent=2))
