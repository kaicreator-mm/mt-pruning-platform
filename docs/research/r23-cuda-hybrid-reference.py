"""CPU exact oracle for two fixed Fit trajectories, not a quality qualification."""
import json,hashlib,sys,pathlib,importlib.util,time,torch
P=pathlib.Path;R=P.home()/".cache/mt-pruning";D=P(__file__).parent;S=R/"r23h-pareto-screen-v1"
def need(v,m):
 if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def pin(p,h,git=False):
 b=p.read_bytes();got=hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest() if git else sha(b)
 need(got==h,"pin "+str(p));return b
def load(n,h):
 p=S/(n+".py");pin(p,h,True);s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
c=load("collector","02daf65bb7fe0526b2a700d17013054f7b0eea46");drv=load("driver","0735238e15f13fd75e3f6939fc31aa834d8b5854");prop=load("prop","88468e8c2a5905628365da52e1575d5251516512")
c.verify_code();need(torch.__version__=="2.10.0+cpu" and not drv.preflight()["blockers"],"CPU oracle preflight")
ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True);need(ident["teacher"]==drv.TEACHER and ident["num_experts"]==128,"teacher")
g=json.loads(pin(R/"r23g-recovery-ssd-38ed4798/recovery-receipt.json","ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"))
pp=json.loads(pin(R/"r23g-recovery-ssd-38ed4798/prop-plan.json","cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde",True))
p=json.loads(pin(R/"r23-results/r23h-all8-one-swap-prefill-v1/plan.json","a80858f97ff201b3cf62f25c779ab0786bb6d0a60973204c6fc1f3dd1f8c700b"))
original=json.loads(pin(S/"original.json","d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed",True))["candidates"]["96"]["kept"]
candidate=next(x["kept"] for x in p["candidates"] if x["name"]=="l42_remove63_add120")
need(c.canonical(original)=="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7" and c.canonical(candidate)=="f5d93021de52c065e0dddede1045d6410e0cbf6f645649731d1580626e66a6dc","masks")
raw=pin(R/"run-d01b-r19/corpus/corpus.jsonl","9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4")
ctx={x["id"]:x for x in map(json.loads,raw.splitlines()) if x["split"]=="calibration"}
torch.set_num_threads(8);torch.set_num_interop_threads(1)
e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=8)
masks={"teacher":None,"original96":original,"l42":candidate};out=[]
for i in (0,2):
 r=pp["rows"][i];prompt=ctx[r["id"]]["prompt_ids"];prefix=r["teacher_prefix_tokens"];need(c.canonical(prompt)==r["prompt_ids_sha256"] and len(prefix)==8,"prompt")
 old=json.loads(pin(R/"r23-results/r23g-l38-propagation-v1"/f"{i:02}.json",g["receipt_sha256"][i]))
 cand=json.loads((R/"r23-results/r23h-all8-one-swap-primary-v1"/f"l42_remove63_add120-{i:02}.json").read_bytes())
 need(cand["id"]==r["id"] and cand["mask_sha256"]==c.canonical(candidate),"historical candidate")
 result={}
 for n,m in masks.items():
  start=time.monotonic();z=prop.observe(c,e,prompt,prefix,None if m is None else {int(l):v for l,v in m.items()});dt=time.monotonic()-start
  want=old["original96"]["teacher_hashes"] if n=="teacher" else old["original96"]["masked_hashes"] if n=="original96" else cand["candidate96"]["masked_hashes"]
  need(prop.loghash(z)==want,"exact prior logits "+n)
  result[n]=dict(logits=[q["selected"] for q in z["logits"]],hashes=prop.loghash(z),routes=[dict(layer=k[0],phase=k[1],step=k[2],ids=v) for k,v in sorted(z["routes"].items())],seconds=dt)
  del z
  print(json.dumps(dict(status="CPU_EXACT_REFERENCE_CAPTURE",id=r["id"],mask=n,seconds=dt)),flush=True)
 out.append(dict(id=r["id"],prompt=prompt,prefix=prefix,results=result))
with (D/"cpu-reference.pt").open("xb") as f:torch.save(dict(teacher=drv.TEACHER,masks=masks,rows=out),f)
print(json.dumps(dict(status="CPU_EXACT_REFERENCE_COMPLETE",sha256=sha((D/"cpu-reference.pt").read_bytes()),trajectories=6,steps=54)),flush=True)
