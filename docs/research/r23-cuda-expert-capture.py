import pathlib,json,hashlib,sys,time,importlib.util,torch
P=pathlib.Path;R=P.home()/".cache/mt-pruning";D=P(__file__).parent;S=R/"r23h-pareto-screen-v1"
def need(x,m):
 if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p,h,n):
 b=p.read_bytes();need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==h,"source "+n)
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
c=load(S/"collector.py","02daf65bb7fe0526b2a700d17013054f7b0eea46","collector")
drv=load(S/"driver.py","0735238e15f13fd75e3f6939fc31aa834d8b5854","driver")
c.verify_code();need(torch.__version__=="2.10.0+cpu" and not drv.preflight()["blockers"],"CPU reference preflight")
ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True);need(ident["teacher"]==drv.TEACHER,"teacher")
raw=(R/"run-d01b-r19/corpus/corpus.jsonl").read_bytes();need(sha(raw)=="9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4","corpus")
planraw=(R/"r23-results/r23h-all8-one-swap-prefill-v1/plan.json").read_bytes();need(sha(planraw)=="a80858f97ff201b3cf62f25c779ab0786bb6d0a60973204c6fc1f3dd1f8c700b","plan")
graw=(R/"r23g-recovery-ssd-38ed4798/recovery-receipt.json").read_bytes();need(sha(graw)=="ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad","recovery")
g=json.loads(graw)
pp=R/"r23g-recovery-ssd-38ed4798/prop-plan.json";need(hashlib.sha1(b"blob "+str(pp.stat().st_size).encode()+b"\0"+pp.read_bytes()).hexdigest()=="cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde","fixed rows")
rows=json.loads(pp.read_bytes())["rows"];ctx={x["id"]:x for x in map(json.loads,raw.splitlines()) if x["split"]=="calibration"}
torch.set_num_threads(8);torch.set_num_interop_threads(1)
e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=8);orig=e._moe
out=[];state={}
def capture(l,x,phase):
 if l.idx in (10,15,25,40,42):
  ix=[len(x)//2-1,len(x)-1] if phase=="prefill" else [0]
  z=x[ix].clone();top,w=e._route(l,z,phase)
  ref=c.engine_module.expert_combine(z,top,w,l.gate,l.up,l.down)
  out.append(dict(id=state["id"],layer=l.idx,phase=phase,x=z,top=top,weights=w,reference=ref))
 return orig(l,x,phase)
e._moe=capture
for i in (0,2):
 r=rows[i];state["id"]=r["id"];prompt=ctx[r["id"]]["prompt_ids"];need(c.canonical(prompt)==r["prompt_ids_sha256"],"prompt")
 old=R/"r23-results/r23g-l38-propagation-v1"/f"{i:02}.json";need(sha(old.read_bytes())==g["receipt_sha256"][i],"baseline receipt")
 expected=json.loads(old.read_bytes())["original96"]["teacher_hashes"][:2]
 seen=c.forward_sequence(e,prompt,r["teacher_prefix_tokens"][:1]);need(all(v["sha256"]==w["sha256"] and v["next"]==w["next"] for v,w in zip(seen,expected)),"teacher exact")
need(len(out)==20,"20 routed witnesses")
with (D/"witnesses.pt").open("xb") as f:torch.save(dict(teacher=drv.TEACHER,rows=out),f)
print(json.dumps(dict(status="REAL_FIT_CPU_EXPERT_WITNESSES_CAPTURED",count=len(out),sha256=sha((D/"witnesses.pt").read_bytes()))),flush=True)
