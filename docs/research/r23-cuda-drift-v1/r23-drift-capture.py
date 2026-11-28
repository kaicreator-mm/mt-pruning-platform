"""Exact CPU prefill layer-boundary witnesses on two pinned Fit prompts."""
import pathlib,json,hashlib,torch,sys,importlib.util
import mtpruning.engine as E
P=pathlib.Path;D=P(__file__).parent;R=P.home()/".cache/mt-pruning";S=R/"r23h-pareto-screen-v1"
def need(v,m):
 if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def pin(p,h,git=False):
 b=p.read_bytes();z=hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest() if git else sha(b);need(z==h,"pin "+str(p));return b
def load(n,h):
 p=S/(n+".py");pin(p,h,True);s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
c=load("collector","02daf65bb7fe0526b2a700d17013054f7b0eea46");drv=load("driver","0735238e15f13fd75e3f6939fc31aa834d8b5854")
c.verify_code();need(torch.__version__=="2.10.0+cpu" and not drv.preflight()["blockers"],"CPU preflight")
ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True);need(ident["teacher"]==drv.TEACHER,"teacher")
raw=pin(R/"run-d01b-r19/corpus/corpus.jsonl","9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4")
ctx={x["id"]:x for x in map(json.loads,raw.splitlines()) if x["split"]=="calibration"}
pp=json.loads(pin(R/"r23g-recovery-ssd-38ed4798/prop-plan.json","cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde",True))
g=json.loads(pin(R/"r23g-recovery-ssd-38ed4798/recovery-receipt.json","ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"))
torch.set_num_threads(8);torch.set_num_interop_threads(1)
e=E.Engine(E.Checkpoint(drv.SOURCE),threads=8);norm=E.rms_norm;moe=e._moe
weights={w.data_ptr():(l.idx,k) for l in e.layers for k,w in (("ln1",l.ln1),("q_norm",l.q_norm),("k_norm",l.k_norm),("ln2",l.ln2))}
weights[e.final_norm.data_ptr()]=(len(e.layers),"final")
need(len(weights)==4*len(e.layers)+1,"weight identities")
for i in (0,2):
 r=pp["rows"][i];ids=ctx[r["id"]]["prompt_ids"];need(c.canonical(ids)==r["prompt_ids_sha256"],"prompt")
 hist=json.loads(pin(R/"r23-results/r23g-l38-propagation-v1"/f"{i:02}.json",g["receipt_sha256"][i]))
 rec={l.idx:{} for l in e.layers};final={}
 def observe(x,w,eps):
  l,k=weights[w.data_ptr()];y=norm(x,w,eps)
  d=final if k=="final" else rec[l]
  need(k not in d,"duplicate");d[k]=x.clone();d[k+"_out"]=y.clone();return y
 def onroute(l,phase,top,w):
  need(phase=="prefill","phase");rec[l]["top"]=top.clone();rec[l]["weights"]=w.clone()
 def observe_moe(l,x,phase):
  y=moe(l,x,phase);rec[l.idx]["moe"]=y.clone();return y
 E.rms_norm=observe;e.on_route=onroute;e._moe=observe_moe
 try:
  z=e.forward(ids,e.new_cache(len(ids)),phase="prefill")
 finally:
  E.rms_norm=norm;e.on_route=None;del e._moe
 want=hist["original96"]["teacher_hashes"][0];need(c.logits_sha(z)==want["sha256"] and int(z[-1].float().argmax())==want["next"],"historical teacher logits")
 for l in e.layers:
  d=rec[l.idx];need(set(d)==({"ln1","ln1_out","q_norm","q_norm_out","k_norm","k_norm_out","ln2","ln2_out"}|({"top","weights","moe"} if l.sparse else set())),"layer coverage")
  d["output"]=rec[l.idx+1]["ln1"] if l.idx+1<len(e.layers) else final["final"]
 p=D/f"cpu-layer-{i:02}.pt"
 with p.open("xb") as f:torch.save(dict(id=r["id"],teacher=drv.TEACHER,ids=ids,layers=rec,final=final,logits=z[[len(ids)//2-1,len(ids)-1]].clone()),f)
 print(json.dumps(dict(status="CPU_LAYER_WITNESS",id=r["id"],layers=len(rec),sha256=sha(p.read_bytes()),bytes=p.stat().st_size)),flush=True)
