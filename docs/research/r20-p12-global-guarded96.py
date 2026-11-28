"""P03 offline R19 aggregate comparison, standard library only."""
import ast,hashlib,json,math,struct,zipfile,importlib.util,pathlib,sys
root=pathlib.Path(sys.argv[1])
def need(v,msg):
    if not v: raise ValueError(msg)
def raw(n): return (root/n).read_bytes()
def read(n): return json.loads(raw(n))
def lines(n): return [json.loads(x) for x in raw(n).splitlines() if x]
pins={"stats.npz":"8190126a4e69943cd7bceb5f136b8ca9b9201c06bda005b2d24ffd9b229a4169","stats.json":"aa1092914aaa4310397a9e87e9dd3d4897bba9e2cda7001481487fd7d5912c72","records.jsonl":"37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8","corpus.jsonl":"9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"}
for k,v in pins.items(): need(hashlib.sha256(raw(k)).hexdigest()==v,k+" hash mismatch")
for k,v in {"prep.py":"6eec10aa3b345bd2a570ccc4f4b7c7a8791c74af","selection.json":"d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"}.items():
    b=raw(k); need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==v,k+" git blob")
s=read("selection.json"); summary=read("stats.json"); dirs=summary["directions"]; layers=summary["layers"]; D,L,E=len(dirs),len(layers),128
need(D==6 and layers==list(range(1,48)) and summary["sentences"]==dict.fromkeys(dirs,24),"axes")
cs={x["id"]:x for x in lines("corpus.jsonl") if x["split"]=="calibration"}
hd,*rs=lines("records.jsonl")
need(hd["mode"]=="real" and hd["num_experts"]==128 and hd["masked_keep"] is None and hd["corpus_sha256"]==pins["corpus.jsonl"],"header")
rs=[x for x in rs if x["split"]=="calibration"]
need(len(rs)==len(cs)==144 and set(x["id"] for x in rs)==set(cs),"calibration IDs")
tokens={d:{"prefill":0,"decode":0} for d in dirs}
for x in rs:
    c=cs[x["id"]];need(x["direction"]==c["direction"] and x["line"]==c["line"] and x["prompt_tokens"]==len(c["prompt_ids"]) and x["new_tokens"]==len(x["tokens"]),"context")
    tokens[x["direction"]]["prefill"]+=x["prompt_tokens"];tokens[x["direction"]]["decode"]+=x["new_tokens"]
need(tokens==summary["tokens"],"summary tokens")
def npy(z,name,dtype,shape):
    b=z.read(name+".npy");need(b[:6]==b"\x93NUMPY" and b[6] in (1,2,3),"npy magic/version")
    nbytes=2 if b[6]==1 else 4;start=8+nbytes;n=int.from_bytes(b[8:start],"little")
    hdr=ast.literal_eval(b[start:start+n].decode("utf-8" if b[6]==3 else "latin1"))
    need(hdr=={"descr":dtype,"fortran_order":False,"shape":shape},"metadata "+name)
    p=b[start+n:];need(len(p)==8*math.prod(shape),"bytes "+name)
    return [a[0] for a in struct.iter_unpack("<"+("q" if dtype=="<i8" else "d"),p)]
mass={};count={}
with zipfile.ZipFile(root/"stats.npz") as z:
    need(len(z.namelist())==7 and set(z.namelist())=={n+".npy" for n in ["sentences","count_prefill","count_decode","mass_prefill","mass_decode","tokens_prefill","tokens_decode"]},"members")
    need(npy(z,"sentences","<i8",(D,))==[24]*D,"sentences")
    for p in ("prefill","decode"):
        ts=npy(z,"tokens_"+p,"<i8",(D,));need(ts==[tokens[d][p] for d in dirs],"tokens")
        c=npy(z,"count_"+p,"<i8",(D,L,E));m=npy(z,"mass_"+p,"<f8",(D,L,E))
        for di,d in enumerate(dirs):
            for li,l in enumerate(layers):
                j=(di*L+li)*E;cv=c[j:j+E];mv=m[j:j+E]
                need(sum(cv)==ts[di]*8 and all(0<=v<=ts[di] for v in cv),"count coverage")
                need(all(math.isfinite(v) and v>=0 and (n>0 or v==0) for v,n in zip(mv,cv)) and sum(mv)>0,"mass coverage")
                mass[d,l,p]=mv;count[d,l,p]=cv
profiles={str(l):{d:{p:mass[d,l,p] for p in ("prefill","decode")} for d in dirs} for l in layers}
sp=importlib.util.spec_from_file_location("prep",root/"prep.py");prep=importlib.util.module_from_spec(sp);sp.loader.exec_module(prep)
def share(v):return [x/sum(v) for x in v]
def original(k):
    out={}
    for l in layers:
        vs=[share([a+b for a,b in zip(mass[d,l,"prefill"],mass[d,l,"decode"])]) for d in dirs]
        out[str(l)]=sorted(sorted(range(E),key=lambda e:(-sum(v[e] for v in vs),e))[:k])
    return out
def metrics(ids):
    ratios={};counts=[];direction={}
    for d in dirs:
        kept=total=0
        for p in ("prefill","decode"):
            vlist=[]
            for l in layers:
                es=ids[str(l)];v=mass[d,l,p];c=count[d,l,p];a=sum(v[i] for i in es);b=sum(v)
                vlist.append(a/b);kept+=a;total+=b;counts.append(sum(c[i] for i in es)/sum(c))
            ratios[d+"/"+p]=vlist
        direction[d]=kept/total
    vals=[v for vs in ratios.values() for v in vs];g={k:sum(v)/L for k,v in ratios.items()}
    return {"mean_phase_layer_mass":sum(vals)/len(vals),"min_phase_layer_mass":min(vals),"min_group_mean":min(g.values()),"worst_group":min(g,key=g.get),"direction_mass":direction,"mean_count_fraction":sum(counts)/len(counts),"group_mean":g}

# P12: one global guarded selection from a finite layer-option inventory.
# Fit on original teacher calibration only. Not measured translation quality.
import numpy as np
import scipy
from scipy.optimize import milp,Bounds,LinearConstraint
need(np.__version__=="2.5.5" and scipy.__version__=="1.16.3","solver dependency versions")
groups=[(d,p) for d in dirs for p in ("prefill","decode")]
primary=[i for i,(d,p) in enumerate(groups) if d in ("zh-my","my-zh")]
secondary=[i for i in range(len(groups)) if i not in primary]
K=96;orig=original(K);need(orig==s["candidates"]["96"]["kept"],"original replay")
p04=read("p04.json");need(p04["source_head"]=="3212303e1e02994a5124db8c85195b91325c7a7c" and p04["input_sha256"]==pins and p04["keep"]==K,"P04 provenance")
plans={name:item["kept"] for name,item in p04["plans"].items()}
for name,ids in plans.items():
    need(set(ids)==set(orig) and all(len(v)==K and v==sorted(set(v)) for v in ids.values()),"P04 shape")
    need(hashlib.sha256(json.dumps(ids,sort_keys=True,separators=(",",":")).encode()).hexdigest()==p04["plans"][name]["mask_sha256"],"P04 digest")
def totals(v,ids):return v[:,ids].sum(axis=1)
def vectors(l):return np.asarray([share(mass[d,l,p]) for d,p in groups])
basev=np.stack([totals(vectors(l),orig[str(l)]) for l in layers])
primary_floor=float(basev[:,primary].min());global_floor=float(basev.min())
options=[]
for li,l in enumerate(layers):
    v=vectors(l);key=str(l)
    candidates={"original":orig[key],**{n:x[key] for n,x in plans.items()}}
    for weight in (1.5,2,3):
        utility=sum(v[j]*(weight if j in primary else 1) for j in range(len(groups)))
        candidates["primary"+str(weight)]=sorted(sorted(range(E),key=lambda e:(-utility[e],e))[:K])
    seen=set()
    for label,ids in candidates.items():
        if tuple(ids) in seen:continue
        seen.add(tuple(ids))
        vals=totals(v,ids)
        # Never make the global worst single cell worse than original.
        if min(vals)<global_floor-1e-12 or min(vals[primary])<primary_floor-1e-12:continue
        options.append((li,l,label,ids,vals))
n=len(options)
need(n>=L,"candidate availability")
V=np.stack([x[4] for x in options])
# One option for each layer. Across the entire model, protect each primary
# phase group; allow at most 0.5pp loss in secondary group mean.
A=[];lo=[];hi=[]
for li in range(L):
    A.append([float(x[0]==li) for x in options]);lo.append(1);hi.append(1)
for j in range(len(groups)):
    A.append(V[:,j]);lo.append(float(basev[:,j].sum())-(0 if j in primary else .005*L));hi.append(np.inf)
A.append(V.mean(axis=1));lo.append(float(basev.mean(axis=1).sum()));hi.append(np.inf)
# Also protect original token-weighted direction mass for zh-my and my-zh,
# not just equal-phase group means.
for d in ("zh-my","my-zh"):
    vals=[];b=0
    for l in layers:
        b+=sum(sum(mass[d,l,p][e] for e in orig[str(l)]) for p in ("prefill","decode"))
    denom=sum(sum(mass[d,l,p]) for l in layers for p in ("prefill","decode"))
    for _,l,_,ids,_ in options:
        vals.append(sum(sum(mass[d,l,p][e] for e in ids) for p in ("prefill","decode"))/denom)
    A.append(vals);lo.append(b/denom);hi.append(np.inf)
# Maximize mean layer-local weakest primary cell, with secondary tie
# preference for primary mean coverage. The bounded option inventory
# is not the set of all possible expert masks.
score=V[:,primary].min(axis=1)+.25*V[:,primary].mean(axis=1)
res=milp(c=-score,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=LinearConstraint(np.asarray(A),np.asarray(lo),np.asarray(hi)),options={"time_limit":30.0,"mip_rel_gap":.00005})
need(res.x is not None and res.status in (0,1),"no feasible solver result: "+str(res.message))
chosen=np.rint(res.x).astype(int)
need(np.max(np.abs(res.x-chosen))<1e-5 and np.all((chosen==0)|(chosen==1)),"integrality")
check=np.asarray(A)@chosen
need(np.all(check>=np.asarray(lo)-1e-8) and np.all(check<=np.asarray(hi)+1e-8),"solver constraints")
mask={};decisions=[]
for x,sel in zip(options,chosen):
    if sel:
        li,l,label,ids,vals=x
        need(str(l) not in mask,"duplicate layer")
        mask[str(l)]=ids
        decisions.append({"layer":l,"option":label,"primary_min":float(vals[primary].min()),"original_primary_min":float(basev[li,primary].min())})
need(set(mask)==set(orig) and all(len(v)==K and v==sorted(set(v)) and all(0<=i<E for i in v) for v in mask.values()),"final shape")
def summarize(ids):
    m=metrics(ids)
    m["primary_min_group"]=min(m["group_mean"][groups[j][0]+"/"+groups[j][1]] for j in primary)
    m["primary_mean_group"]=sum(m["group_mean"][groups[j][0]+"/"+groups[j][1]] for j in primary)/len(primary)
    m["primary_min_cell"]=min(sum(share(mass[d,l,p])[e] for e in ids[str(l)]) for l in layers for d,p in groups if d in ("zh-my","my-zh"))
    m["primary_mean_layer_min"]=sum(min(sum(share(mass[d,l,p])[e] for e in ids[str(l)]) for d,p in groups if d in ("zh-my","my-zh")) for l in layers)/L
    m["overlap_original"]=sum(len(set(ids[l])&set(orig[l])) for l in ids)/(L*K)
    m["mask_sha256"]=hashlib.sha256(json.dumps(ids,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return m
base=summarize(orig);m=summarize(mask)
need(m["min_phase_layer_mass"]>=base["min_phase_layer_mass"]-1e-9 and m["primary_min_cell"]>=base["primary_min_cell"]-1e-9 and m["mean_phase_layer_mass"]>=base["mean_phase_layer_mass"]-1e-9,"recomputed floors")
for d,p in groups:
    g=d+"/"+p;need(m["group_mean"][g]>=base["group_mean"][g]-(0 if d in ("zh-my","my-zh") else .005)-1e-9,"recomputed group floor")
for d in ("zh-my","my-zh"):need(m["direction_mass"][d]>=base["direction_mass"][d]-1e-9,"recomputed direction floor")
print(json.dumps({"status":"GLOBAL_GUARDED96_CALIBRATION_PROXY_ONLY","source_head":"3212303e1e02994a5124db8c85195b91325c7a7c","input_sha256":pins,"p04_blob":"daeca1c6e4e55b86d3981c6246d63e89cf7907e2","keep":K,"top_k":8,"constraints":{"primary_phase_group_mean_loss":0,"primary_token_weighted_direction_loss":0,"secondary_phase_group_max_loss":.005,"overall_mean_loss":0,"global_min_cell_loss":0,"primary_min_cell_loss":0},"solver":{"numpy":np.__version__,"scipy":scipy.__version__,"options":n,"status":int(res.status),"message":str(res.message),"gap":float(res.mip_gap) if getattr(res,"mip_gap",None) is not None else None,"objective":float(score@chosen)},"results":{"original":base,**{name:summarize(ids) for name,ids in plans.items()},"global_guarded96":m},"candidate":{"kept":mask,"mask_sha256":m["mask_sha256"]},"layer_decisions":decisions},allow_nan=False,separators=(",",":")))

