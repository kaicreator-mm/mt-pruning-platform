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
result={}
for k in (96,80,64,48,32):
    base=original(k);bm=metrics(base)
    if str(k) in s["candidates"]:
        need(base==s["candidates"][str(k)]["kept"],"original replay "+str(k))
        need(all(abs(bm["direction_mass"][d]-s["candidates"][str(k)]["retained_mass_overall"][d])<1e-6 for d in dirs),"mass replay")
    result[str(k)]={}
    for method in ("original","balanced","worst_first","random"):
        ids=base if method=="original" else {l:x["keep_ids"] for l,x in prep.plan(profiles,k,method=method,seed=42,top_k=8,expert_count=128)["layers"].items()}
        need(set(ids)==set(base) and all(len(v)==k and len(set(v))==k and v==sorted(v) and all(0<=i<E for i in v) for v in ids.values()),"mask")
        met=metrics(ids);met["overlap_original"]=sum(len(set(ids[l])&set(base[l])) for l in ids)/(L*k);met["mask_sha256"]=hashlib.sha256(json.dumps(ids,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        result[str(k)][method]=met
print(json.dumps({"status":"CALIBRATION_PROXY_ONLY","input_sha256":pins,"original_replay_96_64":"PASS","layers":L,"directions":dirs,"results":result},allow_nan=False,separators=(",",":")))
