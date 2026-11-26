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

# P05: real aggregate structure, and hypothetical nonuniform allocation.
# No validation outcomes used; language affinities are contextual, not semantic labels.
from collections import Counter
langs=("zh","en","my")
phases=("prefill","decode")
groups=[(d,p) for d in dirs for p in phases]
def avg(v): return sum(v)/len(v)
def mix(vs): return [avg([v[i] for v in vs]) for i in range(E)]
def rank(v): return sorted(range(E),key=lambda i:(-v[i],i))
def tv(a,b): return sum(abs(x-y) for x,y in zip(a,b))/2
def effective(v): return 1/sum(x*x for x in v)
vects={l:[share(mass[d,l,p]) for d,p in groups] for l in layers}
ranks={l:rank(mix(vects[l])) for l in layers}
layer_stats={};affinity=Counter();pair_tv={};pair_overlap={};common_counts={};union_counts={}
for l in layers:
    langvec={}
    for p in phases:
        for lang in langs:
            langvec[p,lang]=mix([share(mass[d,l,p]) for d in dirs if d.split("-")[0 if p=="prefill" else 1]==lang])
        for i,x in enumerate(langs):
            for y in langs[i+1:]:
                u,v=langvec[p,x],langvec[p,y];key=p+"/"+x+"-"+y
                pair_tv.setdefault(key,[]).append(tv(u,v))
                pair_overlap.setdefault(key,[]).append(len(set(rank(u)[:64])&set(rank(v)[:64]))/64)
        top=[set(rank(langvec[p,lang])[:64]) for lang in langs]
        common_counts.setdefault(p,[]).append(len(set.intersection(*top)))
        union_counts.setdefault(p,[]).append(len(set.union(*top)))
        for e in range(E):
            vals=[langvec[p,lang][e] for lang in langs]
            # Descriptive high affinity: >=2/3 of balanced language mass, and
            # >= uniform share in dominant language. Not a language-specific function.
            j=max(range(3),key=lambda i:(vals[i],-i))
            if sum(vals)>0 and vals[j]/sum(vals)>=2/3 and vals[j]>=1/E:affinity[p+"/"+langs[j]]+=1
    v=mix(vects[l])
    layer_stats[str(l)]={"effective_experts_inverse_simpson":effective(v),
      "ever_selected":sum(any(count[d,l,p][e]>0 for d,p in groups) for e in range(E)),
      "top64_mean":sum(v[e] for e in ranks[l][:64]),
      "top64_min":min(sum(w[e] for e in ranks[l][:64]) for w in vects[l]),
      "prefill_decode_tv":tv(mix([share(mass[d,l,"prefill"]) for d in dirs]),mix([share(mass[d,l,"decode"]) for d in dirs])),
      "common_top64_prefill":common_counts["prefill"][-1],"common_top64_decode":common_counts["decode"][-1]}
def summarize(a):
    return {"min":min(a),"mean":avg(a),"max":max(a)}
# Fixed ranking per layer, options 32..128 step 8, equal physical slots.
# Prefix marginal gains diminish, so greedy allocation maximizes the
# average objective within these discrete nested prefixes.
options=list(range(32,129,8))
curves={}
for l in layers:
    curves[l]={}
    for k in options:
        a=[sum(v[e] for e in ranks[l][:k]) for v in vects[l]]
        curves[l][k]=(avg(a),min(a))
def allocate(k,objective):
    budget=L*k
    ks={l:32 for l in layers}
    if objective=="max_min_then_mean":
        # All candidate breakpoints; feasibility is sum of each layer's
        # minimum count reaching the floor. Exact for fixed prefix rankings.
        thresholds=sorted({x[1] for c in curves.values() for x in c.values()},reverse=True)
        for floor in thresholds:
            possible={l:next((j for j in options if curves[l][j][1]>=floor),None) for l in layers}
            if all(v is not None for v in possible.values()) and sum(possible.values())<=budget:
                ks=possible;break
    while sum(ks.values())<budget:
        l=max((l for l in layers if ks[l]<128),key=lambda l:(curves[l][ks[l]+8][0]-curves[l][ks[l]][0],-l))
        ks[l]+=8
    need(sum(ks.values())==budget and all(v in options for v in ks.values()),"budget")
    return ks
result={}
for k in (96,80,64,48):
    uniform={l:k for l in layers}
    result[str(k)]={}
    for method,ks in [("uniform",uniform),("max_mean",allocate(k,"max_mean")),("max_min_then_mean",allocate(k,"max_min_then_mean"))]:
        ids={str(l):sorted(ranks[l][:ks[l]]) for l in layers}
        met=metrics(ids)
        need(sum(len(v) for v in ids.values())==L*k,"equal slots")
        met["counts"]={str(l):ks[l] for l in layers};met["count_histogram"]=dict(sorted(Counter(ks.values()).items()))
        met["same_slot_tensor_bytes_estimate"]=3330699264+9441284*L*k
        result[str(k)][method]=met
    need(result[str(k)]["max_mean"]["mean_phase_layer_mass"]>=result[str(k)]["uniform"]["mean_phase_layer_mass"]-1e-12,"mean invariant")
    need(result[str(k)]["max_min_then_mean"]["min_phase_layer_mass"]>=result[str(k)]["uniform"]["min_phase_layer_mass"]-1e-12,"floor invariant")
print(json.dumps({"status":"CALIBRATION_PROXY_ONLY_NONUNIFORM_NOT_RUNTIME_SUPPORTED","input_sha256":pins,"layers":L,"language_context":{"affinity_count":dict(affinity),"threshold":"dominant >=2/3 and >=1/128","pair_tv":{k:summarize(v) for k,v in pair_tv.items()},"pair_top64_overlap":{k:summarize(v) for k,v in pair_overlap.items()},"common_top64":{k:summarize(v) for k,v in common_counts.items()},"union_top64":{k:summarize(v) for k,v in union_counts.items()}},"layer_stats":layer_stats,"allocations":result},allow_nan=False,separators=(",",":")))
