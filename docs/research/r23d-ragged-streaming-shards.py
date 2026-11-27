"""Streaming multi-shard ragged HF research pack, synthetic-only admission.
Never loads tensor bodies in the writer. Full router axis, raw BF16/F32 bytes.
Atomic exclusive directory publication; interrupted stage retained for inspection.
"""
import argparse,ctypes,hashlib,os,json,math,pathlib,re,struct,tempfile,shutil,importlib.util,sys
P=pathlib.Path
FORMAT="r23-ragged-hf-sharded-full-router-v1"
ADAPTER="56ceecad1fd2db0d33399840e2231c718c7e6ab0"
DT={"BF16":2,"F32":4}
CHUNK=65536
def need(x,m):
    if not x:raise ValueError(m)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def strict(b):
    def pairs(xs):
        d={}
        for k,v in xs:need(k not in d,"duplicate JSON key");d[k]=v
        return d
    return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def regular(p):
    need(p.is_file() and not p.is_symlink(),"regular file "+str(p));return p
def read(p):return regular(p).read_bytes()
def fsha(p):
    h=hashlib.sha256()
    with regular(p).open("rb") as f:
        for b in iter(lambda:f.read(CHUNK),b""):h.update(b)
    return h.hexdigest()
def fsdir(p):
    fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def write(p,b):
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
def safe_name(n):
    need(isinstance(n,str) and re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors",n) and "/" not in n,"shard name");return n
def descriptors(p):
    size=regular(p).stat().st_size
    with p.open("rb") as f:
        prefix=f.read(8);need(len(prefix)==8,"header prefix");length=struct.unpack("<Q",prefix)[0]
        need(2<=length<=16<<20 and length<=size-8,"bounded header")
        header=f.read(length);need(len(header)==length and header.startswith(b"{"),"header")
    data=strict(header);need(isinstance(data,dict),"header object")
    meta=data.pop("__metadata__",{});need(isinstance(meta,dict) and all(isinstance(v,str) for v in meta.values()),"metadata")
    out={};start=8+length;spans=[]
    for k,v in data.items():
        need(set(v)=={"dtype","shape","data_offsets"} and v["dtype"] in DT and isinstance(v["shape"],list) and all(type(i) is int and i>0 for i in v["shape"]),"descriptor")
        a,b=v["data_offsets"];need(type(a) is type(b) is int and 0<=a<b<=size-start and b-a==math.prod(v["shape"])*DT[v["dtype"]],"tensor extent")
        out[k]=dict(dtype=v["dtype"],shape=v["shape"],offset=start+a,bytes=b-a);spans.append((a,b))
    end=0
    for a,b in sorted(spans):need(a==end,"gap/overlap");end=b
    need(start+end==size,"trailing bytes");return out
def expected_shapes(cfg,sp,mask):
    h=cfg["hidden_size"];w=cfg["moe_intermediate_size"];dense=cfg["intermediate_size"];v=cfg["vocab_size"];hd=cfg["head_dim"];a=cfg["num_attention_heads"]*hd;kv=cfg["num_key_value_heads"]*hd;n=cfg["num_experts"]
    out={}
    def put(k,s,d="BF16"):out[k]=(list(s),d)
    def mlp(p,width):
        put(p+"gate_proj.weight",(width,h));put(p+"up_proj.weight",(width,h));put(p+"down_proj.weight",(h,width))
    put("model.embed_tokens.weight",(v,h));put("lm_head.weight",(v,h));put("model.norm.weight",(h,))
    for l,sparse in enumerate(sp):
        p=f"model.layers.{l}."
        for k in ("input_layernorm.weight","post_attention_layernorm.weight"):put(p+k,(h,))
        for k,shape in (("q_proj",(a,h)),("k_proj",(kv,h)),("v_proj",(kv,h)),("o_proj",(h,a)),("q_norm",(hd,)),("k_norm",(hd,))):put(p+"self_attn."+k+".weight",shape)
        if sparse:
            put(p+"mlp.router.gate.weight",(n,h));put(p+"mlp.expert_bias",(n,),"F32");mlp(p+"mlp.shared_mlp.",w)
            for e in range(len(mask[str(l)])):mlp(p+f"mlp.experts.{e}.",w)
        else:mlp(p+"mlp.",dense)
    return out
def check_shapes(desc,shapes):
    need(set(desc)==set(shapes),"tensor inventory")
    for k,d in desc.items():need((d["shape"],d["dtype"])==shapes[k],"shape/dtype "+k)
def source_plan(r,source,mask,cap):
    need(source.is_dir() and not source.is_symlink(),"source directory")
    pr=read(source/"checkpoint_provenance.json");prov=strict(pr);need(prov["kind"]=="synthetic","synthetic only")
    cfgraw=read(source/"config.json");cfg=strict(cfgraw);sp=r.validate(cfg,mask)
    index=strict(read(source/"model.safetensors.index.json"));wm=index["weight_map"];files=sorted(set(map(safe_name,wm.values())))
    want={"config.json","model.safetensors.index.json",*files}
    need(set(prov["files"])==want and sum(regular(source/n).stat().st_size for n in want)<=64<<20,"bounded synthetic source inventory")
    need(all(fsha(source/n)==h for n,h in prov["files"].items()),"source provenance")
    desc={}
    for f in files:
        for k,v in descriptors(source/f).items():
            need(k not in desc and wm.get(k)==f,"source index disagreement");desc[k]=dict(v,file=f)
    full={str(l):list(range(cfg["num_experts"])) for l,x in enumerate(sp) if x}
    check_shapes(desc,expected_shapes(cfg,sp,full))
    need(set(desc)==set(wm),"source index inventory")
    mp=r.mapping(cfg,mask);shapes=expected_shapes(cfg,sp,mask)
    need(set(mp)==set(shapes),"compact inventory")
    need(type(cap) is int and CHUNK<=cap<=1<<30,"shard payload cap")
    groups=[];g=[];used=0
    for name,old in sorted(mp.items()):
        v=desc[old];need(v["bytes"]<=cap,"tensor exceeds shard payload cap")
        if g and used+v["bytes"]>cap:groups.append(g);g=[];used=0
        g.append(dict(v,name=name,source_name=old));used+=v["bytes"]
    if g:groups.append(g)
    return cfgraw,prov,sha(pr),groups
def header(group):
    d={};off=0
    for t in group:
        d[t["name"]]=dict(dtype=t["dtype"],shape=t["shape"],data_offsets=[off,off+t["bytes"]]);off+=t["bytes"]
    b=canon(d);b+=b" "*((-len(b))%8);return struct.pack("<Q",len(b))+b
def copy_tensor(inp,out,offset,size):
    inp.seek(offset);h=hashlib.sha256();left=size
    while left:
        n=min(CHUNK,left);b=inp.read(n);need(len(b)==n,"short source read");out.write(b);h.update(b);left-=n
    return h.hexdigest()
def stage(r,source,dest,mask,cap):
    need(dest.parent.is_dir() and not dest.parent.is_symlink() and not os.path.lexists(dest),"new destination")
    cfg,prov,ph,groups=source_plan(r,source,mask,cap)
    folder=P(tempfile.mkdtemp(prefix="."+dest.name+".unpublished-",dir=dest.parent))
    write(folder/"config.json",cfg);wm={};files={};hashes={}
    for i,g in enumerate(groups,1):
        f=f"compact-{i:05d}-of-{len(groups):05d}.safetensors"
        with (folder/f).open("xb") as out:
            out.write(header(g))
            for t in g:
                with regular(source/t["file"]).open("rb") as inp:hashes[t["name"]]=copy_tensor(inp,out,t["offset"],t["bytes"])
                wm[t["name"]]=f
            out.flush();os.fsync(out.fileno())
        files[f]=fsha(folder/f)
    m=dict(format=FORMAT,source_kind="synthetic",engine_blob=r.ENGINE,source_provenance_sha256=ph,config_sha256=sha(cfg),kept=mask,mask_sha256=sha(canon(mask)),shard_payload_cap=cap,weight_map=wm,shards=files,tensor_sha256=hashes)
    raw=canon(m)+b"\n";write(folder/"ragged_manifest.json",raw);fsdir(folder);return folder,sha(raw)
def inspect(r,folder,digest):
    need(folder.is_dir() and not folder.is_symlink(),"pack directory")
    raw=read(folder/"ragged_manifest.json");m=strict(raw);need(sha(raw)==digest and raw==canon(m)+b"\n" and m["format"]==FORMAT and m["source_kind"]=="synthetic" and m["engine_blob"]==r.ENGINE,"manifest")
    cfgraw=read(folder/"config.json");need(sha(cfgraw)==m["config_sha256"],"config");cfg=strict(cfgraw);mask=m["kept"];sp=r.validate(cfg,mask);need(sha(canon(mask))==m["mask_sha256"],"mask")
    files=m["shards"];wm=m["weight_map"];need(set(map(safe_name,files))==set(wm.values()) and set(p.name for p in folder.iterdir())=={"config.json","ragged_manifest.json",*files},"pack inventory")
    desc={}
    for f,h in sorted(files.items()):
        need(fsha(folder/f)==h,"shard digest");d=descriptors(folder/f);need(sum(t["bytes"] for t in d.values())<=m["shard_payload_cap"],"shard cap")
        for k,t in d.items():
            need(k not in desc and wm.get(k)==f,"pack index");desc[k]=dict(t,file=f)
    check_shapes(desc,expected_shapes(cfg,sp,mask));need(set(desc)==set(wm)==set(m["tensor_sha256"]),"manifest tensor inventory")
    return m,cfg,sp,desc
def verify(r,source,folder,digest):
    m,cfg,sp,desc=inspect(r,folder,digest)
    config,prov,ph,groups=source_plan(r,source,m["kept"],m["shard_payload_cap"])
    need(m["source_provenance_sha256"]==ph and sha(config)==m["config_sha256"],"source binding")
    for g in groups:
        for t in g:
            z=desc[t["name"]];need((z["shape"],z["dtype"],z["bytes"])==(t["shape"],t["dtype"],t["bytes"]),"slice shape")
            h=hashlib.sha256()
            with regular(source/t["file"]).open("rb") as a,regular(folder/z["file"]).open("rb") as b:
                a.seek(t["offset"]);b.seek(z["offset"]);left=t["bytes"]
                while left:
                    n=min(CHUNK,left);x=a.read(n);y=b.read(n);need(len(x)==n and x==y,"raw tensor mismatch");h.update(y);left-=n
            need(h.hexdigest()==m["tensor_sha256"][t["name"]],"tensor digest")
    return m
def publish(r,source,folder,dest,digest):
    need(folder.parent==dest.parent and folder.name.startswith("."+dest.name+".unpublished-") and not os.path.lexists(dest),"exclusive destination")
    verify(r,source,folder,digest);fsdir(folder)
    libc=ctypes.CDLL(None,use_errno=True);fn=libc.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
    if fn(-100,os.fsencode(folder),-100,os.fsencode(dest),1)!=0:raise OSError(ctypes.get_errno(),"exclusive renameat2")
    fsdir(dest.parent);return digest
def checkpoint_class(r):
    from safetensors import safe_open
    class Sharded(r.RaggedCheckpoint):
        def __init__(self,path,digest):
            self.path=P(path);m,self.config,self.sparse,desc=inspect(r,self.path,digest)
            self.mask=m["kept"];self.sparse_layers=[i for i,v in enumerate(self.sparse) if v];self.num_experts=self.config["num_experts"];self.kept=None
            self.source_map=r.mapping(self.config,self.mask);self.weight_map=m["weight_map"];self.handles={}
        def tensor(self,name):
            f=self.weight_map[name]
            if f not in self.handles:self.handles[f]=safe_open(self.path/f,framework="pt",device="cpu")
            return self.handles[f].get_tensor(name)
    return Sharded
def selftest(r):
    import torch
    from mtpruning.testing import tiny_checkpoint
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as d:
        p=P(d);src=tiny_checkpoint(p/"source",p/"none",vocab_size=256,seed=81,num_experts=128,top_k=8,layers=6)
        budgets=[64,80,96,112,128];mask={str(i):list(range(128-k,128)) for i,k in enumerate(budgets,1)}
        dest=p/"compact";folder,h=stage(r,src,dest,mask,256<<10)
        need(not dest.exists() and folder.exists(),"prepublication crash visibility")
        m=verify(r,src,folder,h)
        def reject(fn):
            try:fn()
            except (ValueError,OSError):return
            raise AssertionError("bad artifact accepted")
        reject(lambda:publish(r,src,folder,dest,"0"*64));need(folder.exists() and not dest.exists(),"failed stage preserved")
        # Simulate a competing publication; no overwrite.
        dest.mkdir();write(dest/"sentinel",b"other");reject(lambda:publish(r,src,folder,dest,h));need(read(dest/"sentinel")==b"other","existing destination intact");(dest/"sentinel").unlink();dest.rmdir()
        publish(r,src,folder,dest,h);need(dest.exists() and not folder.exists(),"atomic visibility")
        ck=checkpoint_class(r)(dest,h);need(not ck.handles,"lazy handles");orig=r.eng.Checkpoint(src)
        need(all(torch.equal(ck.tensor(k),orig.tensor(v)) for k,v in ck.source_map.items()),"independent tensors")
        a=r.eng.Engine(orig,1);a.set_mask({int(k):v for k,v in mask.items()});b=r.RaggedEngine(ck,1)
        x,y=r.compare_fixed(a,b,[4,8,15,16,23,42],[11,12,13,14,15,16,17,18])
        ga=a.generate([4,8,15],8,trace=True);gb=b.generate([4,8,15],8,trace=True)
        need(ga.tokens==gb.tokens and ga.finish==gb.finish and all(torch.equal(v,gb.trace[k]) for k,v in ga.trace.items()),"greedy trace")
        reject(lambda:stage(r,src,dest,mask,256<<10));reject(lambda:source_plan(r,src,mask,CHUNK-1))
        bad=p/"bad";shutil.copytree(dest,bad);f=next(iter(m["shards"]));raw=read(bad/f);(bad/f).write_bytes(raw[:-1]+bytes([raw[-1]^1]));reject(lambda:inspect(r,bad,h))
        bad2=p/"bad2";shutil.copytree(dest,bad2);write(bad2/"extra",b"x");reject(lambda:inspect(r,bad2,h))
        # Malformed safetensors descriptors, not merely mismatched digest.
        malformed=p/"bad.safetensors";hdr=canon({"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[2,6]}});malformed.write_bytes(struct.pack("<Q",len(hdr))+hdr+b"123456");reject(lambda:descriptors(malformed))
        need(all(fsha(src/n)==v for n,v in strict(read(src/"checkpoint_provenance.json"))["files"].items()),"source unchanged")
        print(json.dumps(dict(status="PASS_CPU_STREAMING_RAGGED_SHARDS",shards=len(m["shards"]),tensor_checks=len(ck.source_map),chunk_bytes=CHUNK,shard_payload_cap=m["shard_payload_cap"],fixed_logits=x,route_hooks=y,greedy_trace=True,negative=7,crash_visibility_simulated=True,source_unchanged=True,real_model="NOT_RUN")))
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--adapter",type=P,required=True);a.add_argument("--selftest",action="store_true");q=a.parse_args()
    need(q.selftest and blob(read(q.adapter))==ADAPTER,"synthetic only/exact adapter")
    s=importlib.util.spec_from_file_location("ragged",q.adapter);r=importlib.util.module_from_spec(s);s.loader.exec_module(r);selftest(r)
