"""R23 research-only ragged FFN storage with preserved original router axis. NOT a GGUF or deployable model.

Planning uses the pinned F16/F32 GGUF reader. Compact tensor payload has
per-layer expert counts, canonical original-ID maps and byte ranges.
Only synthetic write is exposed by CLI; no production build admission.
"""
import argparse,hashlib,importlib.util,json,math,os,pathlib,shutil,tempfile
P=pathlib.Path
SLICER="b2cd41401b73fbad04e4ced615d7ea1861c14183"
def need(v,m):
    if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def load(p):
    need(blob(p.read_bytes())==SLICER,"slicer pin")
    sp=importlib.util.spec_from_file_location("slicer",p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def plan(s,source,mask,*,experts=128,layers=48,hidden=2048,width=768,top=8):
    h=s.parse(source);s.profile(h,experts,layers,hidden,width,top)
    need(set(mask)=={str(l) for l in range(1,layers)},"layer inventory")
    need(all(isinstance(v,list) and top<=len(v)<=experts and v==sorted(set(v)) and all(type(i) is int and 0<=i<experts for i in v) for v in mask.values()),"expert IDs")
    a=h["alignment"];off=0;out=[]
    for t in h["tensors"]:
        m=s.EXPERT.fullmatch(t["name"]);dims=list(t["dims"])
        if m and m[2] not in ("exp_probs_b","ffn_gate_inp.weight"):
            ids=mask[m[1]];need(t["bytes"]%experts==0,"expert stride")
            stride=t["bytes"]//experts;dims[-1]=len(ids)
            ranges=[[h["start"]+t["off"]+i*stride,stride] for i in ids]
        else:ranges=[[h["start"]+t["off"],t["bytes"]]]
        nb=sum(n for _,n in ranges);need(nb==math.prod(dims)*s.DTYPE[t["type"]],"tensor size")
        need(all(p>=h["start"] and p+n<=h["size"] for p,n in ranges),"source bounds")
        off=s.align(off,a);out.append(dict(name=t["name"],dims=dims,type=t["type"],off=off,bytes=nb,source_ranges=ranges));off+=nb
    return dict(status="RAGGED_EXPERT_STORAGE_PLAN_NOT_RUNTIME_QUALIFIED",format="r23-ragged-ffn-full-router-pack-v1",router_axis="original_source_width",router_policy="mask original IDs before top-k; map selected original IDs to compact local FFN indices",source_header_sha256=sha(h["raw"]),source_bytes=h["size"],source_experts=experts,layers=layers,hidden=hidden,width=width,top_k=top,alignment=a,kept=mask,mask_sha256=sha(canon(mask)),expert_instances=sum(map(len,mask.values())),source_payload_bytes=sum(t["bytes"] for t in h["tensors"]),tensor_payload_bytes=sum(t["bytes"] for t in out),aligned_payload_bytes=off,tensors=out,limitations="Original GGUF header retained separately as provenance; not a GGUF; no stock loader or quantized support.")
def read_exact(f,n):
    b=f.read(n);need(len(b)==n,"short read");return b
def write_fixture(s,src,dst,p,source_sha):
    # Deliberately limited to tiny synthetic inputs; never builds real model.
    need(p["source_bytes"]<=1<<20 and src.stat().st_size==p["source_bytes"] and s.sha_file(src)==source_sha,"fixture source")
    need(not os.path.lexists(dst),"new destination")
    dst.mkdir(mode=0o700)
    hashes={}
    with src.open("rb") as inp,(dst/"tensors.bin").open("xb") as o:
        for t in p["tensors"]:
            need(o.tell()<=t["off"],"ordered payload")
            o.write(b"\0"*(t["off"]-o.tell()));d=hashlib.sha256()
            for pos,n in t["source_ranges"]:
                inp.seek(pos);b=read_exact(inp,n);o.write(b);d.update(b)
            hashes[t["name"]]=d.hexdigest()
        need(o.tell()==p["aligned_payload_bytes"],"payload length")
        o.flush();os.fsync(o.fileno())
    h=s.parse(src);(dst/"source-header.bin").write_bytes(h["raw"])
    m=dict(p,source_sha256=source_sha,tensor_sha256=hashes,payload_sha256=s.sha_file(dst/"tensors.bin"))
    (dst/"manifest.json").write_bytes(canon(m)+b"\n")
    return m
def verify(s,src,dst,mask,source_sha,**kw):
    raw=(dst/"manifest.json").read_bytes();m=json.loads(raw);need(raw==canon(m)+b"\n","canonical manifest")
    p=plan(s,src,mask,**kw)
    need(all(m.get(k)==v for k,v in p.items()) and set(m)==set(p)|{"source_sha256","tensor_sha256","payload_sha256"},"manifest replay")
    need(m["source_sha256"]==source_sha==s.sha_file(src),"source identity")
    h=s.parse(src);need((dst/"source-header.bin").read_bytes()==h["raw"],"header provenance")
    data=dst/"tensors.bin";need(data.stat().st_size==p["aligned_payload_bytes"] and s.sha_file(data)==m["payload_sha256"],"payload identity")
    need(set(m["tensor_sha256"])=={t["name"] for t in p["tensors"]},"tensor hash inventory")
    with src.open("rb") as inp,data.open("rb") as o:
        end=0
        for t in p["tensors"]:
            need(read_exact(o,t["off"]-end)==b"\0"*(t["off"]-end),"padding")
            d=hashlib.sha256()
            for pos,n in t["source_ranges"]:
                inp.seek(pos);left=n
                while left:
                    k=min(left,1<<16);a=read_exact(inp,k);b=read_exact(o,k);need(a==b,"source slice mismatch");d.update(b);left-=k
            need(d.hexdigest()==m["tensor_sha256"][t["name"]],"tensor hash");end=t["off"]+t["bytes"]
        need(o.read(1)==b"","trailing bytes")
    return m
def selftest(s):
    with tempfile.TemporaryDirectory() as d:
        p=P(d);src=p/"in.gguf";table=s.fixture(src);kw=dict(experts=4,layers=3,hidden=3,width=2,top=2);sh=s.sha_file(src)
        mask={"1":[0,3],"2":[0,1,3]};a=plan(s,src,mask,**kw);need(a["expert_instances"]==5,"budget")
        dst=p/"fixture.r23pack";write_fixture(s,src,dst,a,sh);m=verify(s,src,dst,mask,sh,**kw)
        # Independent fixture oracle, not merely planner replay.
        raw=(dst/"tensors.bin").read_bytes();count=0
        for t,(name,dims,typ,off,b) in zip(m["tensors"],table):
            need(t["name"]==name,"tensor order");x=s.EXPERT.fullmatch(name)
            if x and x[2] not in ("exp_probs_b","ffn_gate_inp.weight"):
                stride=len(b)//4;want=b"".join(b[i*stride:(i+1)*stride] for i in mask[x[1]])
                need(t["dims"]==list(dims[:-1])+[len(mask[x[1]])],"ragged shape")
            else:
                want=b;need(t["dims"]==list(dims),"unchanged router/shared/dense shape")
            need(raw[t["off"]:t["off"]+t["bytes"]]==want,"independent byte oracle");count+=1
        def reject(f):
            try:f()
            except (ValueError,OSError):return
            raise AssertionError("bad input accepted")
        reject(lambda:s.plan(src,mask,**kw))
        for bad in ({"1":[0,3]}, {"1":[0],"2":[0,1,3]}, {"1":[3,0],"2":[0,1,3]}, {"1":[0,4],"2":[0,1,3]}, {"1":[0,0],"2":[0,1,3]}):
            reject(lambda:plan(s,src,bad,**kw))
        reject(lambda:write_fixture(s,src,dst,a,sh))
        reject(lambda:verify(s,src,dst,{"1":[1,3],"2":[0,1,3]},sh,**kw))
        data=dst/"tensors.bin";data.write_bytes(bytes([raw[0]^1])+raw[1:]);reject(lambda:verify(s,src,dst,mask,sh,**kw))
        need(s.sha_file(src)==sh,"source unchanged")
        need(a["source_payload_bytes"]-a["tensor_payload_bytes"]==(8-5)*3*3*2*2,"FFN-only savings")
        print(json.dumps(dict(status="PASS_SYNTHETIC_RAGGED_FFN_FULL_ROUTER_STORAGE",tensors=count,layer_counts=[2,3],expert_instances=5,negative=9,stock_uniform_slicer_rejects=True,source_unchanged=True,real_build="NOT_RUN")))
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--slicer",required=True,type=P);a.add_argument("--selftest",action="store_true");q=a.parse_args()
    need(q.selftest,"synthetic only");selftest(load(q.slicer))
