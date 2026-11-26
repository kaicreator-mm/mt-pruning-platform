#!/usr/bin/env python3
"""Bounded direct little-endian F16/F32 GGUF v3 expert slicer.

Research prototype. No quantized tensor support; no implicit model execution.
Plan is header-only. Write requires exact input and mask SHA256, new output,
and free disk reserve. Source and all produced tensor bytes are checked.
"""
import argparse, hashlib, json, math, os, re, shutil, struct, tempfile
from pathlib import Path

SOURCE_SHA = "c0f31f9243f2be7c67971b3195606bbb5ed4fa9ceb035add2520480b67f38e5b"
MASK_SHA = "4589d613512ac0a07f02be500e8b2fabe3f4beb78ddd171e14fd26cd8078960c"
SCALARS = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
DTYPE = {0:4,1:2}
CAP = 64<<20
CHUNK = 4<<20
RESERVE = 10<<30
EXPERT = re.compile(r"^blk\.(\d+)\.(exp_probs_b|ffn_gate_inp\.weight|ffn_(?:down|gate|up)_exps\.weight)$")
FIXED = ("attn_norm.weight","ffn_norm.weight","attn_k_norm.weight","attn_k.weight","attn_output.weight","attn_q_norm.weight","attn_q.weight","attn_v.weight")
DENSE = ("ffn_down.weight","ffn_gate.weight","ffn_up.weight")
SHARED = ("ffn_down_shexp.weight","ffn_gate_shexp.weight","ffn_up_shexp.weight")
MUTABLE = ("exp_probs_b","ffn_gate_inp.weight","ffn_down_exps.weight","ffn_gate_exps.weight","ffn_up_exps.weight")
def need(ok, why):
    if not ok: raise ValueError(why)
def align(n,a): return (n+a-1)//a*a
def pack(fmt,*a): return struct.pack("<"+fmt,*a)
def string(s):
    b=s.encode("utf-8");return pack("Q",len(b))+b
def canon(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(CHUNK),b""):h.update(b)
    return h.hexdigest()
def expected_names(layers):
    return {"token_embd.weight","output_norm.weight","output.weight"}|{f"blk.{l}.{n}" for l in range(layers) for n in (*FIXED,*(DENSE if l==0 else (*SHARED,*MUTABLE)))}
def parse(path):
    size=path.stat().st_size
    with path.open("rb") as f:
        def read(n):
            need(0<=n<=CAP and f.tell()+n<=min(size,CAP),"header bound")
            b=f.read(n);need(len(b)==n,"short header");return b
        def u(fmt):return struct.unpack("<"+fmt,read(struct.calcsize("<"+fmt)))[0]
        def s():return read(u("Q")).decode("utf-8")
        def value(t):
            if t==8:return s()
            if t==9:
                sub=u("I");n=u("Q");need(sub==8 or sub in SCALARS,"unsupported array type")
                need(n<=2_000_000,"array count")
                if sub==8:
                    for _ in range(n):s()
                else:read(n*SCALARS[sub])
                return None
            need(t in SCALARS,"unsupported value type")
            b=read(SCALARS[t]);return int.from_bytes(b,"little") if t==4 else None
        need(read(4)==b"GGUF" and u("I")==3,"GGUF v3 LE only")
        nt,nk=u("Q"),u("Q");need(0<nt<=2000 and 0<nk<=10000,"table size")
        kvstart=f.tell();fields={}
        for _ in range(nk):
            k=s();t=u("I");pos=f.tell();need(k not in fields,"duplicate metadata")
            v=value(t);fields[k]=(t,pos,v)
        kvend=f.tell();tensors=[];names=set()
        for _ in range(nt):
            name=s();nd=u("I");need(1<=nd<=4 and name not in names,"tensor rank/name")
            dims=tuple(u("Q") for _ in range(nd));t=u("I");off=u("Q")
            need(all(0<d<=2**32 for d in dims) and t in DTYPE,"unsupported tensor shape/type")
            nb=math.prod(dims)*DTYPE[t];need(nb<=size,"tensor size")
            tensors.append({"name":name,"dims":dims,"type":t,"off":off,"bytes":nb});names.add(name)
        a=fields.get("general.alignment",(4,0,32))[2]
        need(type(a)==int and 1<=a<=4096 and a&(a-1)==0,"alignment")
        start=align(f.tell(),a)
        need(start<=size and all(t["off"]%a==0 and start+t["off"]+t["bytes"]<=size for t in tensors),"tensor range")
        end=0
        for t in sorted(tensors,key=lambda t:t["off"]):
            need(t["off"]>=end,"overlap");end=t["off"]+t["bytes"]
        f.seek(0);raw=read(start)
    return {"size":size,"nt":nt,"nk":nk,"kvstart":kvstart,"kvend":kvend,"fields":fields,"tensors":tensors,"start":start,"alignment":a,"raw":raw}
def profile(h,experts,layers,hidden,width,top):
    f=h["fields"]
    for k,t,v in [("general.architecture",8,"hy_v3"),("general.file_type",4,1),("hy_v3.block_count",4,layers),("hy_v3.expert_count",4,experts),("hy_v3.expert_used_count",4,top),("hy_v3.embedding_length",4,hidden),("hy_v3.expert_feed_forward_length",4,width)]:
        need(k in f and f[k][0]==t and f[k][2]==v,"metadata "+k)
    need({t["name"] for t in h["tensors"]}==expected_names(layers),"tensor inventory")
    for t in h["tensors"]:
        m=EXPERT.fullmatch(t["name"])
        if not m:continue
        l=int(m[1]);n=m[2];need(1<=l<layers,"sparse layer")
        expected=(experts,) if n=="exp_probs_b" else (hidden,experts) if n=="ffn_gate_inp.weight" else (width,hidden,experts) if n=="ffn_down_exps.weight" else (hidden,width,experts)
        need(t["dims"]==expected and t["type"]==(0 if n in MUTABLE[:2] else 1),"expert shape/type "+n)
def plan(path,mask,*,experts=128,layers=48,hidden=2048,width=768,top=8):
    h=parse(path);profile(h,experts,layers,hidden,width,top)
    need(set(mask)=={str(l) for l in range(1,layers)},"mask coverage")
    sizes={len(v) for v in mask.values()}
    need(len(sizes)==1,"uniform budget");k=sizes.pop();need(top<=k<experts,"keep count")
    need(all(isinstance(v,list) and all(type(i)==int and 0<=i<experts for i in v) and v==sorted(set(v)) for v in mask.values()),"mask IDs/order")
    out=[];off=0;a=h["alignment"];kv=bytearray(h["raw"][h["kvstart"]:h["kvend"]])
    pos=h["fields"]["hy_v3.expert_count"][1]-h["kvstart"];struct.pack_into("<I",kv,pos,k)
    header=bytearray(b"GGUF"+pack("IQQ",3,h["nt"],h["nk"])+kv)
    for t in h["tensors"]:
        m=EXPERT.fullmatch(t["name"]);dims=t["dims"]
        if m:
            stride=t["bytes"]//experts;dims=(*dims[:-1],k);ranges=[(h["start"]+t["off"]+i*stride,stride) for i in mask[m[1]]]
        else:ranges=[(h["start"]+t["off"],t["bytes"])]
        nb=sum(n for _,n in ranges);need(nb==math.prod(dims)*DTYPE[t["type"]],"slice bytes")
        off=align(off,a)
        header+=string(t["name"])+pack("I",len(dims))+pack("Q"*len(dims),*dims)+pack("IQ",t["type"],off)
        out.append({"name":t["name"],"dims":dims,"type":t["type"],"off":off,"bytes":nb,"ranges":ranges});off+=nb
    header+=b"\0"*(align(len(header),a)-len(header))
    return h,bytes(header),out,len(header)+off,k
def write_new(src,dst,mask,source_sha,mask_sha,reserve,**profile_args):
    need(src.resolve()!=dst.resolve() and not dst.exists() and not dst.is_symlink(),"output must be new")
    need(canon(mask)==mask_sha,"mask digest")
    h,header,out,sz,k=plan(src,mask,**profile_args)
    need(shutil.disk_usage(dst.parent).free>=sz+reserve,"disk reserve")
    before=src.stat();need(sha_file(src)==source_sha,"source digest")
    checks={};created=False
    try:
        with src.open("rb") as inp,dst.open("xb") as o:
            created=True;o.write(header)
            for t in out:
                target=len(header)+t["off"];need(o.tell()<=target,"output order")
                o.write(b"\0"*(target-o.tell()));digest=hashlib.sha256()
                for pos,n in t["ranges"]:
                    inp.seek(pos)
                    while n:
                        b=inp.read(min(CHUNK,n));need(bool(b),"short source")
                        o.write(b);digest.update(b);n-=len(b)
                checks[t["name"]]=digest.hexdigest()
            need(o.tell()==sz,"output size");o.flush();os.fsync(o.fileno())
        after=src.stat();need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),"source changed")
        got=parse(dst);profile(got,k,profile_args.get("layers",48),profile_args.get("hidden",2048),profile_args.get("width",768),profile_args.get("top",8))
        need(got["size"]==sz and got["raw"]==header and len(got["tensors"])==len(out),"output header")
        with dst.open("rb") as f:
            for t,expect in zip(got["tensors"],out):
                need(all(t[key]==expect[key] for key in ("name","dims","type","off","bytes")),"output tensor table")
                f.seek(got["start"]+t["off"]);n=t["bytes"];digest=hashlib.sha256()
                while n:
                    b=f.read(min(CHUNK,n));need(bool(b),"short output");digest.update(b);n-=len(b)
                need(digest.hexdigest()==checks[t["name"]],"output tensor digest "+t["name"])
        return {"status":"BYTE_SLICE_VERIFIED_NOT_RUNTIME_QUALIFIED","keep":k,"source_sha256":source_sha,"mask_sha256":mask_sha,"bytes":sz,"sha256":sha_file(dst),"tensors_verified":len(checks),"tensor_sha256":checks}
    except BaseException:
        if created:dst.unlink(missing_ok=True)
        raise
def fixture(path):
    kv=[("general.architecture",8,"hy_v3"),("general.file_type",4,1),("hy_v3.block_count",4,3),("hy_v3.expert_count",4,4),("hy_v3.expert_used_count",4,2),("hy_v3.embedding_length",4,3),("hy_v3.expert_feed_forward_length",4,2),("general.alignment",4,64)]
    # Arbitrary non-target arrays verify raw metadata preservation.
    raw=b"".join(string(n)+pack("I",t)+(string(v) if t==8 else pack("I",v)) for n,t,v in kv)
    raw+=string("test.array")+pack("IIQ",9,8,2)+string("中文")+string("မြန်မာ")
    table=[];payload=bytearray()
    for j,name in enumerate(sorted(expected_names(3))):
        m=EXPERT.fullmatch(name);dims=(3,2);t=1
        if m:
            n=m[2];dims=(4,) if n=="exp_probs_b" else (3,4) if n=="ffn_gate_inp.weight" else (2,3,4) if n=="ffn_down_exps.weight" else (3,2,4);t=0 if n in MUTABLE[:2] else 1
        off=align(len(payload),64);payload+=b"\0"*(off-len(payload));nb=math.prod(dims)*DTYPE[t]
        data=bytes((j*11+i)%251 for i in range(nb));payload+=data
        table.append((name,dims,t,off,data))
    header=b"GGUF"+pack("IQQ",3,len(table),len(kv)+1)+raw+b"".join(string(n)+pack("I",len(d))+pack("Q"*len(d),*d)+pack("IQ",t,o) for n,d,t,o,_ in table)
    path.write_bytes(header+b"\0"*(align(len(header),64)-len(header))+payload)
    return table
def selftest():
    with tempfile.TemporaryDirectory() as d:
        p=Path(d);src=p/"in.gguf";out=p/"out.gguf";table=fixture(src);mask={"1":[0,3],"2":[1,2]};kw=dict(experts=4,layers=3,hidden=3,width=2,top=2)
        sh=sha_file(src);ms=canon(mask)
        result=write_new(src,out,mask,sh,ms,0,**kw)
        h=parse(out);raw=out.read_bytes();old=parse(src)
        # Independent direct byte oracle from generated fixture, including unchanged tensors.
        count=0
        for actual,(_,dims,t,off,source_bytes) in zip(h["tensors"],table):
            m=EXPERT.fullmatch(actual["name"])
            if m:
                stride=len(source_bytes)//4;want=b"".join(source_bytes[i*stride:(i+1)*stride] for i in mask[m[1]])
            else:want=source_bytes
            start=h["start"]+actual["off"];need(raw[start:start+actual["bytes"]]==want,"fixture oracle");count+=1
        original=bytearray(old["raw"][old["kvstart"]:old["kvend"]]);struct.pack_into("<I",original,old["fields"]["hy_v3.expert_count"][1]-old["kvstart"],2)
        need(bytes(original)==h["raw"][h["kvstart"]:h["kvend"]],"unchanged metadata")
        def fails(fn):
            try:fn()
            except (ValueError,FileExistsError):return
            raise AssertionError("expected rejection")
        fails(lambda:write_new(src,out,mask,sh,ms,0,**kw))
        fails(lambda:plan(src,{"1":[0,0],"2":[1,2]},**kw))
        fails(lambda:plan(src,{"1":[3,0],"2":[1,2]},**kw))
        fails(lambda:plan(src,{"1":[0,3]},**kw))
        fails(lambda:plan(src,{"1":[0,4],"2":[1,2]},**kw))
        fails(lambda:plan(src,{"1":[0,3],"2":[0,1,2]},**kw))
        fails(lambda:write_new(src,p/"bad.gguf",mask,"0"*64,ms,0,**kw))
        fails(lambda:write_new(src,p/"bad.gguf",mask,sh,"0"*64,0,**kw))
        fails(lambda:write_new(src,p/"bad.gguf",mask,sh,ms,10**30,**kw))
        (p/"short.gguf").write_bytes(src.read_bytes()[:-1]);fails(lambda:parse(p/"short.gguf"))
        need(not (p/"bad.gguf").exists() and sha_file(src)==sh,"no mutation")
        print(json.dumps({"status":"PASS_SYNTHETIC_ONLY","tensor_byte_oracle":count,"negative_checks":10,"metadata_preserved":True,"source_unchanged":True,"output_sha256":result["sha256"]}))
def main():
    a=argparse.ArgumentParser();a.add_argument("--selftest",action="store_true");a.add_argument("--source",type=Path);a.add_argument("--mask-json");a.add_argument("--out",type=Path);a.add_argument("--write",action="store_true");args=a.parse_args()
    if args.selftest:selftest();return
    need(args.source is not None and args.mask_json is not None,"source and mask required")
    mask=json.loads(args.mask_json);need(canon(mask)==MASK_SHA,"pinned P12 mask digest")
    h,header,out,size,k=plan(args.source,mask)
    if args.write:
        need(args.out is not None,"explicit new path required")
        print(json.dumps(write_new(args.source,args.out,mask,SOURCE_SHA,MASK_SHA,RESERVE)));return
    free=shutil.disk_usage(args.source.parent).free
    print(json.dumps({"status":"HEADER_ONLY_NOT_MATERIALIZED","source":str(args.source),"source_bytes":h["size"],"source_header_sha256":hashlib.sha256(h["raw"]).hexdigest(),"source_header_bytes":h["start"],"source_expected_full_sha256":SOURCE_SHA,"source_full_sha256_rechecked":False,"mask_sha256":canon(mask),"keep":k,"tensor_count":len(out),"sliced_tensors":sum(bool(EXPERT.fullmatch(t["name"])) for t in out),"copied_tensors":sum(not EXPERT.fullmatch(t["name"]) for t in out),"planned_bytes":size,"planned_header_sha256":hashlib.sha256(header).hexdigest(),"free_bytes":free,"reserve_bytes":RESERVE,"disk_eligible":free>=size+RESERVE,"shortfall_bytes":max(0,size+RESERVE-free)}))
if __name__=="__main__":main()

