"""Hash-bound resumable chunk archive on Nilar. No source deletion or runtime publication."""
import argparse,hashlib,json,os,pathlib,re
P=pathlib.Path
SIZE=31747952288
FULL="c77b03276ecc1e4cb70d32f249996ed6c3f49a9e78f087bd734e7d187c20b444"
CHUNK=256*2**20
def need(x,m):
    if not x:raise ValueError(m)
def hex64(x):return type(x) is str and re.fullmatch("[0-9a-f]{64}",x) is not None
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n"
def digest(p,h=None):
    need(p.is_file() and not p.is_symlink(),"regular file")
    d=hashlib.sha256();n=0
    with p.open("rb") as f:
        for b in iter(lambda:f.read(4*2**20),b""):
            n+=len(b);d.update(b)
            if h is not None:h.update(b)
    return n,d.hexdigest()
def publish(p,x):
    with p.open("xb") as f:f.write(canon(x));f.flush();os.fsync(f.fileno())
def count(size,chunk):return (size+chunk-1)//chunk
def receipt(root,i,size,chunk,full):
    p=root/f"{i:04d}.json";q=root/f"{i:04d}.bin"
    need(p.is_file() and not p.is_symlink(),"receipt missing")
    raw=p.read_bytes();x=json.loads(raw)
    n=min(chunk,size-i*chunk)
    need(raw==canon(x) and x==dict(index=i,offset=i*chunk,bytes=n,sha256=x["sha256"],source_sha256=full) and hex64(x["sha256"]) and digest(q)==(n,x["sha256"]),"verified chunk")
    return x
def accept(root,i,src,expected,size=SIZE,chunk=CHUNK,full=FULL):
    need(0<=i<count(size,chunk) and hex64(expected),"index/hash")
    need(root.is_dir() and not root.is_symlink(),"archive root")
    n=min(chunk,size-i*chunk);x=dict(index=i,offset=i*chunk,bytes=n,sha256=expected,source_sha256=full)
    q=root/f"{i:04d}.bin";p=root/f"{i:04d}.json"
    need(src.parent==root and src.name.startswith(f"incoming-{i:04d}-") and src.suffix==".part","incoming path")
    need(digest(src)==(n,expected),"incoming bytes")
    if p.exists():
        need(receipt(root,i,size,chunk,full)==x,"existing receipt conflict")
        return x
    # Never overwrite: a prior orphan must be identical to the new source bytes.
    if q.exists():need(digest(q)==(n,expected),"orphan conflict")
    else:
        with src.open("rb") as a,q.open("xb") as b:
            for z in iter(lambda:a.read(4*2**20),b""):b.write(z)
            b.flush();os.fsync(b.fileno())
    need(digest(q)==(n,expected),"persisted chunk")
    publish(p,x);return x
def verify(root,size=SIZE,chunk=CHUNK,full=FULL,complete=False):
    n=count(size,chunk);done=[];h=hashlib.sha256()
    for i in range(n):
        p=root/f"{i:04d}.json";q=root/f"{i:04d}.bin"
        if not p.exists():
            need(not complete,"missing chunk")
            continue
        x=receipt(root,i,size,chunk,full);done.append(i)
        if complete:need(digest(q,h)==(x["bytes"],x["sha256"]),"second read")
    if complete:need(done==list(range(n)) and h.hexdigest()==full,"whole archive digest")
    return dict(status="ARCHIVE_BYTES_VERIFIED_NOT_RUNTIME" if complete else "ARCHIVE_PREFIX_VERIFIED",source_sha256=full,total_bytes=size,chunk_bytes=chunk,total_chunks=n,verified_indices=done,verified_bytes=sum(min(chunk,size-i*chunk) for i in done))
def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        r=P(d);b=b"abcdefghijk";f=hashlib.sha256(b).hexdigest()
        for i,z in enumerate((b[:4],b[4:8],b[8:])):
            p=r/f"incoming-{i:04d}-test.part";p.write_bytes(z)
            accept(r,i,p,hashlib.sha256(z).hexdigest(),len(b),4,f)
        need(verify(r,len(b),4,f,True)["verified_bytes"]==len(b),"full")
        p=r/"0001.bin";p.write_bytes(b"xxxx")
        try:verify(r,len(b),4,f,True)
        except ValueError:pass
        else:raise AssertionError("corruption accepted")
    print(json.dumps(dict(status="PASS_SYNTHETIC_CHUNK_ARCHIVE",positive=1,negative=1)))
def main():
    a=argparse.ArgumentParser();a.add_argument("--selftest",action="store_true");a.add_argument("--root",type=P);a.add_argument("--index",type=int);a.add_argument("--incoming",type=P);a.add_argument("--sha");a.add_argument("--verify",action="store_true");a.add_argument("--complete",action="store_true");q=a.parse_args()
    if q.selftest:selftest();return
    need(q.root is not None and q.root.name=="k64-f16-c77b0327" and q.root.parent.name=="transfer","fixed root")
    if q.verify:
        x=verify(q.root,complete=q.complete)
        if q.complete:
            p=q.root/"complete.json"
            if p.exists():need(p.read_bytes()==canon(x),"existing completion receipt")
            else:publish(p,x)
    else:
        need(q.index is not None and q.incoming is not None and q.sha is not None,"input")
        x=accept(q.root,q.index,q.incoming,q.sha)
        # Only remove our verified incoming duplicate, never source or published chunk.
        q.incoming.unlink()
        x=dict(status="ARCHIVE_CHUNK_VERIFIED",**x)
    print(json.dumps(x),flush=True)
if __name__=="__main__":main()
