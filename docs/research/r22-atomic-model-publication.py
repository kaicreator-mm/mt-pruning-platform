"""R22 preparation: crash-safe, non-overwriting publication of a model bundle.
Linux only; no inference; real writing requires resource admission by R21.
The parent is an owner-controlled dedicated directory on a local filesystem.
"""
import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

SLICER_BLOB = "b2cd41401b73fbad04e4ced615d7ea1861c14183"
def need(ok,why):
    if not ok: raise ValueError(why)
def load_slicer(path):
    b=path.read_bytes()
    need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==SLICER_BLOB,"slicer blob")
    spec=importlib.util.spec_from_file_location("p13",path)
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);return s
def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
def rename_exclusive(old,new):
    # No fallback to an overwriting rename or a check-then-rename race.
    libc=ctypes.CDLL(None,use_errno=True)
    fn=getattr(libc,"renameat2",None)
    need(fn is not None,"renameat2 unavailable")
    fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    fn.restype=ctypes.c_int
    if fn(-100,os.fsencode(old),-100,os.fsencode(new),1):
        e=ctypes.get_errno();raise OSError(e,os.strerror(e),str(new))
def publish(s,src,dest,mask,source_sha,mask_sha,reserve,reader_check,**profile_args):
    src=Path(src);dest=Path(dest)
    need(dest.is_absolute() and dest.parent==dest.parent.resolve() and dest.name not in ("",".",".."),"absolute canonical destination")
    need(dest.parent.is_dir() and not dest.parent.is_symlink(),"dedicated parent directory")
    need(not os.path.lexists(dest),"destination exists")
    need(src.is_file() and not src.is_symlink(),"source regular file")
    # P13 checks mask and disk reserve again inside its writer.
    stage=Path(tempfile.mkdtemp(prefix=".unpublished-",dir=dest.parent))
    published=False
    try:
        model=stage/"model.gguf"
        receipt=s.write_new(src,model,mask,source_sha,mask_sha,reserve,**profile_args)
        # Independent reader must validate the entire output descriptor inventory.
        witness=reader_check(model)
        need(witness.get("status")=="PASS" and witness.get("tensor_count")==receipt["tensors_verified"],"reader verification")
        receipt={"status":"PUBLISHED_VERIFIED_BYTES_NOT_RUNTIME_QUALIFIED","slicer_blob":SLICER_BLOB,"artifact":receipt,"reader_witness":witness,"runtime_qualification":"NOT_RUN","quality_qualification":"NOT_RUN"}
        with (stage/"receipt.json").open("x",encoding="utf-8") as f:
            json.dump(receipt,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
        fsync_dir(stage)
        rename_exclusive(stage,dest)
        published=True
        # If this fails, report failure; do not remove a published verified bundle.
        fsync_dir(dest.parent)
        return receipt
    finally:
        if not published and stage.exists():shutil.rmtree(stage)
def check_reader(s,model,gguf_path,expected_reader_sha):
    import sys
    path=Path(gguf_path).resolve()
    actual=hashlib.sha256((path/"gguf/gguf_reader.py").read_bytes()).hexdigest()
    need(actual==expected_reader_sha,"reader module SHA256")
    sys.path.insert(0,str(path))
    from gguf import GGUFReader
    r=GGUFReader(model);h=s.parse(model)
    need(r.data_offset==h["start"] and r.alignment==h["alignment"],"reader header")
    need(len(r.tensors)==len(h["tensors"]),"reader count")
    for a,b in zip(h["tensors"],r.tensors):
        need((a["name"],a["dims"],a["type"],h["start"]+a["off"],a["bytes"])==(b.name,tuple(b.shape.tolist()),int(b.tensor_type),b.data_offset,b.n_bytes),"reader descriptor "+a["name"])
    need(r.get_field("hy_v3.expert_count").contents()==h["fields"]["hy_v3.expert_count"][2],"reader count metadata")
    return {"status":"PASS","tensor_count":len(r.tensors),"reader_module_sha256":actual}
def selftest(s):
    with tempfile.TemporaryDirectory() as d:
        p=Path(d);src=p/"teacher.gguf";s.fixture(src)
        mask={"1":[0,3],"2":[1,2]};sh=s.sha_file(src);ms=s.canon(mask)
        kw=dict(experts=4,layers=3,hidden=3,width=2,top=2)
        def oracle(model):
            h=s.parse(model);return {"status":"PASS","tensor_count":len(h["tensors"])}
        def run(dest,reader=oracle,source_sha=sh,reserve=0):
            return publish(s,src,p/dest,mask,source_sha,ms,reserve,reader,**kw)
        def reject(f):
            try:f()
            except (ValueError,OSError,RuntimeError):return
            raise AssertionError("expected failure")
        rec=run("good");need((p/"good/model.gguf").is_file(),"published model")
        need(s.sha_file(p/"good/model.gguf")==rec["artifact"]["sha256"],"published digest")
        need(json.loads((p/"good/receipt.json").read_text())==rec,"receipt")
        before=s.sha_file(p/"good/model.gguf")
        reject(lambda:run("good"))
        need(s.sha_file(p/"good/model.gguf")==before,"existing unchanged")
        reject(lambda:run("bad-reader",lambda m:{"status":"FAIL"}))
        reject(lambda:run("bad-source",source_sha="0"*64))
        reject(lambda:run("bad-disk",reserve=10**30))
        def racing_reader(model):
            (p/"race").mkdir();(p/"race/other").write_text("owner")
            return oracle(model)
        reject(lambda:run("race",racing_reader))
        need((p/"race/other").read_text()=="owner","race destination unchanged")
        (p/"symlink").symlink_to(p/"missing")
        reject(lambda:run("symlink"))
        need(not any((p/n).exists() for n in ("bad-reader","bad-source","bad-disk")),"no failure publication")
        need(not list(p.glob(".unpublished-*")),"stage cleanup")
        # Simulate process death: private stale stage is not a published bundle.
        stale=p/".unpublished-crash";stale.mkdir();(stale/"model.gguf").write_bytes(b"partial")
        need(not (p/"crashed").exists() and not (stale/"receipt.json").exists(),"crash visibility")
        need(s.sha_file(src)==sh,"source unchanged")
        print(json.dumps({"status":"PASS_SYNTHETIC_ONLY","success_publication":1,"negative_cases":6,"crash_visibility_simulated":True,"source_unchanged":True,"test_reader":"synthetic oracle, not independent production GGUFReader"}))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--slicer",type=Path,required=True)
    p.add_argument("--selftest",action="store_true")
    # No real-write CLI yet. R21 pins resource admission, and R22 adds the actual driver.
    args=p.parse_args();s=load_slicer(args.slicer)
    need(args.selftest,"preparation only: real execution requires R21")
    selftest(s)
if __name__=="__main__":main()

