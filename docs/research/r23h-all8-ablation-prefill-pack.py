import pathlib,tarfile,hashlib
R=pathlib.Path.home()/".cache/mt-pruning";Q=R/"r23-results";D=R/"r23h-all8-ablation-screen-run-v2";O=Q/"r23h-all8-ablation-prefill-v2"
assert (D/"exit-code").read_text().strip()=="0" and (D/"run.stderr").read_bytes()==b""
assert (O/"report.json").is_file()
files={"screen/"+x:O/x for x in ["plan.json","report.json"]+[f"{i:02d}.json" for i in range(8)]}
files.update({"screen/"+x:D/x for x in ["run.log","run.stderr","exit-code"]})
for n in ("all8_revert40","all8_revert42","all8_revert40_42"):files["masks/"+n+".json"]=D/(n+".json")
assert all(p.is_file() and not p.is_symlink() and p.stat().st_size<=2*1024*1024 for p in files.values())
dest=pathlib.Path("/tmp/r23h-all8-ablation-prefill-v2.tgz")
with tarfile.open(dest,"w:gz") as t:
 for k,p in sorted(files.items()):t.add(p,arcname=k,recursive=False)
with tarfile.open(dest) as t:assert len(t.getmembers())==len(files)
print("ARCHIVE_SHA256="+hashlib.sha256(dest.read_bytes()).hexdigest(),"FILES="+str(len(files)),"BYTES="+str(dest.stat().st_size))
