import pathlib,tarfile,hashlib,json,gzip
R=pathlib.Path.home()/".cache/mt-pruning";Q=R/"r23-results"
B=Q/"r23h-all8-ablation-primary-v3";BR=R/"r23h-all8-ablation-primary-run-v3"
A=Q/"r23h-pareto-all10-propagation-v2";AR=R/"r23h-pareto-all10-run-v2"
S=R/"r23h-pareto-screen-v1";SD=Q/"r23h-pareto-prefill-v1"
S2=R/"r23h-all8-ablation-screen-run-v2";SD2=Q/"r23h-all8-ablation-prefill-v2"
U=R/"r23g-recovery-ssd-38ed4798";P=Q/"r23g-l38-propagation-v1"
ad=json.loads((B/"admission.json").read_text())
names=sorted(ad["candidates"]);indices=ad["indices"]
assert set(names)=={"all8_revert42","all8_revert40_42"} and len(indices)==10
assert (BR/"exit-code").read_text().strip()=="0" and (BR/"run.stderr").read_bytes()==b""
assert (AR/"exit-code").read_text().strip()=="0" and (AR/"run.stderr").read_bytes()==b""
assert (S2/"exit-code").read_text().strip()=="0" and (S2/"run.stderr").read_bytes()==b""
files={}
for x in ["admission.json","report.json"]:files["batch/"+x]=B/x
for x in ["run.log","run.stderr","exit-code"]:files["batch/"+x]=BR/x
for n in names:
 for i in indices:files[f"batch/{n}-{i:02d}.json"]=B/f"{n}-{i:02d}.json"
for x in ["admission.json","report.json"]:files["prior-all10/"+x]=A/x
for i in range(18):files[f"prior-all10/{i:02d}.json"]=A/f"{i:02d}.json"
for x in ["run.log","run.stderr","exit-code"]:files["prior-all10/"+x]=AR/x
for x in ["plan.json","report.json"]:files["screen/"+x]=SD2/x
for i in range(8):files[f"screen/{i:02d}.json"]=SD2/f"{i:02d}.json"
screen_plan=json.loads((SD2/"plan.json").read_text())
for n in screen_plan["hybrid_sources"]:files[f"screen/masks/{n}.json"]=S2/f"{n}.json"
for x in ["run.log","run.stderr","exit-code"]:files["screen/"+x]=S2/x
for x in ["plan.json","report.json"]:files["old-screen/"+x]=SD/x
for i in range(8):files[f"old-screen/{i:02d}.json"]=SD/f"{i:02d}.json"
for x in ["run.log","run.stderr","exit-code","original.json"]:files["old-screen/"+x]=S/x
for x in ["recovery-receipt.json","expanded.json","prop-plan.json"]:files["recovery/"+x]=U/x
for i in range(18):files[f"prior-r23g/{i:02d}.json"]=P/f"{i:02d}.json"
assert len([k for k in files if k.startswith("batch/") and k.endswith(".json") and "-" in k])==20
assert all(p.is_file() and not p.is_symlink() and p.stat().st_size<=2*1024*1024 for p in files.values())
dest=pathlib.Path("/tmp/r23h-all8-ablation-primary-v3.tgz")
with dest.open("wb") as raw:
 with gzip.GzipFile(filename="",fileobj=raw,mode="wb",compresslevel=9,mtime=0) as gz:
  with tarfile.open(fileobj=gz,mode="w",format=tarfile.USTAR_FORMAT) as t:
   for k,p in sorted(files.items()):
    info=t.gettarinfo(str(p),arcname=k);info.uid=0;info.gid=0;info.uname="";info.gname="";info.mtime=0;info.mode=0o600
    with p.open("rb") as f:t.addfile(info,f)
with tarfile.open(dest,"r:gz") as t:assert len(t.getmembers())==len(files)
print("ARCHIVE_SHA256="+hashlib.sha256(dest.read_bytes()).hexdigest(),"FILES="+str(len(files)),"BYTES="+str(dest.stat().st_size))
