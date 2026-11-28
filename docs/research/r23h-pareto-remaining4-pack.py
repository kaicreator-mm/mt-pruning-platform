import pathlib,tarfile,hashlib
R=pathlib.Path.home()/".cache/mt-pruning";Q=R/"r23-results"
B=Q/"r23h-pareto-remaining4-primary-v1";BR=R/"r23h-pareto-remaining4-run-v1"
A=Q/"r23h-pareto-all10-propagation-v2";AR=R/"r23h-pareto-all10-run-v2"
S=R/"r23h-pareto-screen-v1";SD=Q/"r23h-pareto-prefill-v1"
U=R/"r23g-recovery-ssd-38ed4798";P=Q/"r23g-l38-propagation-v1"
files={}
for x in ["admission.json","report.json"]:files["batch/"+x]=B/x
for x in ["run.log","run.stderr","exit-code"]:files["batch/"+x]=BR/x
for n in ("all12","all14","primary4_10","primary4_14"):
 for i in range(18):
  p=B/f"{n}-{i:02d}.json"
  if p.exists():files[f"batch/{n}-{i:02d}.json"]=p
for x in ["admission.json","report.json"]:files["prior-all10/"+x]=A/x
for i in range(18):files[f"prior-all10/{i:02d}.json"]=A/f"{i:02d}.json"
for x in ["run.log","run.stderr","exit-code"]:files["prior-all10/"+x]=AR/x
for x in ["plan.json","report.json"]:files["screen/"+x]=SD/x
for i in range(8):files[f"screen/{i:02d}.json"]=SD/f"{i:02d}.json"
for x in ["run.log","run.stderr","exit-code","original.json"]:files["screen/"+x]=S/x
for x in ["recovery-receipt.json","expanded.json","prop-plan.json"]:files["recovery/"+x]=U/x
for i in range(18):files[f"prior-r23g/{i:02d}.json"]=P/f"{i:02d}.json"
assert len([k for k in files if k.startswith("batch/") and k.endswith(".json") and "-" in k])==40
assert all(p.is_file() and not p.is_symlink() and p.stat().st_size<=2*1024*1024 for p in files.values())
dest=pathlib.Path("/tmp/r23h-remaining4-primary-v1.tgz")
with tarfile.open(dest,"w:gz") as t:
 for k,p in sorted(files.items()):t.add(p,arcname=k,recursive=False)
with tarfile.open(dest) as t:assert len(t.getmembers())==len(files)
print("ARCHIVE_SHA256="+hashlib.sha256(dest.read_bytes()).hexdigest(),"FILES="+str(len(files)),"BYTES="+str(dest.stat().st_size))
