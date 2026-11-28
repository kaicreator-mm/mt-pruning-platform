"""Finite V100 cleanup. Move only exact whitelisted retired model artifacts after
source + durable USB byte verification; retain all evidence, teacher and live baseline.
No follow-symlinks, no overwrite, per-file recovery receipts; detached local log."""
import hashlib,json,os,pathlib,shutil,stat,subprocess,time,fcntl
P=pathlib.Path;R=P.home()/".cache/mt-pruning";M=P("/mnt/mt-storage");DEST=M/"mt-pruning/models/v100-retired-20261127";E=R/"retired-build-artifacts/v100-usb-20261127"
def need(x,m):
    if not x:raise RuntimeError(m)
def event(status,**kw):print(json.dumps(dict(status=status,**kw),sort_keys=True),flush=True)
def cmd(*a):return subprocess.check_output(a,text=True).strip()
def mount():
    need(cmd("findmnt","-rn","-M",str(M),"-o","UUID")=="6039-CF12" and cmd("findmnt","-rn","-M",str(M),"-o","FSTYPE")=="exfat" and os.access(M,os.W_OK),"USB identity/mount")
    need(DEST.resolve()==DEST and M.stat().st_dev!=R.stat().st_dev,"separate storage")
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def identity(p):
    s=p.lstat();need(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and p.resolve()==p,"ordinary single-link file "+str(p));return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
def inactive(src):
    s=src.stat();matches=[]
    for p in P("/proc").iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid() or int(p.name)==os.getpid():continue
            for fd in (p/"fd").iterdir():
                try:
                    t=fd.stat()
                    if (t.st_dev,t.st_ino)==(s.st_dev,s.st_ino):matches.append(str(fd))
                except FileNotFoundError:pass
            try:
                for line in (p/"maps").read_text(errors="replace").splitlines():
                    if str(src) in line:matches.append(str(p/"maps"));break
            except FileNotFoundError:pass
        except FileNotFoundError:pass
        except PermissionError:
            # Only the observed OS session infrastructure is exempt; any other
            # inaccessible process remains fail-closed. No model process exemption.
            try:
                comm=(p/"comm").read_text().strip()
                argv=(p/"cmdline").read_bytes().replace(bytes([0]),b" ").decode(errors="replace").strip()
            except FileNotFoundError:continue
            trusted_session=(comm=="systemd" and argv=="/usr/lib/systemd/systemd --user") or (comm=="(sd-pam)" and argv=="(sd-pam)") or (comm=="sshd-session" and argv.startswith("sshd-session: "+os.environ["USER"]+"@") and (argv.endswith("@notty") or __import__("re").fullmatch(r"sshd-session: "+__import__("re").escape(os.environ["USER"])+r"@pts/[0-9]+",argv)))
            need(trusted_session,"cannot inspect non-session process "+str(p)+" "+comm)

    need(not matches,"active file "+str(src)+" "+str(matches))
def publish(p,x):
    raw=(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    if p.exists():need(p.is_file() and not p.is_symlink() and p.read_bytes()==raw,"receipt conflict")
    else:
        with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
def syncusb():subprocess.run(["sync","-f",str(M)],check=True)
def move(src,rel,expected,n):
    mount();dst=DEST/rel;dst.parent.mkdir(parents=True,exist_ok=True)
    rec=E/(str(rel).replace("/","__")+".json")
    if not src.exists():
        need(rec.is_file(),"missing source without receipt")
        z=json.loads(rec.read_bytes());need(z["sha256"]==expected and z["bytes"]==n and z["source"]==str(src) and z["archive"]==str(dst) and identity(dst)[2]==n and digest(dst)==expected,"resume archived receipt")
        event("ALREADY_ARCHIVED",source=str(src),bytes=n);return
    before=identity(src);need(before[2]==n,"source size");inactive(src)
    need(digest(src)==expected and identity(src)==before,"source pinned hash")
    if dst.exists():need(identity(dst)[2]==n and digest(dst)==expected,"existing destination mismatch; preserve both")
    else:
        need(shutil.disk_usage(M).free>=n+(10<<30),"USB reserve")
        h=hashlib.sha256()
        with src.open("rb") as a,dst.open("xb") as b:
            for z in iter(lambda:a.read(8<<20),b""):b.write(z);h.update(z)
            b.flush();os.fsync(b.fileno())
        need(h.hexdigest()==expected and identity(src)==before,"copy stream")
    syncusb();mount();need(identity(dst)[2]==n and digest(dst)==expected and identity(src)==before,"persisted archive integrity")
    receipt=dict(status="VERIFIED_USB_ARCHIVE_BEFORE_SOURCE_REMOVAL",source=str(src),archive=str(dst),bytes=n,sha256=expected,usb_uuid="6039-CF12")
    publish(rec,receipt);publish(dst.parent/(dst.name+".archive-receipt.json"),receipt);syncusb()
    # Keep SSD receipt durable before unlinking, recheck target and active references.
    fd=os.open(E,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    mount();inactive(src);need(identity(src)==before,"source changed before unlink")
    src.unlink()
    fd=os.open(src.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    event("ARCHIVED_AND_REMOVED_SSD_SOURCE",source=str(src),archive=str(dst),bytes=n,sha256=expected,ssd_free=shutil.disk_usage(R).free,usb_free=shutil.disk_usage(M).free)
def main():
    mount()
    with (R/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not cmd("nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"),"GPU active")
        E.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True)
        need(E.is_dir() and not E.is_symlink() and not list(E.iterdir()),"empty prior receipt directory")
        hf=R/"pruned-r19/k96";mf=json.loads((hf/"prune_manifest.json").read_bytes())
        need(mf["num_experts"]==96 and mf["source_num_experts"]==128 and mf["source"]["digest"]=="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94","HF manifest identity")
        selection=R/"run-d01b-r19/selection.json";need(mf["kept"]==json.loads(selection.read_bytes())["candidates"]["96"]["kept"],"HF selection")
        need(digest(hf/"config.json")==mf["config_sha256"] and digest(hf/"model.safetensors.index.json")==mf["index_sha256"],"HF metadata")
        files=mf["files"];need(len(files)==10 and {x["name"] for x in files}=={f"model-{i:05d}-of-00010.safetensors" for i in range(1,11)},"shard inventory")
        guarded=R/"r22-candidates/guarded96-f16";g=json.loads((guarded/"receipt.json").read_bytes())
        need(g["artifact"]["sha256"]=="4b9df0b0382e22c7cc613fd06b50911b3aae22996073184affcbf4128516f41f" and g["artifact"]["bytes"]==45953803936,"guarded manifest")
        # Archive metadata copies; leave original metadata in place.
        for root,sub in ((hf,"original96-hf"),(guarded,"guarded96-f16")):
            d=DEST/sub;d.mkdir(exist_ok=True)
            for p in root.iterdir():
                if p.suffix in (".safetensors",".gguf"):continue
                identity(p);q=d/p.name
                if not q.exists():
                    with p.open("rb") as a,q.open("xb") as b:shutil.copyfileobj(a,b);b.flush();os.fsync(b.fileno())
                need(digest(p)==digest(q),"metadata copy")
        syncusb()
        before=shutil.disk_usage(R).free
        event("CLEANUP_ADMITTED",ssd_free=before,usb_free=shutil.disk_usage(M).free,source_teacher="RETAIN",original96_gguf="RETAIN",planned_files=12,planned_bytes=sum(x["bytes"] for x in files)+31747952288+45953803936)
        # k64 first to immediately unlock enough internal space, then HF and guarded.
        move(R/"gguf-r20-r1/k64-f16.gguf",P("k64-f16/model.gguf"),"c77b03276ecc1e4cb70d32f249996ed6c3f49a9e78f087bd734e7d187c20b444",31747952288)
        for x in files:move(hf/x["name"],P("original96-hf")/x["name"],x["sha256"],x["bytes"])
        move(guarded/"model.gguf",P("guarded96-f16/model.gguf"),g["artifact"]["sha256"],g["artifact"]["bytes"])
        mount();need(len(list(E.glob("*.json")))==12,"receipt count")
        out=dict(status="CLEANUP_COMPLETE_VERIFIED_ARCHIVES",files=12,bytes=sum(x["bytes"] for x in files)+31747952288+45953803936,ssd_free_before=before,ssd_free_after=shutil.disk_usage(R).free,usb_free_after=shutil.disk_usage(M).free,receipt_files=sorted(p.name for p in E.glob("*.json")),source_teacher="RETAINED",original96_baselines="RETAINED")
        publish(E/"complete.json",out);publish(DEST/"complete.json",out);syncusb();event(**out)
if __name__=="__main__":main()
