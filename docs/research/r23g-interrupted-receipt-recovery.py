"""Independent R23G recovery witness after original stdout pipe loss.
Does not modify source logs, invent missing events, or authorize fresh."""
import argparse,hashlib,importlib.util,json,pathlib
P=pathlib.Path
GATE="51439071400d262fccd1d7e5e12c713c1a8ba44d686521d1fd9d6d57cf6c01b5"
HASHES="""70526bab0eedba34e1ce56468285524d3f454c1b1cc45c4a1f8be08e21275f02
8973b9f7c5a5dad3e284ca680b58b497920707c2c8e630a7d668fa3289b32fda
9ac01cb176506b40377444e64fe013266ad6ab4d270f7461696ed87d3007f695
6985569c7ab453b24fd768dfbcbc499638d446fb9dbc23e8fc624087eee401c9
750eaa5ca5ae9455108f856a0433b90f6a07c749fc63eaa41fb8615a02636ed4
73bcba0838b51ef57783d5b69e8973763d44c9a2fb4623735e4d5114d24b8a0a
f43aa1754cbb3389e34077fa428f611ac829353b4a683a83a2fc0218f8927406
a74e813d7344bfa04e7d86838b10d1f85def58e38113059d2997c0bdeb6a4049
929e8563918b6b491e42b06512de44184936749fdd15489b2aed5a59092c6d18
33a24d9e1e421a59ed730256df57580a67a93a08d309c589a8884a959f0e415f
c688af8b3cc53fdddd98dccde7d3bef5234a4783296ebe5a9d080af16eaf443e
c4a5444d7b1c1806b55f00414079d854a386ea5494193028241fce82da08195b
aac7bda5ebc9b4ca2be8cafb030f16dd1cef5a97e1daa4732b4b58ff7eabe5c4
998c375de655f7b132482348d6cdd6a20d62704bb385e792fdfbf38141f86c42
996e52a46b6649df6b43c65476b4dda0397ea743ff3bb38000a4c4717f40f756
4149a209d42b72192805b76103f85d5cd0c9e6f3ae3256baefbf9a617325ab6d
647b0834b31b41299927c92b2aadd9d15edc27167ba7de3328968cbf23d96c9a
130693d76535dcbe54365619cbecfa996b4e974c6cf270b34f7f911fabea5332""".split()
LOG="3dc86c44ab41c1e3fc1fa370c072864bf52b63615aedcc385a31963e3774cfa5"
ERR="6a27088c1bb0ccf445ba1e7570db1d2fe1e34aba4007cb64cf219a4133b2208e"
REPORT="38ed4798ab97bfbae5f7e182c9f7c708eb52f7bb00be092f90f83c8c233a4790"
ADFILE="b431c82ba984514bf0519593df518e40aa4e3da2793a18afee69b8b723ba9729"
def main():
    p=argparse.ArgumentParser();p.add_argument("--inputs",type=P,required=True);p.add_argument("--out",type=P,required=True);q=p.parse_args();d=q.inputs
    assert hashlib.sha256((d/"gate.py").read_bytes()).hexdigest()==GATE
    s=importlib.util.spec_from_file_location("gate",d/"gate.py");g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
    c=g.pin(d/"candidate.json",g.CAND);g.need(g.sha(g.canon(c["kept"]))==g.MASK,"mask")
    ep=g.pin(d/"expanded.json","a1047d1a7c5624ed0e3a868bd5d76eef4c754915",True);pp=g.pin(d/"prop-plan.json","cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde",True)
    rows=pp["rows"]+[r for r in ep["fit"] if r["direction"]=="zh-my" and r["id"] not in {z["id"] for z in pp["rows"]}]
    g.need(len(rows)==18 and len({r["line"] for r in rows})==16 and all(r in ep["fit"] for r in rows) and not ({r["line"] for r in rows}&{r["line"] for r in ep["holdout"]}),"fit only")
    root=d/"r23g-l38-propagation-v1";g.need(sorted(x.name for x in root.iterdir())==sorted(["admission.json","report.json"]+[f"{i:02d}.json" for i in range(18)]),"inventory")
    ad=g.pin(root/"admission.json",ADFILE);g.need(g.sha(g.canon(ad))==g.AD and ad["ids"]==[r["id"] for r in rows],"admission")
    log=g.read(d/"r23g-l38-propagation-v1.log");err=g.read(d/"r23g-l38-propagation-v1.stderr");g.need(g.sha(log)==LOG and g.sha(err)==ERR and b"BrokenPipeError" in err,"original immutable interrupted logs")
    ev=[g.strict(l) for l in log.splitlines() if l.startswith(b"{")]
    ready=[e for e in ev if e.get("status")=="TEACHER_READY"];g.need(len(ready)==1 and ready[0]["teacher"]==g.TEACHER and ready[0]["admission_sha256"]==g.AD,"teacher")
    captures=[e for e in ev if e.get("status")=="R23G_CAPTURED"];g.need(len(captures)==17 and all(e["index"]==i and e["id"]==rows[i]["id"] and e["receipt_sha256"]==HASHES[i] for i,e in enumerate(captures)) and not any(e.get("status")=="R23G_FIT_PROPAGATION_DIAGNOSTIC_NOT_QUALITY" for e in ev),"missing terminal events explicitly retained")
    prior=g.pin(d/"nonuniform-propagation-v1-exact-replay.json",g.PRIOR);g.need(prior["status"]=="NONUNIFORM_PROPAGATION_EXACT_REPLAY_NOT_QUALITY" and prior["samples"]==12,"prior replay")
    caps={}
    for i,r in enumerate(rows):
        x=g.pin(root/f"{i:02d}.json",HASHES[i]);g.check_capture(x,r,i)
        if i<12:
            old=g.pin(d/"nonuniform-propagation-v1"/f"{i:02d}.json",prior["receipt_sha256"][i]);g.need(old["id"]==r["id"] and old["original96"]==x["original96"],"baseline identity")
        caps[r["id"]]=x
    groups,viol=g.summarize(rows,caps);a=g.pin(root/"report.json",REPORT)
    g.need(a["status"]=="R23G_FIT_PROPAGATION_DIAGNOSTIC_NOT_QUALITY" and a["samples"]==18 and a["aligned_lines"]==16 and a["groups"]==groups and a["primary_diagnostic_violations"]==viol==["zh-my/prefill/kl"] and a["primary_diagnostic_pass"] is False,"recovered report reconciliation")
    out=dict(status="R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH",original_job_status="INTERRUPTED_STDOUT_NOT_NORMAL_SUCCESS",logged_captures=17,disk_captures=18,missing_terminal_events=["capture17","report","original_replay"],receipt_sha256=HASHES,original_log_sha256=LOG,original_stderr_sha256=ERR,report_sha256=REPORT,admission_sha256=g.AD,candidate_sha256=g.CAND,primary_violations=viol,groups=groups,limits="Recovery from pinned actual disk bytes; no original successful job, no synthetic terminal events, no fresh, physical build or product qualification.")
    raw=g.canon(out)+b"\n"
    with q.out.open("xb") as f:f.write(raw)
    print(json.dumps(dict(status=out["status"],sha256=g.sha(raw),samples=18,primary_violations=viol)))
if __name__=="__main__":main()
