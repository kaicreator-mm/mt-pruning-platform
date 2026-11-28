"""R23E exact receipt and report replay after real fit-only propagation.
No teacher load, no model inference, no selection or acceptance.
"""
import argparse,hashlib,importlib.util,json,pathlib,math,tempfile
P=pathlib.Path
BATCH="87b12c30ebf801d2e4571b6f0f8348edbb61ba13"
PLAN="cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
ENGINE="3212303e1e02994a5124db8c85195b91325c7a7c"
MASK="2339adfdbb8a438ecfa046384cced8f652fd1a523e722a66bf4480002a0500c9"
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def read(p):
    need(p.is_file() and not p.is_symlink(),"regular file "+str(p));return p.read_bytes()
def strict(raw):
    def pairs(xs):
        d={}
        for k,v in xs:need(k not in d,"duplicate key");d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def replay(batch,rows,folder,log,stderr,ad_sha,layers):
    need(stderr==b"","stderr");need(sorted(p.name for p in folder.iterdir())==sorted(["admission.json","report.json"]+[f"{i:02d}.json" for i in range(12)]),"inventory")
    adraw=read(folder/"admission.json");ad=strict(adraw);need(adraw==canon(ad)+b"\n" and sha(canon(ad))==ad_sha and ad["status"]=="R23E_NONUNIFORM_PROPAGATION_DIAGNOSTIC_ADMITTED" and ad["plan_blob"]==PLAN and ad["teacher"]==TEACHER and ad["engine_head"]==ENGINE and ad["mask_sha256"]==MASK and ad["threads"]==8 and ad["ids"]==[r["id"] for r in rows],"admission")
    events=[strict(l) for l in log.splitlines(keepends=True) if l.startswith(b"{") and l.endswith(b"\n")]
    ready=[x for x in events if x.get("status")=="TEACHER_READY"]
    captured=[x for x in events if x.get("status")=="NONUNIFORM_PROPAGATION_CAPTURED"]
    done=[x for x in events if x.get("status")=="NONUNIFORM_PROPAGATION_DIAGNOSTIC_COMPLETE_NOT_ADMISSION"]
    need(len(ready)==len(done)==1 and ready[0]["teacher"]==TEACHER and ready[0]["admission_sha256"]==ad_sha and len(captured)==len(rows)==12 and done[0]["samples"]==12,"complete actual job witness")
    caps={};hashes=[]
    for i,(r,w) in enumerate(zip(rows,captured)):
        raw=read(folder/f"{i:02d}.json");x=strict(raw);h=sha(raw)
        need(raw==canon(x)+b"\n" and w["index"]==i and w["id"]==r["id"] and w["receipt_sha256"]==h and x["index"]==i and x["teacher"]==TEACHER and x["engine_head"]==ENGINE and x["plan_blob"]==PLAN and x["mask_sha256"]==MASK and x["admission_sha256"]==ad_sha,"capture binding")
        batch.checked(x,r,layers);caps[r["id"]]=x;hashes.append(h)
    raw=read(folder/"report.json");out=batch.report(rows,caps,layers)
    need(raw==canon(out)+b"\n" and sha(raw)==done[0]["receipt_sha256"],"exact report replay")
    return dict(status="NONUNIFORM_PROPAGATION_EXACT_REPLAY_NOT_QUALITY",samples=12,receipt_sha256=hashes,admission_sha256=ad_sha,report_sha256=sha(raw),log_sha256=sha(log),stderr_sha256=sha(stderr),layer_phase_step_records_per_mask=out["layer_phase_step_records_per_mask"])
def selftest(batch):
    rows=[dict(id=str(i),line=i,direction=batch.DIRS[i//2],teacher_prefix_tokens=list(range(8))) for i in range(12)]
    with tempfile.TemporaryDirectory() as d:
        p=P(d);ad=dict(status="R23E_NONUNIFORM_PROPAGATION_DIAGNOSTIC_ADMITTED",plan_blob=PLAN,teacher=TEACHER,engine_head=ENGINE,mask_sha256=MASK,threads=8,ids=[r["id"] for r in rows]);ah=sha(canon(ad));(p/"admission.json").write_bytes(canon(ad)+b"\n")
        events=[dict(status="TEACHER_READY",teacher=TEACHER,admission_sha256=ah)];caps={}
        for i,r in enumerate(rows):
            x=dict(status="PROPAGATED_FIXED_TRAJECTORY_DIAGNOSTIC_NOT_QUALITY",teacher_restored=True,id=r["id"],line=r["line"],direction=r["direction"],consumed_prefix=r["teacher_prefix_tokens"],index=i,teacher=TEACHER,engine_head=ENGINE,plan_blob=PLAN,mask_sha256=MASK,admission_sha256=ah)
            for n in batch.NAMES:
                v=.1 if n=="original96" else .05
                h=[dict(step=j,sha256="a"*64,next=j) for j in range(9)]
                x[n]=dict(teacher_hashes=h,masked_hashes=h,logits=[dict(step=j,nmse=v,teacher_to_masked_kl=v,teacher_argmax=j,masked_argmax=j) for j in range(9)],layer=[dict(layer=1,phase="prefill" if j==0 else "decode",step=j,tokens=2 if j==0 else 1,hidden_input_nmse=v,moe_output_nmse=v,mean_overlap=8.,exact_sets=2 if j==0 else 1) for j in range(9)])
            caps[r["id"]]=x;raw=canon(x)+b"\n";(p/f"{i:02d}.json").write_bytes(raw);events.append(dict(status="NONUNIFORM_PROPAGATION_CAPTURED",index=i,id=r["id"],receipt_sha256=sha(raw)))
        report=canon(batch.report(rows,caps,[1]))+b"\n";(p/"report.json").write_bytes(report);events.append(dict(status="NONUNIFORM_PROPAGATION_DIAGNOSTIC_COMPLETE_NOT_ADMISSION",samples=12,receipt_sha256=sha(report)))
        def log(a):return b"".join(canon(x)+b"\n" for x in a)
        need(replay(batch,rows,p,log(events),b"",ah,[1])["samples"]==12,"positive")
        def reject(f):
            try:f()
            except ValueError:return
            raise AssertionError("bad evidence accepted")
        reject(lambda:replay(batch,rows,p,log(events[:-2]+events[-1:]),b"",ah,[1]))
        reject(lambda:replay(batch,rows,p,log(events),b"error",ah,[1]))
        q=p/"03.json";raw=q.read_bytes();q.write_bytes(raw+b" ");reject(lambda:replay(batch,rows,p,log(events),b"",ah,[1]));q.write_bytes(raw)
        q=p/"report.json";q.write_bytes(report+b" ");reject(lambda:replay(batch,rows,p,log(events),b"",ah,[1]))
    print(json.dumps(dict(status="PASS_SYNTHETIC_PROPAGATION_REPLAY",negative=4,actual_model="NOT_RUN")))
def main():
    a=argparse.ArgumentParser();a.add_argument("--batch",type=P,required=True);a.add_argument("--selftest",action="store_true");a.add_argument("--plan",type=P);a.add_argument("--folder",type=P);a.add_argument("--log",type=P);a.add_argument("--stderr",type=P);a.add_argument("--admission-sha");q=a.parse_args()
    need(blob(read(q.batch))==BATCH,"batch pin");s=importlib.util.spec_from_file_location("batch",q.batch);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    if q.selftest:selftest(m);return
    need(q.plan and q.folder and q.log and q.stderr and q.admission_sha,"explicit evidence");raw=read(q.plan);need(blob(raw)==PLAN,"plan pin")
    print(json.dumps(replay(m,strict(raw)["rows"],q.folder,read(q.log),read(q.stderr),q.admission_sha,list(range(1,48)))))
if __name__=="__main__":main()
