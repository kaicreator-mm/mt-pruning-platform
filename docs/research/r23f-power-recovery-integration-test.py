"""Isolated fixture, real plan and code, synthetic profiles, not actual disk recovery."""
import pathlib,sys,importlib.util,json,hashlib,tempfile,contextlib,io
P=pathlib.Path;d=P(sys.argv[1])
def load(n):
 s=importlib.util.spec_from_file_location(n,d/(n+".py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=load("recover");s=load("stable")
def invoke(m,args):
 old=sys.argv
 try:
  sys.argv=[str(d/(m.__name__+".py"))]+list(map(str,args))
  with contextlib.redirect_stdout(io.StringIO()) as out:m.main()
  return out.getvalue()
 finally:sys.argv=old
def reject(fn):
 try:fn()
 except (ValueError,OSError):return
 raise AssertionError("invalid fixture accepted")
fit=json.loads((d/"expanded.json").read_bytes())["fit"]
w=json.loads((d/"witness-original.json").read_bytes())
with tempfile.TemporaryDirectory(prefix="r23f-fixture-") as tmp:
 root=P(tmp);r.ROOT=s.ROOT=root
 adroot=root/"r23-admissions";adroot.mkdir()
 old=root/"r23-results/reap-full24";old.mkdir(parents=True)
 new=root/"r23-results/expanded-fit-profile-v1";new.mkdir()
 ad=dict(status="R23F_EXPANDED_FIT_PROFILE_ADMITTED",plan_blob=r.PLAN,teacher=r.TEACHER,engine_head=r.ENGINE,threads=8,first=12,stop=48,ids=[q["id"] for q in fit[12:]],prior_receipts=[])
 raw=r.canon(ad);(adroot/"expanded-fit-12-48.json").write_bytes(raw)
 r.OLD_AD=s.OLD_ADMISSION=r.sha(raw)
 def profile(i,adsha):
  q=fit[i];z=[dict(step=j,phase="prefill" if j==0 else "decode",sha256="a"*64,next=q["teacher_prefix_tokens"][j] if j<8 else 0,**({"input_token":q["teacher_prefix_tokens"][j-1]} if j else {})) for j in range(9)]
  def vec(t):
   ids=set(range(8))
   return dict(tokens=t,count=[t if k in ids else 0 for k in range(128)],weight=[t/8 if k in ids else 0 for k in range(128)],norm=[t if k in ids else 0 for k in range(128)],weighted_norm=[t/8 if k in ids else 0 for k in range(128)],energy=[t/64 if k in ids else 0 for k in range(128)])
  pre=vec(q["prompt_tokens"]);dec=vec(8)
  x=dict(index=i,id=q["id"],direction=q["direction"],subset="fit",teacher=r.TEACHER,engine_head=r.ENGINE,threads=8,admission_sha256=adsha,hook_calls=423,num_experts=128,top_k=8,teacher_logits=z,scale=1.,plan_blob=r.PLAN,profile={f"{l}/{phase}":v for l in range(1,48) for phase,v in (("prefill",pre),("decode",dec))},matched_r23a_logits=True if i<12 else None,cross_process_r23a_differences=[] if i<12 else None,same_process_observer_equivalence=True,plain_logits=z,restored_logits=z,historical_argmax_mismatches=[],historical_greedy_match=True)
  p=(old if i<12 else new)/f"{i:02d}.json";b=r.canon(x)+b"\n";p.write_bytes(b)
  return dict(index=i,id=q["id"],receipt_sha256=r.sha(b),admission_sha256=adsha)
 rows=[profile(i,w["rows"][i]["admission_sha256"] if i<12 else r.OLD_AD) for i in range(44)]
 w["rows"]=rows;wb=r.canon(w)+b"\n";(d/"witness.json").write_bytes(wb);r.WITNESS=s.WITNESS=r.blob(wb)
 args=["--inputs",d]
 assert "DISK_PREFIX44_VERIFIED" in invoke(r,args)
 # Simulate an orphan in the isolated fixture; production recovery refuses it.
 orphan=new/"44.json";orphan.write_bytes(b"unwitnessed");reject(lambda:invoke(r,args));assert orphan.read_bytes()==b"unwitnessed";orphan.unlink()
 # Simulate changed prefix without losing fixture data.
 q=new/"43.json";before=q.read_bytes();q.write_bytes(before+b"x");reject(lambda:invoke(r,args));q.write_bytes(before)
 out=root/"resume.json";assert "RESUME44_48_ADMISSION_READY_NOT_EXECUTED" in invoke(r,args+["--stage","resume-admission","--out",out,"--comment",1,"--head","a"*40])
 raw=out.read_bytes();h=r.sha(raw);a=json.loads(raw);assert a["first"]==44 and a["stop"]==48 and a["prior_receipts"]==[x["receipt_sha256"] for x in rows[12:]]
 (adroot/"expanded-fit-44-48.json").write_bytes(raw)
 more=[profile(i,h) for i in range(44,48)]
 logs=[dict(status="TEACHER_READY",teacher=r.TEACHER,admission_sha256=h)]+[dict(status="EXPANDED_FIT_PROFILE_CAPTURED",id=x["id"],index=x["index"],completed=x["index"]+1,receipt_sha256=x["receipt_sha256"],hook_calls=423,same_process_triple=True) for x in more]+[dict(status="EXPANDED_FIT_PROFILE_BATCH_COMPLETE_NOT_SELECTION",completed=48,total=48,holdout_capture="NOT_RUN")]
 log=root/"resume.log";log.write_bytes(b"".join(r.canon(x)+b"\n" for x in logs));err=root/"resume.stderr";err.write_bytes(b"")
 manifest=root/"manifest.json";assert "EXPANDED_FIT48_RECOVERY_VERIFIED_NOT_SELECTION" in invoke(r,args+["--stage","finalize","--out",manifest,"--resume-log",log,"--resume-stderr",err,"--resume-admission-sha",h])
 mf=manifest.read_bytes();assert json.loads(mf)["rows"]==rows+more
 out1=root/"stable1.json";out2=root/"stable2.json"
 args=["--inputs",d,"--manifest",manifest,"--manifest-sha",r.sha(mf)]
 invoke(s,args+["--out",out1]);invoke(s,args+["--out",out2])
 assert out1.read_bytes()==out2.read_bytes() and json.loads(out1.read_bytes())["profile_receipt_sha256"]==[x["receipt_sha256"] for x in rows+more]
 print(json.dumps(dict(status="PASS_SYNTHETIC_CROSS_ADMISSION_FULL_FLOW",profiles=48,old_prefix=44,resumed=4,layer_phase_records=48*94,orphan_rejected=True,tamper_rejected=True,stable_byte_replay=True,actual_v100_disk="NOT_CHECKED",real_mask="NOT_RUN")))
