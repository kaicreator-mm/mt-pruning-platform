"""Freeze actual R23D nonuniform allocation as a mask-only diagnostic artifact."""
import argparse,hashlib,json,math,os,pathlib,collections
P=pathlib.Path
ROOT=P.home()/".cache/mt-pruning/r23-results"
PLAN="a9da78015d010484c16d6c8af9000a15d46946db00677c19f4fac4a0d562e41e"
ALLOCATION="04c2238ac5bb47609bc0ed452f80690b4037c531c9f3dc33b93394c89bf6c6bc"
REPORT="8e35997647591872f4c83f9dc7bfcae63052c604383bc1b2cb15bc8650eee63a"
BASE="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
def need(x,m):
    if not x:raise ValueError(m)
def sha(x):return hashlib.sha256(x).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def read(p,h):
    need(p.is_file() and not p.is_symlink(),"regular file")
    b=p.read_bytes();need(sha(b)==h,"exact source bytes");return json.loads(b)
def materialize(allocated,masks,layers,n=128,cap=4512):
    need(set(allocated)==set(layers) and all(type(k) is int and k in (64,80,96,112,128) for k in allocated.values()) and sum(allocated.values())==cap,"exact budget")
    need(set(masks)>={"original96","k64","k80","k112","k128"},"mask inventory")
    out={}
    for l in layers:
        k=allocated[l];name="original96" if k==96 else f"k{k}";v=masks[name][l]
        need(v==sorted(set(v)) and len(v)==k and all(type(i) is int and 0<=i<n for i in v),"layer mask")
        out[l]=v
    return out
def selftest():
    layers=["1","2"];ms={name:{l:list(range(k)) for l in layers} for k,name in ((64,"k64"),(80,"k80"),(96,"original96"),(112,"k112"),(128,"k128"))}
    q=materialize({"1":80,"2":112},ms,layers,cap=192)
    need(len(q["1"])==80 and len(q["2"])==112,"mask oracle")
    for bad in ({"1":80,"2":96},{"1":80,"2":112,"3":96}):
        try:materialize(bad,ms,layers,cap=192)
        except ValueError:pass
        else:raise AssertionError("invalid allocation")
    print(json.dumps(dict(status="PASS_SYNTHETIC_NONUNIFORM_MASK_FREEZE",negative=2,real_mask="NOT_RUN")))
def main():
    a=argparse.ArgumentParser();a.add_argument("--out",type=P);a.add_argument("--selftest",action="store_true");q=a.parse_args()
    if q.selftest:selftest();return
    need(q.out,"output path")
    p=read(ROOT/"fit-interaction-budget-plan-v1.json",PLAN);a=read(ROOT/"interaction-budget-v1-allocation.json",ALLOCATION);r=read(ROOT/"interaction-budget-v1/report.json",REPORT)
    need(a["report_sha256"]==REPORT and a["status"]=="FIT_ALLOCATION_FOR_INDEPENDENT_DIAGNOSTIC_ONLY" and a["budget_used"]==a["budget_cap"]==4512 and a["overall_guard"] and not a["violations"] and r["status"]=="FIT_ONLY_INTERACTION_BUDGET_DIAGNOSTIC_NOT_ADMISSION","diagnostic binding")
    masks={k:v["kept"] for k,v in p["masks"].items()}
    need(all(sha(canon(masks[k]))==v["sha256"] for k,v in p["masks"].items()) and sha(canon(masks["original96"]))==BASE,"mask digests")
    kept=materialize(a["allocated"],masks,[str(i) for i in range(1,48)])
    out=dict(status="NONUNIFORM96AVG_MASK_FOR_PROPAGATION_DIAGNOSTIC_ONLY",allocation_sha256=ALLOCATION,report_sha256=REPORT,plan_sha256=PLAN,original96_mask_sha256=BASE,kept=kept,mask_sha256=sha(canon(kept)),expert_instances=sum(map(len,kept.values())),layer_budget_counts=dict(collections.Counter(map(len,kept.values()))),limits="No physical storage or runtime deployment qualification; fit-only; compare actual propagated states before any new candidate decision.")
    b=canon(out)+b"\n"
    with q.out.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    print(json.dumps({k:v for k,v in dict(out,receipt_sha256=sha(b)).items() if k!="kept"}),flush=True)
if __name__=="__main__":main()
