"""R23G fit-only robust local feasibility diagnostic.
Minimize swapped experts relative to original96 under ALL 30 aligned-line
deletion folds + full fit guards. No mask publication/admission, no fresh data,
no propagated inference. Solver numerical results independently recomputed."""
import argparse,hashlib,importlib.util,json,math,pathlib,sys
P=pathlib.Path
ATTR="71ee7ebd17c9efdc894be76ddcb2433b6a98c184873c41ea4f418320713da20f"
CAND="e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
SENS="0e071705ea763d45e0b7034a20647cb16305b48ea737386a69d8014f903ad381"
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def pin(p,h):
    need(p.is_file() and not p.is_symlink(),"regular file");b=p.read_bytes();need(sha(b)==h,"hash "+str(p));return b
def solve(A,upper,anchor,*,weights=None,seconds=45):
    import numpy as np
    from scipy.optimize import milp,Bounds,LinearConstraint
    A=np.asarray(A,dtype=float);u=np.asarray(upper,dtype=float);a=np.asarray(anchor,dtype=int)
    need(A.ndim==2 and A.shape[0]==len(u) and A.shape[1]==len(a) and np.isfinite(A).all() and np.isfinite(u).all() and np.isin(a,[0,1]).all(),"matrix")
    # Scale each inequality by its own max magnitude. Independent unscaled check below.
    scale=np.maximum(np.maximum(np.max(np.abs(A),axis=1),np.abs(u)),1e-12)
    w=np.ones(len(a),dtype=int) if weights is None else np.asarray(weights,dtype=int)
    need(w.shape==a.shape and np.all(w>0),"weights")
    c=w*(1.-2.*a)
    r=milp(c,integrality=np.ones(len(a)),bounds=Bounds(np.zeros(len(a)),np.ones(len(a))),constraints=LinearConstraint(A/scale[:,None],-np.inf,u/scale),options={"time_limit":seconds,"mip_rel_gap":0.})
    out=dict(solver_status=int(r.status),solver_message=str(r.message),mip_gap=None if getattr(r,"mip_gap",None) is None else float(r.mip_gap),mip_dual_bound=None if getattr(r,"mip_dual_bound",None) is None else float(r.mip_dual_bound))
    if r.x is None:return dict(out,feasible=False,proved_minimum=False)
    z=np.rint(r.x).astype(int);need(np.max(np.abs(z-r.x))<1e-6 and np.isin(z,[0,1]).all(),"integrality")
    violations=A@z-u
    # Match source gate epsilon; numerical solver success alone is not enough.
    feasible=bool(np.all(violations<=1e-12))
    distance=int(w@(z!=a));lb=out["mip_dual_bound"]
    proof=bool(feasible and r.status==0 and lb is not None and math.ceil(lb+int(w@a)-1e-6)==distance)
    return dict(out,feasible=feasible,proved_minimum=proof,distance=distance,choices=[int(x) for x in z],max_unscaled_violation=float(np.max(violations)))
def selftest():
    # Anchor 11 fails x0+x1<=1, one edit is globally minimal.
    a=solve([[1.,1.]], [1.], [1,1])
    need(a["feasible"] and a["proved_minimum"] and a["distance"]==1,"one edit oracle")
    b=solve([[1.,0.],[-1.,0.]], [0.,-1.], [1,1])
    need(not b["feasible"],"infeasible oracle")
    print(json.dumps(dict(status="PASS_SYNTHETIC_ROBUST_LOCAL_MILP",positive=1,negative=1)))
def main():
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true");p.add_argument("--inputs",type=P);p.add_argument("--out",type=P);q=p.parse_args()
    if q.selftest:selftest();return
    need(q.inputs and q.out,"inputs");d=q.inputs
    import numpy as np,scipy
    pin(d/"analyze-v2.py",ATTR);sp=importlib.util.spec_from_file_location("attr",d/"analyze-v2.py");a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
    c=a.strict(pin(d/"r23g-l38-reverted.json",CAND));sens=a.strict(pin(d/"r23g-sensitivity.json",SENS))
    plan=a.strict(a.pinned(d/"expanded.json",a.PLAN,True));rows=plan["fit"];need(len(rows)==48 and len({r["line"] for r in rows})==30 and not ({r["line"] for r in rows}&{r["line"] for r in plan["holdout"]}),"fit only")
    base=a.strict(a.pinned(d/"original.json",a.ORIG,True))["candidates"]["96"]["kept"];stable=a.strict(a.pinned(d/"candidate.json",a.CAND))["kept"];h=c["kept"];need(a.sha(a.canon(h))==c["mask_sha256"] and h["38"]==base["38"] and [k for k in h if h[k]!=stable[k]]==["38"],"frozen anchor")
    changed=[l for l in range(1,48) if base[str(l)]!=stable[str(l)]];need(len(changed)==42 and 38 in changed,"changed inventory")
    anchor=[int(h[str(l)]==stable[str(l)]) for l in changed];need(anchor.count(0)==1,"anchor")
    proxy=a.pure_proxy(a.pinned(d/"proxy.py",a.PROXY,True));masks={"original96":base,"stable96":stable};layers=list(range(1,48));caps={};loss={}
    for j in range(2):
        ev=[a.strict(x) for x in a.pinned(d/f"fit{j}.log",a.LOGS[j]).splitlines() if x.startswith(b"{")];need(a.read(d/f"fit{j}.stderr")==b"","stderr")
        ev=[x for x in ev if x.get("status")=="STABLE96_PROXY_CAPTURED"];need(len(ev)==24,"batch")
        for i,e in enumerate(ev,j*24):
            r=rows[i];need(e["index"]==i and e["id"]==r["id"],"order")
            x=a.strict(a.pinned(d/"fit"/f"{i:02d}.json",e["receipt_sha256"]));need(x["admission_sha256"]==a.AD[j] and x["teacher_digest"]==a.TEACHER and x["engine_head"]==a.ENGINE and x["candidate_receipt_sha256"]==a.CAND and x["index"]==i,"provenance")
            seen=proxy.checked_capture(x,r,masks,layers);caps[r["id"]]=x
            loss[r["id"]]={n:{p:[math.fsum(seen[l,p,k]["losses"][n]["combined_error"] for k in steps)/len(steps)/47 for l in layers] for p,steps in (("prefill",[0]),("decode",[1,8]))} for n in ("original96","stable96")}
    f=a.strict(a.pinned(d/"fit"/"decision.json",a.DEC));need(f==proxy.decision(rows,caps,masks,layers,"fit"),"exact decision")
    A=[];u=[];blocks=[]
    for omit in [None]+sorted({r["line"] for r in rows}):
        rs=[r for r in rows if r["line"]!=omit];b=[];delta=[]
        for g in a.GROUPS:
            di,ph=g.split("/");rr=[r for r in rs if r["direction"]==di];need(rr,"group")
            b.append(math.fsum(math.fsum(loss[r["id"]]["original96"][ph]) for r in rr)/len(rr))
            delta.append([math.fsum(loss[r["id"]]["stable96"][ph][l-1]-loss[r["id"]]["original96"][ph][l-1] for r in rr)/len(rr) for l in changed])
        A.extend(delta);u.extend([0. if i<4 else .02*v for i,v in enumerate(b)])
        A.append([math.fsum(v[k] for v in delta) for k in range(len(changed))]);u.append(0.)
        A.append([math.fsum(v[k] for v in delta[:4]) for k in range(len(changed))]);u.append(-.05*math.fsum(b[:4]))
        blocks.append((omit,b,delta))
    def assess(z):
        folds=[]
        for omit,b,delta in blocks:
            x=[math.fsum(v*q for v,q in zip(row,z)) for row in delta]
            g=( -math.fsum(x[:4]))/max(math.fsum(b[:4]),1e-12)
            viol=[key for i,key in enumerate(a.GROUPS) if x[i]>(0. if i<4 else .02*b[i])+1e-12]
            allok=math.fsum(x)<=1e-12
            folds.append(dict(omitted_line=omit,primary_gain=g,violations=viol,overall_guard=allok,pass_local_algebra=g>=.05 and not viol and allok))
        return folds
    old=assess(anchor);need(old[0]["pass_local_algebra"] and sum(x["pass_local_algebra"] for x in old[1:])==sens["passed_folds"]==29 and abs(old[0]["primary_gain"]-c["predicted_local"]["primary_relative_improvement"])<1e-12,"anchor reconciliation")
    weights=[len(set(stable[str(l)])-set(base[str(l)])) for l in changed]
    target=[0]*len(changed)
    r=solve(A,u,target,weights=weights);out=dict(status="MINIMUM_SWAP_ROBUST_FIT_LOCAL_DIAGNOSTIC_NOT_ADMISSION",numpy=np.__version__,scipy=scipy.__version__,variables=len(changed),constraints=len(A),candidate_sha256=CAND,sensitivity_sha256=SENS,anchor_folds_passed=29,solver={k:v for k,v in r.items() if k!="choices"},limits="Solver minimizes total swapped experts relative to original96 under all full and aligned leave-one-out local constraints; no candidate mask emitted, no independent holdout, propagation, translation, or qualification. Minimum applies only to stated finite local linear constraints.")
    if r.get("feasible"):
        folds=assess(r["choices"]);need(all(x["pass_local_algebra"] for x in folds),"independent fold recomputation")
        out.update(proposed_layer_choice_changes=[dict(layer=l,from_choice="stable" if aa else "original",to_choice="stable" if zz else "original") for l,aa,zz in zip(changed,target,r["choices"]) if aa!=zz],original96_swaps=r["distance"],stable_layers=[l for l,z in zip(changed,r["choices"]) if z],full_primary_gain=folds[0]["primary_gain"],minimum_fold_primary_gain=min(x["primary_gain"] for x in folds[1:]),folds=folds)
    raw=json.dumps(out,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n"
    with q.out.open("xb") as f:f.write(raw)
    print(json.dumps(dict(status=out["status"],sha256=sha(raw),solver=out["solver"],proposed_layer_choice_changes=out.get("proposed_layer_choice_changes"),full_primary_gain=out.get("full_primary_gain"),minimum_fold_primary_gain=out.get("minimum_fold_primary_gain"),numpy=out["numpy"],scipy=out["scipy"])))
if __name__=="__main__":main()

