"""R23G one predeclared layer-38 reversion; fit-only diagnostic, no admission."""
import argparse, hashlib, json, math, pathlib
P=pathlib.Path
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def pin(p,h,git=False):
    need(p.is_file() and not p.is_symlink(),"regular file")
    b=p.read_bytes();need((blob if git else sha)(b)==h,"pin "+str(p));return json.loads(b)
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
GROUPS=tuple(d+"/"+p for d in DIRS for p in ("prefill","decode"))
def predict(b,s,delta):
    need(set(b)==set(s)==set(delta)==set(GROUPS),"groups")
    need(all(math.isfinite(v) and v>=0 for v in list(b.values())+list(s.values())) and all(math.isfinite(v) for v in delta.values()),"finite")
    h={g:s[g]-delta[g]/47 for g in GROUPS}
    need(all(math.isfinite(v) and v>=0 for v in h.values()),"predicted finite")
    gain=(math.fsum(b[g] for g in GROUPS[:4])-math.fsum(h[g] for g in GROUPS[:4]))/max(math.fsum(b[g] for g in GROUPS[:4]),1e-12)
    violations=[g for i,g in enumerate(GROUPS) if h[g]>b[g]*(1 if i<4 else 1.02)+1e-12]
    overall=math.fsum(h.values())<=math.fsum(b.values())+1e-12
    return dict(means=h,primary_relative_improvement=gain,violations=violations,overall_guard=overall,eligible_for_fit_propagation_only=gain>=.05 and not violations and overall)
def main():
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true");p.add_argument("--inputs",type=P);p.add_argument("--out",type=P);q=p.parse_args()
    if q.selftest:
        b={g:.25 for g in GROUPS};s={g:.2 for g in GROUPS};d={g:0. for g in GROUPS};s["zh-my/prefill"]=.3;d["zh-my/prefill"]=4.7
        need(predict(b,s,d)["eligible_for_fit_propagation_only"],"positive")
        d["zh-my/prefill"]=0;need(not predict(b,s,d)["eligible_for_fit_propagation_only"],"primary guard")
        s["en-my/prefill"]=.3;d["zh-my/prefill"]=4.7;need(not predict(b,s,d)["eligible_for_fit_propagation_only"],"secondary guard")
        print(json.dumps(dict(status="PASS_SYNTHETIC_L38_REVERSION",positive=1,negative=2)));return
    need(q.inputs and q.out,"inputs");d=q.inputs
    report=pin(d/"report.json","528379db75769a7403ab614541085e238dae35a120571fda994878bf69c90a2b")
    fit=pin(d/"fit"/"decision.json","7821abc55b356520bdd02e4b60f32057a95d204b8cc723b894e2fdce3d9d100d")
    c=pin(d/"candidate.json","488a27afd5e75019cb6cc7355b9f847bbca515fc7570da081020a23b3e6213ef")
    base=pin(d/"original.json","d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed",True)["candidates"]["96"]["kept"]
    plan=pin(d/"expanded.json","a1047d1a7c5624ed0e3a868bd5d76eef4c754915",True)
    need(report["status"]=="FIT_LAYER_ATTRIBUTION_ONLY_NOT_NEW_MASK" and report["samples"]==48 and report["zh_my_prefill_rank"][0]==38 and fit["violations"]==["zh-my/prefill"],"frozen rationale")
    stable=c["kept"];need(sha(canon(stable))==c["mask_sha256"]==fit["candidate_mask_sha256"] and sha(canon(base))==c["base_mask_sha256"],"source masks")
    need(set(base)==set(stable)=={str(i) for i in range(1,48)} and all(len(v)==96 and v==sorted(set(v)) and all(type(i) is int and 0<=i<128 for i in v) for m in (base,stable) for v in m.values()),"shape")
    l=report["layers"]["38"];need(l["added"]==sorted(set(stable["38"])-set(base["38"]))==[41,43] and l["removed"]==sorted(set(base["38"])-set(stable["38"]))==[89,123],"exact reversion")
    h=dict(stable);h["38"]=base["38"];need([k for k in h if h[k]!=stable[k]]==["38"],"single layer")
    pred=predict(fit["means"]["original96"],fit["means"]["stable96"],l["delta"])
    need(all(abs(pred["means"][g]-(fit["means"]["original96"][g]+report["single_layer_local_counterfactuals"][0]["local_delta_if_single_layer_reverted"][g]))<1e-12 for g in GROUPS),"algebra reconciliation")
    need(c["fit_ids"]==[r["id"] for r in plan["fit"]] and not ({r["line"] for r in plan["fit"]}&{r["line"] for r in plan["holdout"]}),"fit isolation")
    out=dict(status="FROZEN_L38_REVERSION_FOR_FIT_PROPAGATION_ONLY" if pred["eligible_for_fit_propagation_only"] else "NO_FIT_DIAGNOSTIC_CANDIDATE",fit_samples=48,reverted_layer=38,retained_experts=sum(map(len,h.values())),mask_sha256=sha(canon(h)),kept=h,source_candidate_sha256="488a27afd5e75019cb6cc7355b9f847bbca515fc7570da081020a23b3e6213ef",source_report_sha256="528379db75769a7403ab614541085e238dae35a120571fda994878bf69c90a2b",source_fit_decision_sha256="7821abc55b356520bdd02e4b60f32057a95d204b8cc723b894e2fdce3d9d100d",predicted_local=pred,limits="Selection uses previously observed fit only. Algebraic local reconstruction is not measured propagated error, fresh validation, or translation quality. No fresh data consumed.")
    raw=canon(out)+b"\n"
    with q.out.open("xb") as f:f.write(raw)
    print(json.dumps(dict(status=out["status"],receipt_sha256=sha(raw),mask_sha256=out["mask_sha256"],retained_experts=out["retained_experts"],predicted_local=pred)))
if __name__=="__main__":main()
