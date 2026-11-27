"""R23D exact dynamic programming for one declared additive fit objective.
Layer-local error curves only. Overall/primary/secondary guards evaluated
after optimization; no claim of constrained-global optimality if rejected.
No holdout, physical materialization or translation admission.
"""
import argparse,hashlib,json,math,pathlib,os
P=pathlib.Path
BUDGETS=(64,80,96,112,128)
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
GROUPS=tuple(d+"/"+p for d in DIRS for p in ("prefill","decode"))
KEYS={64:"k64",80:"k80",96:"original96",112:"k112",128:"k128"}
def need(x,m):
    if not x:raise ValueError(m)
def sha(x):return hashlib.sha256(x).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def obj(v):return math.fsum((2 if i<4 else 1)*z for i,z in enumerate(v))
def better(a,b):return a[0]<b[0]-1e-12 or (abs(a[0]-b[0])<=1e-12 and a[1]<b[1])
def solve(curves,cap):
    layers=sorted(curves,key=int);need(layers and type(cap) is int and len(layers)*64<=cap<=len(layers)*128,"budget cap")
    for l in layers:
        q=curves[l]
        need(set(q)==set(BUDGETS) and all(len(v)==12 and all(math.isfinite(z) and z>=0 for z in v) for v in q.values()),"curve shape")
    # Exact DP on discrete expert count for a single additive weighted objective.
    # Stable tie order prefers fewer deviations from uniform96, then lex budgets.
    dp={0:(0.,0,())}
    for l in layers:
        nxt={}
        for used,(cost,dev,seq) in dp.items():
            for k in BUDGETS:
                n=used+k
                if n>cap:continue
                a=(cost+obj(curves[l][k]),dev+abs(k-96),seq+(k,))
                b=nxt.get(n)
                if b is None or a[0]<b[0]-1e-12 or (abs(a[0]-b[0])<=1e-12 and a[1:]<b[1:]):nxt[n]=a
        dp=nxt
    need(dp,"no allocation")
    best=None;used=None
    for n,a in dp.items():
        key=(a[0],a[1],n,a[2])
        if best is None or key[0]<best[0]-1e-12 or (abs(key[0]-best[0])<=1e-12 and key[1:]<best[1:]):best=key;used=n
    ks=best[3]
    def means(q):return [math.fsum(curves[l][k][i] for l,k in zip(layers,q))/len(layers) for i in range(12)]
    baseline=means([96]*len(layers));result=means(ks)
    violations=[GROUPS[i] for i in range(12) if result[i]>baseline[i]*(1 if i<4 else 1.02)+1e-12]
    overall=math.fsum(result)<=math.fsum(baseline)+1e-12
    improvement=(math.fsum(baseline[:4])-math.fsum(result[:4]))/max(math.fsum(baseline[:4]),1e-12)
    changed=any(k!=96 for k in ks);feasible=not violations and overall and changed and improvement>=.05
    return dict(status="FIT_ALLOCATION_FOR_INDEPENDENT_DIAGNOSTIC_ONLY" if feasible else "NO_FIT_ALLOCATION_ADMISSION",allocated=dict(zip(layers,ks)),budget_used=used,budget_cap=cap,mean_experts=used/len(layers),fit_base=dict(zip(GROUPS,baseline)),fit_selected=dict(zip(GROUPS,result)),fit_primary_improvement=improvement,violations=violations,overall_guard=overall,method="exact discrete DP of additive fit weighted group loss (primary x2), then unchanged guards; not exact multi-constraint optimization; no holdout or propagated state",physical="Mask-only research, heterogeneous physical representation not qualified")
def selftest():
    a={64:[.6]*12,80:[.4]*12,96:[.3]*12,112:[.1]*12,128:[.1]*12}
    b={64:[.5]*12,80:[.1]*12,96:[.1]*12,112:[.1]*12,128:[.1]*12}
    r=solve({"1":a,"2":b},192)
    need(r["allocated"]=={"1":112,"2":80} and r["budget_used"]==192 and r["status"]=="FIT_ALLOCATION_FOR_INDEPENDENT_DIAGNOSTIC_ONLY","DP oracle")
    import itertools
    c={"1":a,"2":b,"3":{k:[1/(k/16)]*12 for k in BUDGETS}}
    r=solve(c,288)
    value=sum(obj(c[l][k]) for l,k in r["allocated"].items())
    exact=min(sum(obj(c[l][k]) for l,k in zip(c,ks)) for ks in itertools.product(BUDGETS,repeat=3) if sum(ks)<=288)
    need(abs(value-exact)<1e-12,"exhaustive oracle")
    q=solve({"1":{k:[.2]*12 for k in BUDGETS}},96);need(q["allocated"]=={"1":96} and q["status"]=="NO_FIT_ALLOCATION_ADMISSION","no unjustified change")
    def rejects(f):
        try:f()
        except ValueError:return
        raise AssertionError("bad input accepted")
    rejects(lambda:solve({"1":a},63))
    rejects(lambda:solve({"1":dict(a,**{})|{64:[float("nan")]*12}},96))
    print(json.dumps(dict(status="PASS_SYNTHETIC_LAYER_BUDGET_DP",positive=3,negative=2,real_curves="NOT_RUN")))
def main():
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true");p.add_argument("--report",type=P);p.add_argument("--report-sha");p.add_argument("--out",type=P);q=p.parse_args()
    if q.selftest:selftest();return
    need(q.report and q.report_sha and q.out,"explicit inputs")
    raw=q.report.read_bytes();need(sha(raw)==q.report_sha,"curve report pin")
    x=json.loads(raw);need(x["status"]=="FIT_ONLY_INTERACTION_BUDGET_DIAGNOSTIC_NOT_ADMISSION" and tuple(x["groups"])==GROUPS,"curve report scope")
    data=x["layer_group_combined_error"];need(set(data)=={str(i) for i in range(1,48)},"layer inventory")
    curves={l:{k:v[name] for k,name in KEYS.items()} for l,v in data.items()}
    out=solve(curves,47*96);out["report_sha256"]=q.report_sha
    raw=canon(out)+b"\n"
    with q.out.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(out,receipt_sha256=sha(raw))),flush=True)
if __name__=="__main__":main()
