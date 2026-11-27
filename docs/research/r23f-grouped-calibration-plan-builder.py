"""R23F metadata-only calibration expansion plan, deterministic aligned-line isolation.
Use existing R23 fit12 plus 36 new fit, 24 new holdout, if feasible.
Exclude all previously used R23 aligned lines from new records. Never read
expert profiles, reconstruction losses, validation/test text or reference strings.
"""
import argparse,hashlib,json,pathlib,collections
P=pathlib.Path
CORPUS="9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
BASELINE="37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8"
PLAN="3c92bd0a944637ee7eb8d36d271417396b313026"
FRESH="bc6d92ef87d23c954cd1c3f86527a39b52b5cc18"
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
SALT="R23F_EXPANDED_CALIBRATION_V1"
def need(x,m):
    if not x:raise ValueError(m)
def sha(x):return hashlib.sha256(x).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(x):return hashlib.sha1(b"blob "+str(len(x)).encode()+b"\0"+x).hexdigest()
def select(candidates,quota):
    # Deterministic augmenting-path bipartite matching: exact quota per direction,
    # one aligned line globally, no greedy starvation.
    slots=[(stage,d,j) for stage,n in quota for d in DIRS for j in range(n)]
    options={s:sorted(candidates[s[1]],key=lambda r:(sha((SALT+"|"+s[0]+"|"+r["id"]).encode()),r["id"])) for s in slots}
    owner={};assigned={}
    def match(s,seen):
        for r in options[s]:
            line=r["line"]
            if line in seen:continue
            seen.add(line);prior=owner.get(line)
            if prior is None or match(prior,seen):
                owner[line]=s;assigned[s]=r;return True
        return False
    for s in slots:
        if not match(s,set()):return None
    out={stage:[assigned[s] for s in slots if s[0]==stage] for stage,n in quota}
    allrows=[r for rows in out.values() for r in rows]
    need(len(allrows)==len({r["line"] for r in allrows})==len(slots),"matching coverage")
    return out
def select_grouped(candidates):
    # Group semantic aligned lines. Holdout groups carry at most 5 examples per
    # direction so that 6 new fit examples remain from 11 available. Select by
    # deterministic bounded exact DP of direction counts, not expert results.
    groups=collections.defaultdict(list)
    for d,rs in candidates.items():
        for r in rs:groups[r["line"]].append(r)
    caps=tuple(len(candidates[d])-6 for d in DIRS)
    if min(caps)<4:return None
    order=sorted(groups,key=lambda l:(sha((SALT+"|group|"+str(l)).encode()),l))
    dp={(0,)*6:()}
    for l in order:
        v=tuple(sum(r["direction"]==d for r in groups[l]) for d in DIRS)
        new=dict(dp)
        for counts,path in dp.items():
            q=tuple(a+b for a,b in zip(counts,v))
            if any(a>b for a,b in zip(q,caps)):continue
            alt=path+(l,)
            if q not in new or len(alt)<len(new[q]):new[q]=alt
        dp=new
    valid=[(c,p) for c,p in dp.items() if min(c)>=4]
    if not valid:return None
    # Prefer more distinct holdout lines for coverage; ties deterministic.
    _,chosen=min(valid,key=lambda q:(-len(q[1]),q[0],tuple(order.index(l) for l in q[1])))
    hold_lines=set(chosen)
    out={"holdout":[],"new_fit":[]}
    for d in DIRS:
        for stage,subset,n in (("holdout",[r for r in candidates[d] if r["line"] in hold_lines],4),("new_fit",[r for r in candidates[d] if r["line"] not in hold_lines],6)):
            rows=sorted(subset,key=lambda r:(sha((SALT+"|"+stage+"|"+r["id"]).encode()),r["id"]))[:n]
            need(len(rows)==n,"group quota");out[stage]+=rows
    need(not ({r["line"] for r in out["holdout"]}&{r["line"] for r in out["new_fit"]}),"group isolation")
    return out
def selftest():
    x={d:[{"id":d+":"+str(i),"line":j*10+i} for i in range(4)] for j,d in enumerate(DIRS)}
    q=select(x,(("holdout",1),("new_fit",2)))
    need(q and len(q["holdout"])==6 and len(q["new_fit"])==12,"balanced")
    need(select(x,(("holdout",3),("new_fit",2))) is None,"impossible")
    # Earlier direction can move to an alternative line to satisfy later one.
    x={d:[{"id":d+":0","line":100+i}] for i,d in enumerate(DIRS)}
    x[DIRS[0]]=[{"id":"a","line":101},{"id":"b","line":100}]
    q=select(x,(("holdout",1),));need(q and len({r["line"] for r in q["holdout"]})==6,"augment")
    x={d:[{"id":d+":"+str(i),"direction":d,"line":j//2*100+i} for i in range(11)] for j,d in enumerate(DIRS)}
    q=select_grouped(x);need(q and len(q["new_fit"])==36 and len(q["holdout"])==24 and not ({r["line"] for r in q["new_fit"]}&{r["line"] for r in q["holdout"]}),"grouped oracle")
    print(json.dumps(dict(status="PASS_SYNTHETIC_ALIGNED_CALIBRATION_MATCHING",checks=4)))
def main():
    a=argparse.ArgumentParser();a.add_argument("--selftest",action="store_true");a.add_argument("--plan",type=P);a.add_argument("--fresh",type=P);a.add_argument("--corpus",type=P);a.add_argument("--baseline",type=P);q=a.parse_args()
    if q.selftest:selftest();return
    need(q.plan and q.fresh and q.corpus and q.baseline,"input paths")
    p=q.plan.read_bytes();f=q.fresh.read_bytes();c=q.corpus.read_bytes();b=q.baseline.read_bytes()
    need(blob(p)==PLAN and blob(f)==FRESH and sha(c)==CORPUS and sha(b)==BASELINE,"input pins")
    old=json.loads(p)["plan"];fresh=json.loads(f)["rows"];prior=old["fit"]+old["holdout"]+fresh
    excluded={r["line"] for r in prior};need(len(excluded)==36,"prior isolation")
    # Keep only identity/token metadata of calibration rows; no content consumed for selection.
    fields=("id","line","direction","prompt_ids")
    cs={x["id"]:{k:x[k] for k in fields} for x in map(json.loads,c.splitlines()) if x["split"]=="calibration"}
    fields=("id","line","direction","prompt_tokens","finish","tokens")
    bs={x["id"]:{k:x[k] for k in fields} for x in map(json.loads,b.splitlines()[1:]) if x["split"]=="calibration"}
    need(len(cs)==len(bs)==144 and set(cs)==set(bs),"calibration inventory")
    candidates={d:[] for d in DIRS}
    for i,r in cs.items():
        t=bs[i];need(r["line"]==t["line"] and r["direction"]==t["direction"] and len(r["prompt_ids"])==t["prompt_tokens"],"identity")
        if r["line"] in excluded or t["finish"]!="eos" or len(t["tokens"])<8 or len(r["prompt_ids"])<2:continue
        ids=r["prompt_ids"];prefix=t["tokens"][:8]
        candidates[r["direction"]].append(dict(id=i,line=r["line"],direction=r["direction"],prompt_tokens=len(ids),prompt_ids_sha256=sha(canon(ids)),teacher_prefix_tokens=prefix,teacher_prefix_sha256=sha(canon(prefix)),prefill_rows=[len(ids)//2-1,len(ids)-1],decode_steps_1based=[1,8]))
    available={d:len(x) for d,x in candidates.items()};lines=len({r["line"] for xs in candidates.values() for r in xs})
    out=select_grouped(candidates)
    info=dict(salt=SALT,corpus_sha256=CORPUS,baseline_sha256=BASELINE,prior_plan_blob=PLAN,prior_fresh_blob=FRESH,excluded_aligned_lines=sorted(excluded),available_by_direction=available,available_unique_lines=lines,desired_fit=48,desired_holdout=24,limits="Existing R19 calibration, not untouched global test. New records selected by metadata only. Never use prior holdout for fitting; no expert results used.")
    if out is None:
        print(json.dumps(dict(status="EXPANDED_PLAN_INFEASIBLE",**info)));return
    fit=old["fit"]+out["new_fit"];hold=out["holdout"]
    need(len(fit)==48 and len(hold)==24 and len({r["id"] for r in fit+hold})==72 and not ({r["line"] for r in fit}&{r["line"] for r in hold}) and not ({r["line"] for r in out["new_fit"]+hold}&excluded),"final isolation")
    for d in DIRS:need(sum(r["direction"]==d for r in fit)==8 and sum(r["direction"]==d for r in hold)==4,"balance")
    print(json.dumps(dict(status="FROZEN_GROUPED_EXPANDED_R23F_CALIBRATION",**info,fit=fit,holdout=hold,fit_unique_lines=len({r["line"] for r in fit}),holdout_unique_lines=len({r["line"] for r in hold}),group_policy="aligned line globally disjoint across subsets; direction examples within subset are correlated; not 72 independent semantic lines"),ensure_ascii=False))
if __name__=="__main__":main()
