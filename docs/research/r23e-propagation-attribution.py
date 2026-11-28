"""Offline R23E propagated-state attribution, no model, no new selection."""
import argparse,hashlib,json,math,pathlib
P=pathlib.Path
REPLAY="89a5715647088b7560411f379f90d368348dc4a1453bb32acce9e3dac7410c9c"
REPORT="568555efc1ceb7519964306cb6bfbc4402f7bc94b3f3db457c182e6d39522ab8"
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p,h):
    need(p.is_file() and not p.is_symlink(),"regular file");b=p.read_bytes();need(sha(b)==h,"pin");return json.loads(b)
def avg(xs):return math.fsum(xs)/len(xs)
def summarize(rows):
    need(len(rows)==12 and len({x["id"] for x in rows})==12 and all(sum(x["direction"]==d for x in rows)==2 for d in DIRS),"rows")
    out={};samples=[];discord=[]
    for x in rows:
        a=x["original96"];b=x["candidate96"]
        need(len(a["logits"])==len(b["logits"])==9 and a["teacher_hashes"]==b["teacher_hashes"],"same teacher")
        for j,(p,q) in enumerate(zip(a["logits"],b["logits"])):
            need(p["step"]==q["step"]==j and p["teacher_argmax"]==q["teacher_argmax"],"step")
            need(all(math.isfinite(z[k]) and z[k]>=0 for z in (p,q) for k in ("nmse","teacher_to_masked_kl")),"finite")
            dn=q["nmse"]-p["nmse"];dk=q["teacher_to_masked_kl"]-p["teacher_to_masked_kl"]
            r=dict(id=x["id"],direction=x["direction"],step=j,nmse_delta=dn,kl_delta=dk,teacher_argmax=p["teacher_argmax"],base_argmax=p["masked_argmax"],candidate_argmax=q["masked_argmax"])
            samples.append(r)
            if dn<0 and dk>0:discord.append(r)
    for d in DIRS:
        for phase in ("prefill","decode"):
            group=d+"/"+phase;xs=[x for x in rows if x["direction"]==d]
            rec=[r for r in samples if r["direction"]==d and (r["step"]==0)==(phase=="prefill")]
            layers={}
            for l in range(1,48):
                a=[];b=[]
                for x in xs:
                    a.extend(z for z in x["original96"]["layer"] if z["layer"]==l and z["phase"]==phase)
                    b.extend(z for z in x["candidate96"]["layer"] if z["layer"]==l and z["phase"]==phase)
                n=2 if phase=="prefill" else 16;need(len(a)==len(b)==n and [z["step"] for z in a]==[z["step"] for z in b],"layer inventory")
                layers[str(l)]={k:avg([z[k] for z in b])-avg([z[k] for z in a]) for k in ("hidden_input_nmse","moe_output_nmse")}
            out[group]=dict(nmse_delta=avg([r["nmse_delta"] for r in rec]),kl_delta=avg([r["kl_delta"] for r in rec]),discordant_positions=sum(r["nmse_delta"]<0 and r["kl_delta"]>0 for r in rec),positions=len(rec),layers=layers)
    return dict(groups=out,positions=len(samples),discordant_positions=len(discord),largest_discordant=sorted(discord,key=lambda r:(-r["kl_delta"],r["id"],r["step"]))[:12],samples=samples)
def main():
    p=argparse.ArgumentParser();p.add_argument("--inputs",type=P,required=True);p.add_argument("--out",type=P,required=True);q=p.parse_args();d=q.inputs
    e=read(d/"replay.json",REPLAY);r=read(d/"report.json",REPORT);need(e["status"]=="NONUNIFORM_PROPAGATION_EXACT_REPLAY_NOT_QUALITY" and e["report_sha256"]==REPORT and e["samples"]==12,"replay")
    rows=[read(d/f"{i:02d}.json",h) for i,h in enumerate(e["receipt_sha256"])]
    for i,x in enumerate(rows):need(x["index"]==i and x["teacher_restored"] is True and x["admission_sha256"]==e["admission_sha256"],"receipt")
    x=summarize(rows)
    for g,v in x["groups"].items():
        a=r["groups"]["original96"][g];b=r["groups"]["candidate96"][g]
        need(abs(v["nmse_delta"]-(b["logit_nmse"]-a["logit_nmse"]))<1e-12 and abs(v["kl_delta"]-(b["teacher_to_masked_kl"]-a["teacher_to_masked_kl"]))<1e-12,"group reconciliation")
    x.update(status="R23E_PROPAGATION_ATTRIBUTION_NOT_QUALITY",source_replay_sha256=REPLAY,source_report_sha256=REPORT,limits="Actual fixed-trajectory paired differences, descriptive only, no causal layer attribution or candidate selection.")
    raw=json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n"
    with q.out.open("xb") as f:f.write(raw)
    print(json.dumps(dict(status=x["status"],sha256=sha(raw),positions=x["positions"],discordant_positions=x["discordant_positions"],largest_discordant=x["largest_discordant"][:4],primary={g:{k:v[k] for k in ("nmse_delta","kl_delta","discordant_positions","positions")} for g,v in x["groups"].items() if g.startswith(("zh-my/","my-zh/"))})))
if __name__=="__main__":main()
