"""P09: repetition diagnostic study on saved validation; no new gate."""
import ast, hashlib, json, pathlib, re, statistics, sys, unicodedata
from collections import Counter
root=pathlib.Path(sys.argv[1])
def need(v,msg):
    if not v: raise ValueError(msg)
def raw(n):return (root/n).read_bytes()
def lines(n):return [json.loads(x) for x in raw(n).splitlines() if x]
pins={"corpus.jsonl":"5fc8e3586a486e89ca6e0e6644d6c79ad1e4de72","manifest.json":"05322b3a96e06d2a7676dcc39b6efccd15093bfb","baseline.jsonl":"8b2b9a3394c52973201115913374a8a8e1e29749","physical.jsonl":"7593b21a0395650fca1094d2b2900a8f6a4b94c4","masked.jsonl":"1423d039695659dea7d4ea074907bb258d7a825c","evaluate.py":"ac122a7f46f0381cba5a2b019c42e391c670d968"}
for n,v in pins.items():
    b=raw(n);need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==v,n+" blob")
m=json.loads(raw("manifest.json"));need(hashlib.sha256(raw("corpus.jsonl")).hexdigest()==m["corpus_sha256"],"corpus")
dirs=m["directions"];expected={f"validation:{d}:{i}" for d in dirs for i in m["indices"][d.split("-")[0]]["validation"]}
rows=[x for x in lines("corpus.jsonl") if x["split"]=="validation"];ctx={x["id"]:x for x in rows};need(len(rows)==len(ctx)==120 and set(ctx)==expected,"corpus IDs")
def load(n,k,mask,num,anchor=None):
    h,*rows=lines(n)
    need(h["mode"]=="real" and h["synthetic"] is False and h["num_experts"]==k and h["masked_keep"]==mask and h["corpus_sha256"]==m["corpus_sha256"] and h["max_new_tokens"]==640 and h["decoding"]=="greedy (argmax of bf16 logits)" and "test" not in h["splits"] and all(x["split"]!="test" for x in rows),"header")
    if anchor:need(all(h["identity"][k]==anchor["identity"][k] for k in ("teacher","corpus","prompt","runtime","code","decoding","mode")),"shared identity")
    rows=[x for x in rows if x["split"]=="validation"];res={x["id"]:x for x in rows}
    need(len(rows)==len(res)==num and set(res)<=expected,"rows")
    for i,x in res.items():
        c=ctx[i];need(x["direction"]==c["direction"] and x["line"]==c["line"] and x["prompt_tokens"]==len(c["prompt_ids"]) and x["new_tokens"]==len(x["tokens"]) and x["new_tokens"]<=640 and x["identity_key"]==h["identity_keys"]["validation"],"context")
    return h,res
bh,b=load("baseline.jsonl",128,None,120);_,p=load("physical.jsonl",96,None,48,bh);_,z=load("masked.jsonl",128,96,48,bh)
need(set(p)==set(z)=={i for i,x in b.items() if x["traced"]},"witness IDs")
# Exact existing heuristic, extracted from pinned source without torch.
tree=ast.parse(raw("evaluate.py"));names={"digit_runs","script_share","failure_flags"}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_DIGITS" for t in n.targets)]
need(len(nodes)==4 and {n.name for n in nodes if isinstance(n,ast.FunctionDef)}==names,"definitions")
env={"re":re,"unicodedata":unicodedata};exec(compile(ast.Module(body=nodes,type_ignores=[]),"pinned-flags","exec"),env)
def normalize(s):return " ".join(unicodedata.normalize("NFC",s).split())
def scan(seq,max_period,min_span):
    # Longest exact contiguous tandem repeat span with >=3 copies.
    # For fixed period p, adjacent matches run >=2*p implies >=3 copies.
    n=len(seq);best={"span":0,"period":0,"copies":0,"start":0};flag=False;tail=False
    for p in range(1,min(max_period,n//3)+1):
        run=0
        for i in range(p,n):
            if seq[i]==seq[i-p]:run+=1
            else:run=0
            if run>=2*p:
                span=(run//p+1)*p;start=i-run-p+1
                if span>best["span"]:best={"span":span,"period":p,"copies":span//p,"start":start}
                if span>=min_span and 2*span>=n:flag=True
                # trailing periodic run of >= min_span and >=3 copies,
                # even when a long non-repeated prefix precedes it
                if i==n-1 and span>=min_span:tail=True
    return {**best,"length":n,"dominant":flag,"tail":tail,"flag":flag or tail}
def check():
    tests=[
      ("no_repeat","abcdef ghijkl mnopqr",False),
      ("char_loop","a"*24,True),
      ("phrase_loop","abc def "*8,True),
      ("late_loop","unique prefix words "+"oops "*12,True),
      ("short_legitimate","ha ha ha",False),
      ("burmese_repeated_phrase","မင်္ဂလာပါ "*8,True),
      ("empty","",False)]
    for name,s,want in tests:need(scan(normalize(s),64,24)["flag"]==want,"synthetic "+name)
    need(scan([1,2,3]*8,32,24)["flag"],"token loop")
    need(not scan(list(range(40)),32,24)["flag"],"token unique")
    # ensure known run lengths exact, not off-by-one.
    need(scan("xyzxyzxyz",64,24)["span"]==9,"span length")
    need(scan("pxyzxyzxyzq",64,24)["span"]==9,"internal run")
    return len(tests)+4
checks=check()
out={};detail=[]
for name,rs in (("baseline",b),("physical96",p),("masked96",z)):
    out[name]={}
    for d in dirs:
        ids=sorted(i for i in rs if ctx[i]["direction"]==d)
        need(len(ids)==(20 if name=="baseline" else 8),"direction coverage")
        v=[]
        for i in ids:
            x=rs[i];c=ctx[i];t=x["tokens"];s=normalize(x["text"]);ref=normalize(c["reference"])
            flags=env["failure_flags"](c["source"],c["reference"],x["text"],d.split("-")[1],x["finish"],t)
            token=scan(t,32,24);char=scan(s,64,24);refchar=scan(ref,64,24)
            a={"id":i,"run":name,"direction":d,"old_repetition":"degenerate_repetition" in flags,"token_unique_ratio":len(set(t))/len(t) if t else None,"tokens":len(t),"normalized_chars":len(s),"token_tandem":token,"char_tandem":char,"reference_char_tandem":refchar}
            v.append(a);detail.append(a)
        q=[a["token_unique_ratio"] for a in v if a["token_unique_ratio"] is not None]
        out[name][d]={"n":len(v),"old":sum(a["old_repetition"] for a in v),"token_flag":sum(a["token_tandem"]["flag"] for a in v),"char_flag":sum(a["char_tandem"]["flag"] for a in v),"both":sum(a["token_tandem"]["flag"] and a["char_tandem"]["flag"] for a in v),"reference_char_flag":sum(a["reference_char_tandem"]["flag"] for a in v),"median_unique_ratio":statistics.median(q),"min_unique_ratio":min(q),"max_unique_ratio":max(q),"median_tokens_per_codepoint":statistics.median(a["tokens"]/a["normalized_chars"] for a in v if a["normalized_chars"]),"max_token_repeat_span":max(a["token_tandem"]["span"] for a in v),"max_char_repeat_span":max(a["char_tandem"]["span"] for a in v)}
print(json.dumps({"status":"DIAGNOSTIC_STUDY_ONLY","input_blobs":pins,"synthetic_checks":checks,"settings":{"normalization":"NFC + collapse whitespace","token_max_period":32,"char_max_period":64,"min_span":24,"min_copies":3,"flag":"dominant >= half sequence or trailing span"},"summary":out,"rows":detail},allow_nan=False,separators=(",",":")))

