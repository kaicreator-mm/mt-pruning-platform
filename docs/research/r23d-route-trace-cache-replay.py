"""Replay grouped expert matrix requests from ordered real engine route hooks.
Synthetic model capture only in CLI. Does not use aggregate counts as traces.
"""
import argparse,collections,hashlib,importlib.util,json,math,pathlib,tempfile,torch
P=pathlib.Path
PINS={"cache":"bb8bcc4dee528c6e6a98cf84b21bfe5256b31bcb","quant":"b9f938cc0ff430c127017e2b30d4ae04392f82b6"}
def need(x,m):
    if not x:raise ValueError(m)
def load(p,n):
    b=p.read_bytes();need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==PINS[n],"source pin")
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def requests(events,mask,topk):
    """One GEMM per distinct selected expert, ordered by compact ID.
    Grouped original engine indexes gate, up, down once per group.
    """
    for e in events:
        l=e["layer"];phase=e["phase"];ids=mask[str(l)];rows=e["ids"]
        need(phase in ("prefill","decode") and bool(rows) and all(len(row)==topk and len(set(row))==topk and set(row)<=set(ids) for row in rows),"route hook identity")
        for original in sorted({i for row in rows for i in row}):
            local=ids.index(original)
            for p in ("gate","up","down"):
                yield (phase,l,f"model.layers.{l}.mlp.experts.{local}.{p}_proj.weight")
def replay(seq,sizes,quotas,mode):
    need(mode in ("global","layer") and all(type(v) is int and v>=0 for v in quotas.values()),"policy/quota")
    slots={l:collections.OrderedDict() for l in quotas};used={l:0 for l in quotas};peak=0
    stats={p:dict(hits=0,misses=0,evictions=0,bypasses=0) for p in ("prefill","decode")}
    for phase,l,key in seq:
        need(phase in stats and key in sizes and sizes[key]>0,"request")
        k=0 if mode=="global" else l;need(k in slots,"layer quota")
        c=slots[k];z=stats[phase];n=sizes[key]
        if key in c:c.move_to_end(key);z["hits"]+=1;continue
        z["misses"]+=1
        if n>quotas[k]:z["bypasses"]+=1;continue
        while used[k]+n>quotas[k]:
            _,v=c.popitem(last=False);used[k]-=v;z["evictions"]+=1
        c[key]=n;used[k]+=n;peak=max(peak,sum(used.values()))
        need(peak<=sum(quotas.values()),"bound")
    total={k:sum(v[k] for v in stats.values()) for k in next(iter(stats.values()))}
    return dict(total,peak_retained_bytes=peak,retained_bytes=sum(used.values()),by_phase=stats)
def capture(engine):
    events=[];prev=engine.on_route
    engine.on_route=lambda l,p,t,w:events.append(dict(layer=l,phase=p,ids=t.tolist()))
    try:
        outputs=[]
        for prompt in ([4,8,15,16,23,42],[17,7,31,9,3,2,5,8],[4,8,15,16,23,42]):
            g=engine.generate(prompt,8,trace=True);outputs.append(g)
        return events,outputs
    finally:engine.on_route=prev
def same(a,b):
    return len(a)==len(b) and all(x.tokens==y.tokens and x.finish==y.finish and set(x.trace)==set(y.trace) and all(torch.equal(v,y.trace[k]) for k,v in x.trace.items()) for x,y in zip(a,b))
def main():
    a=argparse.ArgumentParser()
    for n in ("cache","quant","adapter","stream","layout"):a.add_argument("--"+n,type=P,required=True)
    a=a.parse_args();c=load(a.cache,"cache");q=load(a.quant,"quant");r,s,layout=[q.load(getattr(a,n),n) for n in q.PINS]
    for f,h in layout.PINS.items():need(s.fsha(P(q.q.__file__).parent/f)==h,"quant source")
    from mtpruning.testing import tiny_checkpoint
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as d:
        p=P(d);src=tiny_checkpoint(p/"src",p/"none",vocab_size=256,seed=91,num_experts=128,top_k=8,layers=6)
        mask={str(i):list(range(128-k,128)) for i,k in enumerate((64,80,96,112,128),1)}
        dest=p/"q4";h=q.build(r,s,layout,src,dest,mask,"Q4_0")
        Base=q.checkpoint_class(r,s,layout);ck=Base(dest,h);observed=[];orig=ck.tensor
        def traced(n):
            if n in ck.records:observed.append(n)
            return orig(n)
        ck.tensor=traced;engine=q.engine_class(r)(ck,1);events,outputs=capture(engine)
        seq=list(requests(events,mask,8));need([x[2] for x in seq]==observed and len(seq)==ck.requests,"trace request oracle")
        sizes={n:math.prod(v["shape"])*2 for n,v in ck.records.items()};one=max(sizes.values())
        results=[]
        for matrices in (0,24,120,240,480):
            cap=matrices*one;g=replay(seq,sizes,{0:cap},"global");quota={i:cap//5+(i<=cap%5) for i in range(1,6)};l=replay(seq,sizes,quota,"layer")
            cc=c.cached_class(q,r,s,layout)(dest,h,cap);e=q.engine_class(r)(cc,1);ev,oo=capture(e);need(ev==events and same(outputs,oo),"cached trajectory")
            actual=cc.cache.stats()
            need(all(g[k]==actual[k] for k in ("hits","misses","evictions","bypasses","retained_bytes","peak_retained_bytes")) and actual["misses"]==cc.requests,"simulation exact live cache ledger")
            results.append(dict(matrix_capacity=matrices,fixture_capacity_bytes=cap,global_lru=g,layer_lru=l))
        # Strictly reject unobserved/removed expert and malformed hook.
        bad=dict(events[0],ids=[[0]*8])
        try:list(requests([bad],mask,8))
        except ValueError:pass
        else:raise AssertionError("invalid route accepted")
        # Check actual global missing file invalidates existing cache.
        cc=c.cached_class(q,r,s,layout)(dest,h,one*24);key=next(iter(cc.records));cc.tensor(key)
        cc.payload.rename(dest/"moved.bin")
        try:cc.tensor(key)
        except ValueError:pass
        else:raise AssertionError("stale cached weight")
        need(cc.cache.invalid and cc.cache.bytes==0,"missing payload invalidation")
        print(json.dumps(dict(status="PASS_CPU_ROUTE_TRACE_CACHE_REPLAY",route_hooks=len(events),requests=len(seq),trace_sha256=s.sha(s.canon(events)),expert_instances=sum(map(len,mask.values())),top_k=8,results=results,global_replay_matches_actual=True,missing_payload_cleared=True,real_teacher="NOT_RUN",limits="Actual synthetic-engine ordered hooks; not real translation cache hit rates, no latency claim.")))
if __name__=="__main__":main()
