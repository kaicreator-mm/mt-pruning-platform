"""Layer-partitioned vs global byte LRU on actual CPU synthetic inference.
No real workload extrapolation. Cache partition is static, no holdout tuning.
"""
import argparse,hashlib,importlib.util,json,math,pathlib,re,tempfile
import torch
P=pathlib.Path
CACHE="9f0d3737d2b2b260cd31fa8c291af9e945101256"
QUANT="b9f938cc0ff430c127017e2b30d4ae04392f82b6"
def need(x,m):
    if not x:raise ValueError(m)
def load(p,n,h):
    b=p.read_bytes();need(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()==h,"source pin")
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def partition_class(cache,q,r,s,layout):
    Base=q.checkpoint_class(r,s,layout)
    class Partitioned(Base):
        def __init__(self,path,digest,capacity):
            super().__init__(path,digest);need(type(capacity) is int and capacity>=0,"capacity")
            self.capacity=capacity;self.invalid=False;self.stamp=cache.stamp(self.payload)
            n=len(self.sparse_layers);need(n>0,"sparse layers")
            self.caches={l:cache.Cache(capacity//n+(i<capacity%n),lambda key:Base.tensor(self,key),lambda:self.stamp) for i,l in enumerate(self.sparse_layers)}
        def check(self):
            if self.invalid:raise ValueError("partition cache invalidated")
            try:current=cache.stamp(self.payload)
            except (OSError,ValueError):
                self.invalidate();raise ValueError("payload unavailable")
            if current!=self.stamp:
                self.invalidate();raise ValueError("immutable payload changed")
        def invalidate(self):
            self.invalid=True
            for c in self.caches.values():c.clear();c.invalid=True
        def tensor(self,name):
            if name not in self.records:return super().tensor(name)
            self.check();l=int(r.EXPERT.fullmatch(name)[1])
            # Check identity after a miss as well; do not return a value if the
            # file changed during loading. Ordinary metadata-change guard only.
            try:
                t=self.caches[l].get(name,math.prod(self.records[name]["shape"])*2)
                self.check();return t
            except (OSError,ValueError):
                self.invalidate();raise
        def stats(self):
            per={str(l):c.stats() for l,c in self.caches.items()}
            total={k:sum(v[k] for v in per.values()) for k in ("capacity","hits","misses","evictions","bypasses","entries","retained_bytes","peak_retained_bytes")}
            # Sum of per-layer peaks is a conservative bound, not a measured
            # simultaneous peak or allocator/RSS measurement.
            need(total["capacity"]==self.capacity and total["peak_retained_bytes"]<=self.capacity,"aggregate partition bound")
            return dict(total,per_layer=per)
    return Partitioned
def main():
    a=argparse.ArgumentParser()
    for n in ("cache","quant","adapter","stream","layout"):a.add_argument("--"+n,type=P,required=True)
    args=a.parse_args();c=load(args.cache,"cache",CACHE);q=load(args.quant,"quant",QUANT)
    r,s,layout=[q.load(getattr(args,n),n) for n in q.PINS]
    for f,h in layout.PINS.items():need(s.fsha(P(q.q.__file__).parent/f)==h,"quant source")
    from mtpruning.testing import tiny_checkpoint
    torch.set_num_threads(1);out=[]
    with tempfile.TemporaryDirectory() as d:
        p=P(d);src=tiny_checkpoint(p/"source",p/"none",vocab_size=256,seed=91,num_experts=8,top_k=2,layers=4);mask={"1":[0,2,4],"2":[1,2,3,5,6],"3":list(range(8))}
        for typ in ("Q4_0","Q8_0"):
            dest=p/typ;h=q.build(r,s,layout,src,dest,mask,typ)
            base=q.checkpoint_class(r,s,layout)(dest,h);a=q.engine_class(r)(base,1)
            one=max(math.prod(v["shape"])*2 for v in base.records.values())
            for cap in (one*6,one*18,one*30):
                for policy in ("global","partitioned"):
                    ck=(c.cached_class(q,r,s,layout) if policy=="global" else partition_class(c,q,r,s,layout))(dest,h,cap)
                    b=q.engine_class(r)(ck,1);need(ck.requests==0,"no eager")
                    before=base.requests
                    fixed,route=r.compare_fixed(a,b,[4,8,15,16,23,42],[11,12,13,14,15,16,17,18])
                    ga=a.generate([4,8,15],8,trace=True);gb=b.generate([4,8,15],8,trace=True)
                    need(ga.tokens==gb.tokens and ga.finish==gb.finish and all(torch.equal(v,gb.trace[k]) for k,v in ga.trace.items()),"greedy")
                    stats=ck.cache.stats() if policy=="global" else ck.stats()
                    need(stats["misses"]==ck.requests and stats["hits"]+stats["misses"]==base.requests-before and stats["peak_retained_bytes"]<=cap,"ledger and cap")
                    out.append(dict(type=typ,policy=policy,capacity=cap,hits=stats["hits"],misses=stats["misses"],evictions=stats["evictions"],peak_bound=stats["peak_retained_bytes"],fixed_logits=fixed,route_hooks=route,greedy_trace=True))
            ck=partition_class(c,q,r,s,layout)(dest,h,one*18);key=next(iter(ck.records));ck.tensor(key)
            ck.payload.rename(dest/"renamed.bin")
            try:ck.tensor(key)
            except ValueError:pass
            else:raise AssertionError("missing payload accepted")
            need(ck.invalid and all(x.bytes==0 for x in ck.caches.values()),"all partitions cleared")
    print(json.dumps(dict(status="PASS_CPU_LAYER_CACHE_COMPARISON",results=out,missing_payload_invalidates=True,actual_model="NOT_RUN",limits="same tiny fixture and sequence, static equal per-layer allocation; no latency or real hit-rate claim")))
if __name__=="__main__":main()
