"""Byte-budgeted dequantized expert matrix cache for the CPU research adapter.

Single-threaded, inference-only, immutable local pack. Budget covers retained
BF16 tensor payload, not allocator, live GEMM references, or OS page cache.
"""
import argparse,collections,hashlib,importlib.util,json,math,os,pathlib,tempfile
import torch
P=pathlib.Path
QUANT="b9f938cc0ff430c127017e2b30d4ae04392f82b6"
def need(x,m):
    if not x:raise ValueError(m)
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
class Cache:
    def __init__(self,capacity,load,identity):
        need(type(capacity) is int and capacity>=0,"cache capacity")
        self.capacity=capacity;self.load=load;self.identity=identity;self.stamp=identity()
        self.entries=collections.OrderedDict();self.bytes=0;self.peak=0
        self.hits=0;self.misses=0;self.evictions=0;self.bypasses=0;self.invalid=False
    def check(self):
        if self.invalid:raise ValueError("cache invalidated")
        if self.identity()!=self.stamp:
            self.clear();self.invalid=True;raise ValueError("immutable payload changed")
    def clear(self):self.entries.clear();self.bytes=0
    def get(self,key,expected_bytes):
        need(type(expected_bytes) is int and expected_bytes>0,"matrix bytes")
        self.check()
        if key in self.entries:
            t,n=self.entries[key];need(n==expected_bytes,"cache shape identity")
            self.entries.move_to_end(key);self.hits+=1;return t
        self.misses+=1
        if expected_bytes<=self.capacity:
            while self.bytes+expected_bytes>self.capacity:
                _,(_,n)=self.entries.popitem(last=False);self.bytes-=n;self.evictions+=1
        t=self.load(key);self.check()
        need(t.device.type=="cpu" and t.dtype==torch.bfloat16 and t.is_contiguous() and t.numel()*t.element_size()==expected_bytes,"cache tensor contract")
        if expected_bytes>self.capacity:self.bypasses+=1;return t
        self.entries[key]=(t,expected_bytes);self.bytes+=expected_bytes;self.peak=max(self.peak,self.bytes)
        need(self.bytes<=self.capacity,"resident byte budget");return t
    def stats(self):
        return dict(capacity=self.capacity,retained_bytes=self.bytes,peak_retained_bytes=self.peak,hits=self.hits,misses=self.misses,evictions=self.evictions,bypasses=self.bypasses,entries=len(self.entries))
def stamp(path):
    st=path.stat(follow_symlinks=False)
    need(path.is_file() and not path.is_symlink(),"regular payload")
    return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns
def cached_class(quant,r,s,layout):
    Base=quant.checkpoint_class(r,s,layout)
    class Cached(Base):
        def __init__(self,path,digest,capacity):
            super().__init__(path,digest)
            self.cache=Cache(capacity,lambda key:Base.tensor(self,key),lambda:stamp(self.payload))
        def tensor(self,name):
            if name not in self.records:return super().tensor(name)
            return self.cache.get(name,math.prod(self.records[name]["shape"])*2)
    return Cached
def unit():
    calls=[];state=[0]
    c=Cache(8,lambda k:(calls.append(k) or torch.zeros(2,dtype=torch.bfloat16)),lambda:state[0])
    for k in ("a","b","a","c","a","b"):c.get(k,4)
    need(calls==["a","b","c","b"] and c.hits==2 and c.evictions==2 and list(c.entries)==["a","b"],"LRU oracle")
    z=Cache(0,lambda k:torch.ones(2,dtype=torch.bfloat16),lambda:0);z.get("a",4);need(z.bytes==0 and z.bypasses==1,"zero capacity")
    o=Cache(2,lambda k:torch.ones(2,dtype=torch.bfloat16),lambda:0);o.get("a",4);need(o.bytes==0 and o.bypasses==1,"oversize")
    def reject(fn):
        try:fn()
        except ValueError:return
        raise AssertionError("bad input accepted")
    reject(lambda:Cache(-1,None,lambda:0))
    reject(lambda:c.get("a",2))
    state[0]=1;reject(lambda:c.get("a",4));need(c.invalid and c.bytes==0,"change invalidates all")
    reject(lambda:c.get("b",4))
    f=Cache(8,lambda k:torch.zeros(2),lambda:0);reject(lambda:f.get("a",4));need(f.bytes==0,"invalid tensor not cached")
    print(json.dumps(dict(status="PASS_CACHE_UNIT",negative=5,lru_oracle=True,oversize_bypass=True,zero_bypass=True)))
def main():
    a=argparse.ArgumentParser()
    for n in ("quant","adapter","stream","layout"):a.add_argument("--"+n,type=P,required=True)
    args=a.parse_args();need(blob(args.quant.read_bytes())==QUANT,"quant pin")
    spec=importlib.util.spec_from_file_location("quant",args.quant);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    r,s,layout=[q.load(getattr(args,n),n) for n in q.PINS]
    for f,h in layout.PINS.items():need(s.fsha(P(q.q.__file__).parent/f)==h,"quant source")
    from mtpruning.testing import tiny_checkpoint
    torch.set_num_threads(1);unit();out=[]
    with tempfile.TemporaryDirectory() as d:
        p=P(d);src=tiny_checkpoint(p/"source",p/"none",vocab_size=256,seed=91,num_experts=8,top_k=2,layers=4);mask={"1":[0,2,4],"2":[1,2,3,5,6],"3":list(range(8))}
        for typ in ("Q4_0","Q8_0"):
            dest=p/typ;h=q.build(r,s,layout,src,dest,mask,typ)
            Base=q.checkpoint_class(r,s,layout);oracle=Base(dest,h);a=q.engine_class(r)(oracle,1)
            size=sum(math.prod(v["shape"])*2 for v in oracle.records.values());one=max(math.prod(v["shape"])*2 for v in oracle.records.values())
            for cap in (0,one*6,one*18,size):
                ck=cached_class(q,r,s,layout)(dest,h,cap);b=q.engine_class(r)(ck,1);need(ck.requests==0,"no eager dequant")
                before=oracle.requests
                fixed,route=r.compare_fixed(a,b,[4,8,15,16,23,42],[11,12,13,14,15,16,17,18])
                ga=a.generate([4,8,15],8,trace=True);gb=b.generate([4,8,15],8,trace=True)
                need(ga.tokens==gb.tokens and ga.finish==gb.finish and all(torch.equal(v,gb.trace[k]) for k,v in ga.trace.items()),"greedy")
                stats=ck.cache.stats();need(stats["hits"]+stats["misses"]==oracle.requests-before and stats["misses"]==ck.requests,"access ledger")
                need(stats["peak_retained_bytes"]<=cap,"budget")
                if cap==0:need(stats["hits"]==0 and stats["bypasses"]==ck.requests,"zero oracle")
                if cap==size:need(ck.requests<=len(ck.records) and stats["evictions"]==0 and stats["hits"]>0,"full cache")
                out.append(dict(type=typ,**stats,dequant_requests=ck.requests,fixed_logits=fixed,route_hooks=route,greedy_trace=True))
            # Mutate a populated cached expert. Metadata change must invalidate the
            # entire cache, not silently return stale weights on a hit.
            ck=cached_class(q,r,s,layout)(dest,h,size);key=next(iter(ck.records));ck.tensor(key);v=ck.records[key]
            with ck.payload.open("r+b") as f:f.seek(v["offset"]);x=f.read(1);f.seek(v["offset"]);f.write(bytes([x[0]^1]));f.flush();os.fsync(f.fileno())
            try:ck.tensor(key)
            except ValueError:pass
            else:raise AssertionError("stale cache hit")
            need(ck.cache.invalid and ck.cache.bytes==0,"mutation invalidation")
    print(json.dumps(dict(status="PASS_CPU_BOUNDED_EXPERT_CACHE",results=out,ordinary_file_mutation_invalidates=True,actual_model="NOT_RUN",rss_or_speed="NOT_MEASURED",scope="single-thread immutable pack; tensor payload accounting only")))
if __name__=="__main__":main()
