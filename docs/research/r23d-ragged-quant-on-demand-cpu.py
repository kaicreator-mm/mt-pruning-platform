"""Synthetic-only quantized routed-FFN pack and on-demand CPU dequantization.
Not native GGML quantized GEMM; no production builder, real model or VRAM claim.
"""
import argparse,hashlib,importlib.util,json,os,pathlib,tempfile,math
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import gguf.constants as gc
import gguf.quants as q
P=pathlib.Path
PINS={"adapter":"56ceecad1fd2db0d33399840e2231c718c7e6ab0","stream":"2a53ccf64f8dccdc31da096b58d59e91cb8820f8","layout":"a1b29370a7d7bddb2f3679f4783a7b784066fd34"}
def need(x,m):
    if not x:raise ValueError(m)
def load(p,n):
    raw=p.read_bytes();need(hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==PINS[n],"source pin")
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def build(r,s,layout,source,dest,mask,typ):
    need(typ in ("Q4_0","Q8_0") and not dest.exists(),"fixture type/destination")
    cfg,prov,ph,_=s.source_plan(r,source,mask,64<<20)
    ck=r.eng.Checkpoint(source);mp=r.mapping(ck.config,mask);dest.mkdir()
    s.write(dest/"config.json",cfg);plain={};records={};off=0
    enum=getattr(gc.GGMLQuantizationType,typ)
    with (dest/"experts.bin").open("xb") as f:
        for name,old in sorted(mp.items()):
            t=ck.tensor(old)
            if not r.EXPERT.fullmatch(name):plain[name]=t.clone().contiguous();continue
            shape=list(t.shape);z=q.quantize(t.float().numpy(),enum).tobytes()
            nb=layout.shape_layout(list(reversed(shape)),typ)["bytes"];need(len(z)==nb,"quantized size")
            records[name]=dict(shape=shape,offset=off,bytes=nb,sha256=s.sha(z));f.write(z);off+=nb
        f.flush();os.fsync(f.fileno())
    save_file(plain,str(dest/"plain.safetensors"))
    m=dict(format="r23-quant-ragged-ffn-fixture-v1",source_kind="synthetic",engine_blob=r.ENGINE,source_provenance_sha256=ph,config_sha256=s.sha(cfg),mask=mask,mask_sha256=s.sha(s.canon(mask)),type=typ,plain_sha256=s.fsha(dest/"plain.safetensors"),payload_sha256=s.fsha(dest/"experts.bin"),experts=records)
    raw=s.canon(m)+b"\n";s.write(dest/"manifest.json",raw);return s.sha(raw)
def checkpoint_class(r,s,layout):
    class Quant(r.RaggedCheckpoint):
        def __init__(self,path,digest):
            self.path=P(path);raw=s.read(self.path/"manifest.json");m=s.strict(raw)
            need(s.sha(raw)==digest and raw==s.canon(m)+b"\n" and m["format"]=="r23-quant-ragged-ffn-fixture-v1" and m["source_kind"]=="synthetic" and m["engine_blob"]==r.ENGINE,"manifest")
            need(set(p.name for p in self.path.iterdir())=={"config.json","manifest.json","plain.safetensors","experts.bin"},"inventory")
            raw=s.read(self.path/"config.json");need(s.sha(raw)==m["config_sha256"],"config");self.config=s.strict(raw);self.mask=m["mask"];need(s.sha(s.canon(self.mask))==m["mask_sha256"],"mask")
            self.sparse=r.validate(self.config,self.mask);self.sparse_layers=[i for i,v in enumerate(self.sparse) if v];self.num_experts=self.config["num_experts"];self.kept=None
            self.source_map=r.mapping(self.config,self.mask);shapes=s.expected_shapes(self.config,self.sparse,self.mask)
            self.typ=m["type"];need(self.typ in ("Q4_0","Q8_0"),"type");self.enum=getattr(gc.GGMLQuantizationType,self.typ)
            self.records=m["experts"];need(set(self.records)=={n for n in shapes if r.EXPERT.fullmatch(n)},"expert inventory")
            plain=self.path/"plain.safetensors";self.payload=self.path/"experts.bin"
            need(s.fsha(plain)==m["plain_sha256"] and s.fsha(self.payload)==m["payload_sha256"],"payload hash")
            desc=s.descriptors(plain);s.check_shapes(desc,{k:v for k,v in shapes.items() if k not in self.records})
            pos=0
            for k,v in sorted(self.records.items(),key=lambda p:p[1]["offset"]):
                shape,dtype=shapes[k];need(dtype=="BF16" and v["shape"]==shape and type(v["offset"]) is int and v["offset"]==pos and v["bytes"]==layout.shape_layout(list(reversed(shape)),self.typ)["bytes"],"quantized extent")
                pos+=v["bytes"]
            need(pos==self.payload.stat().st_size,"payload length")
            self.handle=safe_open(plain,framework="pt",device="cpu");self.requests=0;self.max_quant_read=0
        def tensor(self,name):
            if name not in self.records:return self.handle.get_tensor(name)
            v=self.records[name]
            with self.payload.open("rb") as f:f.seek(v["offset"]);b=f.read(v["bytes"])
            need(len(b)==v["bytes"] and s.sha(b)==v["sha256"],"expert byte pin")
            row=layout.shape_layout(list(reversed(v["shape"])),self.typ)["row_bytes"]
            arr=q.dequantize(np.frombuffer(b,dtype=np.uint8).reshape(v["shape"][0],row),self.enum)
            need(list(arr.shape)==v["shape"] and np.isfinite(arr).all(),"dequantized shape/finite")
            self.requests+=1;self.max_quant_read=max(self.max_quant_read,len(b))
            return torch.from_numpy(arr.copy()).bfloat16()
    return Quant
def engine_class(r):
    class Weights:
        def __init__(self,ck,prefix,proj,n):self.ck=ck;self.prefix=prefix;self.proj=proj;self.n=n
        def __len__(self):return self.n
        def __getitem__(self,e):
            need(type(e) is int and 0<=e<self.n,"local expert")
            return self.ck.tensor(f"{self.prefix}mlp.experts.{e}.{self.proj}_proj.weight")
    class Engine(r.RaggedEngine):
        def _load_layer(self,idx):
            if not self.ckpt.sparse[idx]:return super()._load_layer(idx)
            tensor,prefix=self.ckpt.tensor,f"model.layers.{idx}."
            l=r.eng.Layer(idx,True,tensor(prefix+"input_layernorm.weight"),tensor(prefix+"post_attention_layernorm.weight"),
                self._attention_projection(prefix,"q_proj",(self.attention_width,self.hidden)),
                self._attention_projection(prefix,"k_proj",(self.kv_width,self.hidden)),
                self._attention_projection(prefix,"v_proj",(self.kv_width,self.hidden)),
                self._attention_projection(prefix,"o_proj",(self.hidden,self.attention_width)),
                tensor(prefix+"self_attn.q_norm.weight"),tensor(prefix+"self_attn.k_norm.weight"))
            l.router=tensor(prefix+"mlp.router.gate.weight").float();l.bias=tensor(prefix+"mlp.expert_bias").float()
            ids=self.ckpt.mask[str(idx)]
            l.gate,l.up,l.down=(Weights(self.ckpt,prefix,p,len(ids)) for p in ("gate","up","down"))
            l.shared=tuple(tensor(prefix+f"mlp.shared_mlp.{p}_proj.weight") for p in ("gate","up","down"))
            lut=torch.full((self.ckpt.num_experts,),-1,dtype=torch.long);lut[ids]=torch.arange(len(ids));self.local[idx]=lut;l.denied=torch.where(lut>=0,0.,float("-inf")).float()
            return l
    return Engine
def main():
    a=argparse.ArgumentParser()
    for n in PINS:a.add_argument("--"+n,type=P,required=True)
    args=a.parse_args();r,s,layout=[load(getattr(args,n),n) for n in PINS]
    for f,h in layout.PINS.items():need(s.fsha(P(q.__file__).parent/f)==h,"quant source")
    from mtpruning.testing import tiny_checkpoint
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as d:
        p=P(d);src=tiny_checkpoint(p/"source",p/"none",vocab_size=256,seed=91,num_experts=8,top_k=2,layers=4);mask={"1":[0,2,4],"2":[1,2,3,5,6],"3":list(range(8))}
        out=[]
        for typ in ("Q4_0","Q8_0"):
            dest=p/typ;h=build(r,s,layout,src,dest,mask,typ);ck=checkpoint_class(r,s,layout)(dest,h)
            b=engine_class(r)(ck,1);need(ck.requests==0,"no eager expert dequantization")
            # Independent oracle: quantize and dequantize each kept original
            # BF16 tensor directly, not through the pack reader or local IDs.
            a=r.eng.Engine(r.eng.Checkpoint(src),1);a.set_mask({int(k):v for k,v in mask.items()})
            enum=getattr(gc.GGMLQuantizationType,typ)
            for l in a.layers:
                if not l.sparse:continue
                for i in mask[str(l.idx)]:
                    for weights in (l.gate,l.up,l.down):
                        t=weights[i];weights[i]=torch.from_numpy(q.dequantize(q.quantize(t.float().numpy(),enum),enum).copy()).bfloat16()
            fixed,route=r.compare_fixed(a,b,[4,8,15,16,23,42],[11,12,13,14,15,16,17,18])
            ga=a.generate([4,8,15],8,trace=True);gb=b.generate([4,8,15],8,trace=True)
            need(ga.tokens==gb.tokens and ga.finish==gb.finish and all(torch.equal(v,gb.trace[k]) for k,v in ga.trace.items()),"greedy trace")
            need(ck.requests>0,"actual selected expert access")
            before=ck.requests;key=next(iter(ck.records));v=ck.records[key]
            with ck.payload.open("r+b") as f:f.seek(v["offset"]);x=f.read(1);f.seek(v["offset"]);f.write(bytes([x[0]^1]))
            try:ck.tensor(key)
            except ValueError:pass
            else:raise AssertionError("corrupted expert read")
            need(ck.requests==before,"rejected access")
            out.append(dict(type=typ,packed_expert_tensors=len(ck.records),fixed_logits=fixed,route_hooks=route,greedy_trace=True,dequant_requests=ck.requests,max_quant_read=ck.max_quant_read,eager_expert_dequant_requests=0,tamper_rejected=True))
        print(json.dumps(dict(status="PASS_CPU_QUANT_RAGGED_ON_DEMAND",results=out,comparison="same quantized/dequantized weights, not original precision quality",native_quant_kernel="NOT_RUN",real_model="NOT_RUN")))
if __name__=="__main__":main()
