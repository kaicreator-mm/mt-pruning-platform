"""Research-only compact HF tensor checkpoint and exact HY-V3 CPU adapter.

Full router/bias; physically absent removed FFNs; original-ID routing and trace.
Only tiny synthetic pack construction exposed. No production model admission.
"""
import argparse,hashlib,json,math,pathlib,re,tempfile
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
import mtpruning.engine as eng
P=pathlib.Path
ENGINE="2af43ca1710b029fdde0d1135ea9879fe8defe56"
TESTING="e8979ac3cd42b739dd93448f3384c188c08568c4"
EXPERT=re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
def need(x,m):
    if not x:raise ValueError(m)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def fsha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def read(p):
    need(p.is_file() and not p.is_symlink(),"regular file");return p.read_bytes()
def validate(cfg,mask):
    eng.check_config(cfg);n=cfg["num_experts"];k=cfg["num_experts_per_tok"];sp=eng.sparse_layers_from_config(cfg)
    need(type(n) is int and type(k) is int and 1<=k<=n and set(mask)=={str(i) for i,v in enumerate(sp) if v},"mask inventory")
    need(all(isinstance(v,list) and k<=len(v)<=n and v==sorted(set(v)) and all(type(i) is int and 0<=i<n for i in v) for v in mask.values()),"mask IDs")
    return sp
def mapping(cfg,mask):
    sp=validate(cfg,mask);out={}
    for name in eng.expected_keys(cfg,sp):
        m=EXPERT.fullmatch(name)
        if not m:out[name]=name;continue
        l,e=m[1],int(m[2]);ids=mask[l]
        if e in ids:
            dst=f"model.layers.{l}.mlp.experts.{ids.index(e)}.{m[3]}_proj.weight"
            need(dst not in out,"duplicate compact key");out[dst]=name
    return out
def build_fixture(source,dest,mask):
    need(not dest.exists() and not dest.is_symlink(),"new fixture destination")
    prov=json.loads(read(source/"checkpoint_provenance.json"));need(prov["kind"]=="synthetic","synthetic source only")
    need(all(fsha(source/name)==h for name,h in prov["files"].items()),"source fixture identity")
    need(sum((source/n).stat().st_size for n in prov["files"])<=64<<20,"tiny fixture only")
    ck=eng.Checkpoint(source);need(ck.kept is None,"unpruned source")
    mp=mapping(ck.config,mask);tensors={k:ck.tensor(v).clone().contiguous() for k,v in sorted(mp.items())}
    dest.mkdir();(dest/"config.json").write_bytes(read(source/"config.json"))
    save_file(tensors,str(dest/"compact.safetensors"))
    manifest=dict(format="r23-ragged-hf-full-router-v1",source_kind="synthetic",engine_blob=ENGINE,source_provenance_sha256=sha(read(source/"checkpoint_provenance.json")),config_sha256=fsha(dest/"config.json"),payload_sha256=fsha(dest/"compact.safetensors"),kept=mask,mask_sha256=sha(canon(mask)),tensor_count=len(tensors))
    (dest/"ragged_manifest.json").write_bytes(canon(manifest)+b"\n")
    return manifest
class RaggedCheckpoint:
    def __init__(self,path,manifest_sha):
        self.path=P(path);need(self.path.is_dir() and not self.path.is_symlink(),"checkpoint directory")
        need({p.name for p in self.path.iterdir()}=={"config.json","compact.safetensors","ragged_manifest.json"},"file inventory")
        raw=read(self.path/"ragged_manifest.json");m=json.loads(raw)
        need(sha(raw)==manifest_sha and raw==canon(m)+b"\n" and m["format"]=="r23-ragged-hf-full-router-v1" and m["source_kind"]=="synthetic" and m["engine_blob"]==ENGINE,"manifest pin")
        cfgraw=read(self.path/"config.json");need(sha(cfgraw)==m["config_sha256"],"config digest");self.config=json.loads(cfgraw)
        self.mask=m["kept"];need(sha(canon(self.mask))==m["mask_sha256"],"mask digest");self.sparse=validate(self.config,self.mask)
        self.sparse_layers=[i for i,v in enumerate(self.sparse) if v];self.num_experts=self.config["num_experts"];self.kept=None
        mp=mapping(self.config,self.mask);data=self.path/"compact.safetensors"
        need(data.is_file() and not data.is_symlink() and fsha(data)==m["payload_sha256"],"payload digest")
        self.handle=safe_open(data,framework="pt",device="cpu")
        need(set(self.handle.keys())==set(mp) and len(mp)==m["tensor_count"],"physical tensor inventory")
        self.source_map=mp
        hidden=self.config["hidden_size"];width=self.config["moe_intermediate_size"]
        for l in self.sparse_layers:
            prefix=f"model.layers.{l}.mlp."
            router=self.tensor(prefix+"router.gate.weight");bias=self.tensor(prefix+"expert_bias")
            need(tuple(router.shape)==(self.num_experts,hidden) and tuple(bias.shape)==(self.num_experts,) and router.dtype==torch.bfloat16 and bias.dtype==torch.float32,"full router shape/dtype")
            for i in range(len(self.mask[str(l)])):
                for name,shape in (("gate",(width,hidden)),("up",(width,hidden)),("down",(hidden,width))):
                    t=self.tensor(prefix+f"experts.{i}.{name}_proj.weight")
                    need(tuple(t.shape)==shape and t.dtype==torch.bfloat16,"compact FFN shape/dtype")
    def tensor(self,name):return self.handle.get_tensor(name)
    def origin_ids(self,layer):
        # Routing hook receives ORIGINAL IDs, as in the unpruned engine.
        return list(range(self.num_experts))
class RaggedEngine(eng.Engine):
    def __init__(self,checkpoint,threads=1):
        need(isinstance(checkpoint,RaggedCheckpoint),"ragged checkpoint required")
        self.local={}
        super().__init__(checkpoint,threads)
    def _load_layer(self,idx):
        if not self.ckpt.sparse[idx]:return super()._load_layer(idx)
        tensor,prefix=self.ckpt.tensor,f"model.layers.{idx}."
        layer=eng.Layer(idx,True,tensor(prefix+"input_layernorm.weight"),tensor(prefix+"post_attention_layernorm.weight"),
            self._attention_projection(prefix,"q_proj",(self.attention_width,self.hidden)),
            self._attention_projection(prefix,"k_proj",(self.kv_width,self.hidden)),
            self._attention_projection(prefix,"v_proj",(self.kv_width,self.hidden)),
            self._attention_projection(prefix,"o_proj",(self.hidden,self.attention_width)),
            tensor(prefix+"self_attn.q_norm.weight"),tensor(prefix+"self_attn.k_norm.weight"))
        layer.router=tensor(prefix+"mlp.router.gate.weight").float();layer.bias=tensor(prefix+"mlp.expert_bias").float()
        ids=self.ckpt.mask[str(idx)]
        experts=[[tensor(f"{prefix}mlp.experts.{e}.{name}_proj.weight") for name in ("gate","up","down")] for e in range(len(ids))]
        layer.gate,layer.up,layer.down=([e[i] for e in experts] for i in range(3))
        layer.shared=tuple(tensor(prefix+f"mlp.shared_mlp.{name}_proj.weight") for name in ("gate","up","down"))
        lut=torch.full((self.ckpt.num_experts,),-1,dtype=torch.long);lut[ids]=torch.arange(len(ids));self.local[idx]=lut
        layer.denied=torch.where(lut>=0,0.,float("-inf")).float()
        return layer
    def set_mask(self,kept):
        # Fixed physical inventory cannot be unmasked or changed silently.
        need(kept=={int(k):v for k,v in self.ckpt.mask.items()},"physical mask immutable")
    def _moe(self,layer,x,phase):
        top,w=self._route(layer,x,phase)
        if self.on_route is not None:self.on_route(layer.idx,phase,top,w)
        local=self.local[layer.idx][top];need(bool((local>=0).all()),"removed expert routed")
        out=eng.expert_combine(x,local,w,layer.gate,layer.up,layer.down)
        gate,up,down=layer.shared
        return out+F.linear(F.silu(F.linear(x,gate))*F.linear(x,up),down)
def compare_fixed(a,b,prompt,prefix):
    def capture(e):
        route=[];e.on_route=lambda l,p,i,w:route.append((l,p,i.clone(),w.clone()))
        try:
            cache=e.new_cache(len(prompt)+len(prefix)+1);z=[]
            with torch.inference_mode():
                for j,ids in enumerate([prompt]+[[t] for t in prefix]):
                    z.append(e.forward(ids,cache,"prefill" if j==0 else "decode").clone())
            return z,route,cache.k[:,:,:cache.len].clone(),cache.v[:,:,:cache.len].clone()
        finally:e.on_route=None
    x,y=capture(a),capture(b);need(len(x[0])==len(y[0]) and all(torch.equal(i,j) for i,j in zip(x[0],y[0])),"fixed logits")
    need(len(x[1])==len(y[1]) and all(i[:2]==j[:2] and torch.equal(i[2],j[2]) and torch.equal(i[3],j[3]) for i,j in zip(x[1],y[1])),"exact original routes/weights")
    need(torch.equal(x[2],y[2]) and torch.equal(x[3],y[3]),"exact KV")
    return len(x[0]),len(x[1])
def selftest():
    from mtpruning.testing import tiny_checkpoint
    base=P(eng.__file__).parent
    need(blob(read(base/"engine.py"))==ENGINE and blob(read(base/"testing.py"))==TESTING,"reference source")
    torch.set_num_threads(1)
    count=0;steps=0;routes=0;gens=0;ties=0
    with tempfile.TemporaryDirectory() as d:
        p=P(d);src=tiny_checkpoint(p/"source",p/"none",vocab_size=256,seed=23,num_experts=8,top_k=2,layers=4)
        mask={"1":[0,2,4],"2":[1,2,3,5,6],"3":list(range(8))}
        dst=p/"compact";m=build_fixture(src,dst,mask);mh=sha(read(dst/"ragged_manifest.json"));ck=RaggedCheckpoint(dst,mh)
        original=eng.Checkpoint(src);a=eng.Engine(original,1);a.set_mask({int(l):v for l,v in mask.items()});b=RaggedEngine(ck,1)
        need(set(ck.source_map)<set(eng.expected_keys(ck.config,ck.sparse)),"removed physical tensor keys")
        for k,v in ck.source_map.items():
            need(torch.equal(ck.tensor(k),original.tensor(v)),"independent tensor oracle");count+=1
        for tie in (False,True):
            if tie:
                # Only replace in-memory router tensors, never source mmap.
                for e in (a,b):
                    for l in e.layers:
                        if l.sparse:l.router=torch.zeros_like(l.router);l.bias=torch.zeros_like(l.bias)
            for prompt in ([4,8,15,16,23,42],[3,7],[1,7,9,17,19,21,31,37]):
                x,y=compare_fixed(a,b,prompt,[11,12,13,14,15,16,17,18]);steps+=x;routes+=y
                g=a.generate(prompt,8,trace=True);h=b.generate(prompt,8,trace=True)
                need(g.tokens==h.tokens and g.finish==h.finish and all(torch.equal(v,h.trace[k]) for k,v in g.trace.items()),"free-running exact trace");gens+=1
                if tie:ties+=1
        def reject(f):
            try:f()
            except (ValueError,OSError):return
            raise AssertionError("bad fixture accepted")
        reject(lambda:b.set_mask(None));reject(lambda:RaggedCheckpoint(dst,"0"*64))
        reject(lambda:build_fixture(src,dst,mask))
        reject(lambda:validate(ck.config,dict(mask,**{"1":[0]})))
        reject(lambda:validate(ck.config,dict(mask,**{"1":[0,0]})))
        payload=dst/"compact.safetensors";raw=read(payload);payload.write_bytes(raw[:-1]+bytes([raw[-1]^1]));reject(lambda:RaggedCheckpoint(dst,mh))
    print(json.dumps(dict(status="PASS_CPU_RAGGED_REFERENCE_ENGINE",torch=str(torch.__version__),physical_tensor_checks=count,layer_counts=[3,5,8],fixed_logit_checks=steps,route_hook_checks=routes,greedy_trace_checks=gens,deliberate_tie_prompts=ties,negative=6,actual_teacher="NOT_RUN",native_llama_cpp="NOT_RUN")))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true");q=p.parse_args();need(q.selftest,"synthetic only");selftest()
