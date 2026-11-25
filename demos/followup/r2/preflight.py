"""Read-only checkpoint inventory; no model import, tensor allocation or network."""
import argparse, hashlib, json, re, struct
from pathlib import Path
from collections import Counter

DTYPES={'BOOL':1,'U8':1,'I8':1,'F8_E4M3':1,'F8_E5M2':1,'I16':2,'U16':2,'F16':2,'BF16':2,'I32':4,'U32':4,'F32':4,'I64':8,'U64':8,'F64':8}
def need(x,msg):
    if not x: raise ValueError(msg)
def unique(pairs):
    d={}
    for k,v in pairs:
        need(k not in d,'duplicate JSON key: '+k); d[k]=v
    return d
def load(p): return json.loads(p.read_text(encoding='utf-8'),object_pairs_hook=unique)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def header(p):
    with p.open('rb') as f:
        prefix=f.read(8); need(len(prefix)==8,'missing header prefix')
        n=struct.unpack('<Q',prefix)[0]; need(2<=n<=100_000_000,'invalid header size')
        raw=f.read(n); need(len(raw)==n,'incomplete header')
    h=json.loads(raw,object_pairs_hook=unique); need(isinstance(h,dict),'bad header')
    h.pop('__metadata__',None)
    spans=[]; tensors={}
    for name,t in h.items():
        shape=t['shape']; dtype=t['dtype']; a,b=t['data_offsets']
        need(dtype in DTYPES,'unsupported dtype '+str(dtype))
        need(isinstance(shape,list) and all(type(v) is int and v>=0 for v in shape),'bad shape')
        need(type(a) is int and type(b) is int and 0<=a<=b,'bad offset')
        elements=1
        for v in shape: elements*=v
        need(b-a==elements*DTYPES[dtype],'tensor byte mismatch: '+name)
        spans.append((a,b)); tensors[name]={'shape':shape,'dtype':dtype,'bytes':b-a}
    end=0
    for a,b in sorted(spans):
        need(a==end,'overlap or gap in tensor offsets'); end=b
    expected=8+n+end; actual=p.stat().st_size
    need(actual<=expected,'unexpected trailing bytes')
    return {'header_sha256':hashlib.sha256(raw).hexdigest(),'expected_bytes':expected,'observed_bytes':actual,'complete_size':actual==expected,'tensors':tensors}

def scan(root):
    root=Path(root); cfgp=root/'config.json'; idxp=root/'model.safetensors.index.json'
    cfg=load(cfgp); idx=load(idxp); wm=idx['weight_map']; need(isinstance(wm,dict) and wm,'empty index')
    groups={}
    for name,shard in wm.items():
        need(isinstance(shard,str) and re.fullmatch(r'[A-Za-z0-9_.-]+\.safetensors',shard) and not shard.startswith('.'),'unsafe shard name')
        groups.setdefault(shard,set()).add(name)
    records={}; inventory={}; missing=[]; incomplete=[]; errors={}
    for shard,expected in sorted(groups.items()):
        p=root/shard
        if not p.is_file(): missing.append(shard); continue
        try:
            h=header(p); ts=h.pop('tensors'); need(set(ts)==expected,'index/header tensor names differ')
            records[shard]=h
            if not h['complete_size']: incomplete.append(shard)
            for name,t in ts.items(): inventory[name]={'shard':shard,**t}
        except (ValueError,KeyError,TypeError,OSError) as e: errors[shard]=str(e)
    total=sum(t['bytes'] for t in inventory.values()); claimed=idx.get('metadata',{}).get('total_size')
    if not missing and not incomplete and not errors and claimed is not None and claimed!=total: errors['index']='metadata total_size mismatch'
    complete=not (missing or incomplete or errors)
    return {'status':'HEADER_INVENTORY_COMPLETE_NOT_CONTENT_VERIFIED' if complete else 'BLOCKED_INCOMPLETE_OR_INVALID','config_sha256':sha(cfgp),'index_sha256':sha(idxp),'config':{k:cfg.get(k) for k in ('model_type','architectures','num_hidden_layers','num_experts','num_local_experts','num_experts_per_tok','moe_intermediate_size','hidden_size','first_k_dense_replace')},'shards':records,'missing_shards':missing,'incomplete_shards':incomplete,'errors':errors,'tensor_count':len(inventory),'expected_tensor_count':len(wm),'inventory_bytes':total,'dtype_counts':dict(Counter(t['dtype'] for t in inventory.values())),'tensors':inventory,'payload_hash_verified':False,'model_load_verified':False}

def main():
    p=argparse.ArgumentParser(); p.add_argument('snapshot'); p.add_argument('report'); a=p.parse_args()
    report=scan(a.snapshot); out=Path(a.report)
    with out.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2,allow_nan=False)
    print(report['status']); return 0 if report['status'].startswith('HEADER_INVENTORY_COMPLETE') else 2
if __name__=='__main__': raise SystemExit(main())
