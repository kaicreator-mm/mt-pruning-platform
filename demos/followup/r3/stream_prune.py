"""D11/D12: explicit axis-0 Safetensors byte slicing, no tensor allocation.
Not an HY-V3 checkpoint adapter. A reviewed adapter must classify every tensor.
"""
import argparse, hashlib, json, math, os, re, shutil, struct, tempfile
from pathlib import Path
CHUNK=1024*1024
DT={'BOOL':1,'U8':1,'I8':1,'I16':2,'U16':2,'F16':2,'BF16':2,'I32':4,'U32':4,'F32':4,'I64':8,'U64':8,'F64':8}
def need(ok,msg):
    if not ok: raise ValueError(msg)
def unique(pairs):
    out={}
    for k,v in pairs:
        need(k not in out,'duplicate key '+k); out[k]=v
    return out
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'),object_pairs_hook=unique)
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(CHUNK),b''): h.update(b)
    return h.hexdigest()
def header(p):
    with p.open('rb') as f:
        raw=f.read(8); need(len(raw)==8,'bad prefix'); n=struct.unpack('<Q',raw)[0]; need(2<=n<=100_000_000,'bad header length'); raw=f.read(n); need(len(raw)==n,'short header')
    h=json.loads(raw,object_pairs_hook=unique); meta=h.pop('__metadata__',{}); need(isinstance(meta,dict) and all(isinstance(x,str) for x in meta.values()),'bad metadata'); spans=[]
    for k,t in h.items():
        s=t['shape']; a,b=t['data_offsets']; d=t['dtype']; need(d in DT and isinstance(s,list) and all(type(x) is int and x>=0 for x in s),'unsupported shape/dtype '+k); need(type(a) is int and type(b) is int and 0<=a<=b and b-a==math.prod(s)*DT[d],'bad tensor span '+k); spans.append((a,b))
    end=0
    for a,b in sorted(spans): need(a==end,'overlap/gap'); end=b
    need(p.stat().st_size==8+n+end,'incomplete/trailing shard'); return h,meta,8+n

def copy_span(src,dst,offset,length):
    src.seek(offset)
    while length:
        b=src.read(min(CHUNK,length)); need(bool(b),'short payload'); dst.write(b); length-=len(b)

def execute(root,out,plan):
    root=Path(root).resolve(); out=Path(out).resolve(); need(root!=out and root not in out.parents and not out.exists(),'output must be new and outside teacher'); need(out.parent.is_dir(),'output parent missing')
    idx=load(root/'model.safetensors.index.json'); cfg=load(root/'config.json'); wm=idx['weight_map']; roles=plan['tensors']; layers=plan['layers']; old=plan['original_count']; new=plan['keep_count']; top=plan['top_k']; field=plan['count_field']
    need(all(type(x) is int for x in (top,new,old)) and 1<=top<=new<=old,'bad budget'); need(cfg.get(field)==old and cfg.get(plan['top_k_field'])==top,'config count/top-k mismatch'); need(set(wm)==set(roles) and wm,'unclassified tensor inventory'); need(layers,'missing layers')
    for l,ids in layers.items(): need(isinstance(ids,list) and len(ids)==new and all(type(i) is int and 0<=i<old for i in ids) and ids==sorted(set(ids)),'bad ids '+l)
    groups={}; seen={l:set() for l in layers}
    for k,s in wm.items():
        need(isinstance(s,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.safetensors',s),'unsafe shard name'); groups.setdefault(s,set()).add(k)
        r=roles[k]; kind=r['role']; need(kind in ('copy','expert','router','bias'),'unknown role '+k)
        if kind=='copy': need(set(r)=={'role'},'bad copy role')
        else: need(set(r)=={'role','layer','axis'} and r['layer'] in layers and r['axis']==0,'only reviewed axis-0 layout supported'); seen[r['layer']].add(kind)
    required=set(plan['required_roles']); need({'expert','router'}<=required<={'expert','router','bias'},'bad required roles'); need(all(required<=s for s in seen.values()),'missing layer roles')
    # Source hashes bind the reviewed plan to actual files, including config/index.
    files=['config.json','model.safetensors.index.json',*sorted(groups)]; expected=plan['source_sha256']; need(set(expected)==set(files),'source hash inventory mismatch'); need(all(digest(root/f)==expected[f] for f in files),'source hash mismatch')
    stage=Path(tempfile.mkdtemp(prefix='.prune-',dir=out.parent)); total=0; before=0
    try:
        for shard,keys in sorted(groups.items()):
            h,meta,base=header(root/shard); need(set(h)==keys,'index/header mismatch'); nh={}; offsets={}; end=0
            for k,t in sorted(h.items()):
                a,b=t['data_offsets']; size=b-a; before+=size; shape=t['shape'][:]; r=roles[k]
                if r['role']=='copy': spans=[(a,size)]
                else:
                    need(shape and shape[0]==old,'wrong expert axis size '+k); unit=size//old; spans=[(a+i*unit,unit) for i in layers[r['layer']]]; shape[0]=new
                length=sum(n for _,n in spans); nh[k]={'dtype':t['dtype'],'shape':shape,'data_offsets':[end,end+length]}; offsets[k]=spans; end+=length
            if meta: nh['__metadata__']=meta
            raw=json.dumps(nh,sort_keys=True,separators=(',',':')).encode(); raw+=b' '*(-len(raw)%8)
            with (root/shard).open('rb') as src,(stage/shard).open('xb') as dst:
                dst.write(struct.pack('<Q',len(raw))); dst.write(raw)
                for k in sorted(offsets):
                    for a,n in offsets[k]: copy_span(src,dst,base+a,n)
            header(stage/shard); total+=end
        need(all(digest(root/f)==expected[f] for f in files),'source changed during copy')
        need(idx.get('metadata',{}).get('total_size',before)==before,'source total_size mismatch'); cfg[field]=new; dump(stage/'config.json',cfg); idx.setdefault('metadata',{})['total_size']=total; dump(stage/'model.safetensors.index.json',idx)
        # No tokenizer/sidecar handling: deliberately not a loadable full model export.
        receipt={'status':'PASS_EXPLICIT_TENSOR_REWRITE_ONLY','source_bytes':before,'output_bytes':total,'copy_chunk_limit_bytes':CHUNK,'source_sha256':expected,'output_sha256':{f:digest(stage/f) for f in files},'plan':plan,'full_model_reload_verified':False}
        dump(stage/'receipt.json',receipt); need(not out.exists(),'output appeared'); stage.rename(out); return receipt
    finally:
        if stage.exists(): shutil.rmtree(stage)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('output'); p.add_argument('plan'); a=p.parse_args(); print(json.dumps(execute(a.source,a.output,load(a.plan)),indent=2))
