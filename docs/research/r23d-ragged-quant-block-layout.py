"""Quantized ragged expert-axis block layout; no model quantization or deployment.

GGML rows are stored with the innermost dimension first. Slice whole expert
slabs only if every row is block-aligned. Router and bias remain F32.
Pinned llama.cpp Python dequantizers give an independent numerical oracle.
"""
import argparse,hashlib,json,math,pathlib
P=pathlib.Path
TYPES={"F16":(1,2),"F32":(1,4),"Q4_0":(32,18),"Q8_0":(32,34),"Q4_K":(256,144),"Q6_K":(256,210)}
PINS={"constants.py":"3e637e339284da81df558c83b9aa4efb9ca3106cf983da15da67766a5d9118db","lazy.py":"dbcf8b7376284ab2d920103657cc52c1c62e93fec3856a08a1ae638394e6a7c1","quants.py":"f896726209ba8c91afd083dd87031c2d39dc4ef6a78e8a6c12763eeb8ed6ff84"}
def need(x,m):
    if not x:raise ValueError(m)
def shape_layout(dims,typ):
    need(typ in TYPES and isinstance(dims,(list,tuple)) and len(dims) in (1,2,3) and all(type(v) is int and 0<v<=2**32 for v in dims),"type/shape")
    block,size=TYPES[typ];need(dims[0]%block==0,"inner row is not block aligned")
    row=dims[0]//block*size
    return dict(row_bytes=row,bytes=row*math.prod(dims[1:]),block_elements=block,block_bytes=size)
def expert_plan(dims,typ,ids,top_k):
    need(len(dims)==3,"expert tensor rank")
    s=shape_layout(dims,typ);n=dims[-1]
    need(type(top_k) is int and 1<=top_k<=len(ids)<=n and ids==sorted(set(ids)) and all(type(i) is int and 0<=i<n for i in ids),"expert IDs")
    slab=s["row_bytes"]*dims[1];out=shape_layout([*dims[:-1],len(ids)],typ)
    return dict(source_bytes=s["bytes"],output_bytes=out["bytes"],expert_stride=slab,row_bytes=s["row_bytes"],ranges=[(i*slab,slab) for i in ids],output_dims=[*dims[:-1],len(ids)])
def slice_bytes(data,plan):
    need(len(data)==plan["source_bytes"],"payload length")
    out=b"".join(data[p:p+n] for p,n in plan["ranges"])
    need(len(out)==plan["output_bytes"],"output size")
    return out
def selftest():
    import numpy as np
    import gguf.constants as c
    import gguf.quants as q
    base=P(q.__file__).parent
    for name,h in PINS.items():need(hashlib.sha256((base/name).read_bytes()).hexdigest()==h,"gguf source pin")
    for n,v in TYPES.items():need(tuple(c.GGML_QUANT_SIZES[getattr(c.GGMLQuantizationType,n)])==v,"block spec "+n)
    rng=np.random.default_rng(23);ids=[0,2,4];checks=0
    for typ in ("Q4_0","Q8_0","Q4_K","Q6_K"):
        for dims in ([512,256,5],[256,512,5]):
            p=expert_plan(dims,typ,ids,2);enum=getattr(c.GGMLQuantizationType,typ);block,bsize=TYPES[typ]
            if typ in ("Q4_0","Q8_0"):
                x=rng.standard_normal((5,dims[1],dims[0]),dtype=np.float32)
                raw=q.quantize(x,enum).tobytes()
            else:
                # Valid finite synthetic K-block payloads, not weight quantization.
                b=rng.integers(0,256,size=(p["source_bytes"]//bsize,bsize),dtype=np.uint8)
                if typ=="Q4_K":b[:,0:4]=np.array([0,56,0,52],dtype=np.uint8)
                else:b[:,-2:]=np.array([0,56],dtype=np.uint8)
                raw=b.tobytes()
            out=slice_bytes(raw,p)
            a=np.frombuffer(raw,dtype=np.uint8).reshape(5,dims[1],p["row_bytes"])
            b=np.frombuffer(out,dtype=np.uint8).reshape(len(ids),dims[1],p["row_bytes"])
            need(np.array_equal(a[ids],b),"independent byte oracle")
            da=q.dequantize(a,enum);db=q.dequantize(b,enum)
            need(np.isfinite(da).all() and np.array_equal(da[ids],db),"dequantized numerical oracle")
            checks+=1
    def reject(fn):
        try:fn()
        except ValueError:return
        raise AssertionError("invalid layout accepted")
    reject(lambda:expert_plan([255,256,5],"Q4_K",ids,2))
    reject(lambda:expert_plan([256,256,5],"Q4_K",[0,0],2))
    reject(lambda:expert_plan([256,256,5],"Q4_K",[0,5],2))
    reject(lambda:expert_plan([256,256,5],"Q4_K",[0],2))
    reject(lambda:expert_plan([256,256,5],"UNKNOWN",ids,2))
    reject(lambda:slice_bytes(b"",expert_plan([256,256,5],"Q4_K",ids,2)))
    # Only routed FFNs; no total model estimate or quantized quality claim.
    payload={}
    for typ in ("F16","Q4_K","Q6_K","Q8_0"):
        per=2*shape_layout([2048,768],typ)["bytes"]+shape_layout([768,2048],typ)["bytes"]
        payload[typ]=4512*per
    print(json.dumps(dict(status="PASS_CPU_QUANT_BLOCK_AXIS_ORACLE",numpy=np.__version__,tensor_checks=checks,negative=6,actual_quantized_fixture_types=["Q4_0","Q8_0"],synthetic_valid_block_types=["Q4_K","Q6_K"],routed_ffn_only_bytes=payload,actual_model_quantized="NOT_RUN",native_runtime="NOT_RUN")))
if __name__=="__main__":selftest()
