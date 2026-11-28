"""Analytic expert-cache resource planner for the fixed HY-V3 geometry.
No actual free-memory, latency, quality or full-model fit claim. CLI selftest
and explicit user-supplied non-cache reservations only.
"""
import argparse,json,math
LAYERS=47
HIDDEN=2048
WIDTH=768
TOP=8
ORIGINAL=128
BUDGETS=(64,80,96,112,128)
MIB=1<<20
GIB=1<<30
MATRIX=HIDDEN*WIDTH*2
EXPERT=3*MATRIX
COUNTS={64:7,80:12,96:3,112:24,128:1}
def need(x,m):
    if not x:raise ValueError(m)
def plan(counts,total,reserve,noncache,transient):
    need(set(counts)<=set(BUDGETS) and all(type(k) is int and type(v) is int and v>0 for k,v in counts.items()) and sum(counts.values())==LAYERS,"layer budgets")
    need(all(type(v) is int and v>=0 for v in (total,reserve,noncache,transient)) and total>0,"explicit byte budgets")
    instances=sum(k*v for k,v in counts.items());remaining=total-reserve-noncache-transient
    cap=max(0,min(remaining,instances*EXPERT))
    # Equal per-layer cache byte quotas. Whole expert counts are accounting
    # equivalents; actual matrix LRU may retain partial experts.
    low,extra=divmod(cap,LAYERS)
    layer_cap=low//EXPERT
    totals={name:instances*3*HIDDEN*WIDTH//block*size for name,(block,size) in {"BF16":(1,2),"Q4_0":(32,18),"Q8_0":(32,34),"Q4_K":(256,144),"Q6_K":(256,210)}.items()}
    need(cap<=instances*EXPERT,"cache budget")
    return dict(status="ANALYTIC_EXPERT_CACHE_PLAN_NOT_MEMORY_QUALIFICATION",sparse_layers=LAYERS,expert_instances=instances,one_bf16_matrix_bytes=MATRIX,one_bf16_expert_bytes=EXPERT,one_layer_top8_payload_bytes=TOP*EXPERT,all_layers_one_top8_set_bytes=LAYERS*TOP*EXPERT,full_dequantized_expert_payload_bytes=instances*EXPERT,explicit_total_bytes=total,noncache_reservation_bytes=noncache,transient_reservation_bytes=transient,safety_reservation_bytes=reserve,accounting_deficit_bytes=max(0,-remaining),cache_cap_bytes=cap,whole_expert_equivalent_global=cap//EXPERT,equal_layer_quota_floor_bytes=low,equal_layer_extra_one_byte_count=extra,equal_layer_whole_expert_equivalent_floor=layer_cap,routed_ffn_only_quantized_payload_bytes=totals,limits="All reservations supplied externally, not measurements. No RSS/VRAM guarantee; includes neither allocator/OS cache nor unaccounted tensors. Top8 footprint assumes one token, one set per layer, not prefill or measured working set. Cache is matrix-level, not atomic expert-level.")
def selftest():
    x=plan(COUNTS,16*GIB,2*GIB,10*GIB,2*GIB)
    need(x["expert_instances"]==4512 and MATRIX==3*MIB and EXPERT==9*MIB,"geometry")
    need(x["one_layer_top8_payload_bytes"]==72*MIB and x["all_layers_one_top8_set_bytes"]==3384*MIB,"top8 footprint")
    need(x["cache_cap_bytes"]==2*GIB and x["equal_layer_whole_expert_equivalent_floor"]==4,"illustrative partition")
    need(x["routed_ffn_only_quantized_payload_bytes"]["Q4_K"]==11975786496 and x["routed_ffn_only_quantized_payload_bytes"]["Q6_K"]==17464688640,"quant block layout")
    y=plan(COUNTS,8*GIB,2*GIB,8*GIB,1*GIB);need(y["cache_cap_bytes"]==0 and y["accounting_deficit_bytes"]==3*GIB,"deficit")
    z=plan(COUNTS,128*GIB,0,0,0);need(z["cache_cap_bytes"]==4512*EXPERT,"full cache cap")
    for args in (({96:46},GIB,0,0,0),(COUNTS,GIB,-1,0,0),(COUNTS,True,0,0,0)):
        try:plan(*args)
        except ValueError:pass
        else:raise AssertionError("invalid budget")
    print(json.dumps(dict(status="PASS_ANALYTIC_RESOURCE_PLAN",negative=3,illustration=x,actual_resource_admission="NOT_RUN")))
def main():
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true")
    for n in ("total","reserve","noncache","transient"):p.add_argument("--"+n,type=int)
    q=p.parse_args()
    if q.selftest:selftest();return
    need(all(getattr(q,n) is not None for n in ("total","reserve","noncache","transient")),"supply all byte budgets")
    print(json.dumps(plan(COUNTS,q.total,q.reserve,q.noncache,q.transient)))
if __name__=="__main__":main()
