"""D06-09: trace statistics, coactivation heuristic, sensitivity and validation-only selection."""
import math, random, re, itertools
DIRS=tuple(a+'-'+b for a,b in itertools.permutations(('zh','en','my'),2))
GROUPS={(d,p) for d in DIRS for p in ('prefill','decode')}
def need(x,msg):
    if not x: raise ValueError(msg)
def valid(x): return type(x) in (int,float) and math.isfinite(x) and x>=0

def profile(events,n,k):
    need(type(n) is int and type(k) is int and 1<=k<=n,'bad n/k'); result={}
    for e in events:
        l=e['layer']; g=(e['direction'],e['phase']); ids=e['ids']; weights=e['weights']; norms=e['output_norms']
        need(isinstance(l,str) and l and g in GROUPS,'unknown layer/group'); need(len(ids)==len(weights)==len(norms)==k and len(set(ids))==k and all(type(i) is int and 0<=i<n for i in ids),'bad selected ids'); need(all(valid(x) for x in weights+norms) and 0<sum(weights)<math.inf,'bad weights/norms')
        rec=result.setdefault(l,{}).setdefault(g,{'frequency':[0.]*n,'mass':[0.]*n,'saliency':[0.]*n,'pairs':{},'tokens':0})
        rec['tokens']+=1
        for i,w,v in zip(ids,weights,norms):
            rec['frequency'][i]+=1; rec['mass'][i]+=w; rec['saliency'][i]+=w*v
        for a,b in itertools.combinations(sorted(ids),2): rec['pairs'][a,b]=rec['pairs'].get((a,b),0)+1
    need(result and all(set(v)==GROUPS for v in result.values()),'missing direction/phase'); return result

def select(stats,n,keep,k,method,seed=42,pair_weight=.25):
    need(type(n) is int and type(k) is int and 1<=k<=n and type(keep) is int and k<=keep<=n and method in ('frequency','mass','saliency','coactivation','random'),'bad plan'); need(valid(pair_weight) and pair_weight<=1,'bad pair weight'); out={}
    for l,groups in sorted(stats.items()):
        need(set(groups)==GROUPS,'missing groups'); vectors=[]; edges={}
        key='saliency' if method=='coactivation' else ('mass' if method=='random' else method)
        for g,r in sorted(groups.items()):
            v=r[key]; s=sum(v); need(len(v)==n and all(valid(x) for x in v) and 0<s<math.inf,'invalid statistic'); vectors.append([x/s for x in v]); es=sum(r['pairs'].values())
            if es:
                for ab,c in r['pairs'].items(): edges[ab]=edges.get(ab,0)+c/es/len(groups)
        mean=[sum(v[i] for v in vectors)/len(vectors) for i in range(n)]
        if method=='random': ids=random.Random(str(seed)+':'+l).sample(range(n),keep)
        else:
            ids=[]
            while len(ids)<keep:
                i=max((i for i in range(n) if i not in ids),key=lambda i:(mean[i]+(pair_weight*sum(edges.get(tuple(sorted((i,j))),0) for j in ids) if method=='coactivation' else 0),-i)); ids.append(i)
        ids=sorted(ids); out[l]={'ids':ids,'min_group_score':min(sum(v[i] for i in ids) for v in vectors),'pair_coverage':sum(v for (a,b),v in edges.items() if a in ids and b in ids)}
    return out

def sensitivity(rows, budget, k):
    """Discrete DP on measured validation sensitivity estimates; additive proxy, not a quality guarantee."""
    need(type(budget) is int and type(k) is int and k>0 and rows,'invalid budget'); states={0:(0.,{})}
    for l,options in sorted(rows.items()):
        need(options and all(type(c) is int and c>=k and valid(v) for c,v in options.items()),'invalid sensitivity options'); new={}
        for used,(loss,m) in states.items():
            for count,v in sorted(options.items()):
                u=used+count; val=loss+v
                if u<=budget and (u not in new or val<new[u][0]): new[u]=(val,{**m,l:count})
        states=new; need(states,'infeasible budget')
    used,(loss,m)=min(states.items(),key=lambda s:(s[1][0],s[0])); return {'counts':m,'sum_experts':used,'estimated_additive_loss':loss,'status':'PLAN_ONLY_REQUIRES_JOINT_VALIDATION'}

def choose(reports, max_drop=1.):
    """Never accepts test data. Only complete validation reports may select a candidate."""
    need(valid(max_drop) and reports,'bad reports'); eligible=[]; ids=set(); baseline=None; identity=None
    for r in reports:
        name=r['id']; need(isinstance(name,str) and name and name not in ids,'duplicate id'); ids.add(name); need(r['split']=='validation','test/unknown split forbidden'); need(type(r['bytes']) is int and r['bytes']>0 and set(r['directions'])==set(DIRS),'missing report coverage'); binding=r['validation_identity']; need(isinstance(binding,str) and re.fullmatch(r'[0-9a-f]{64}',binding),'missing validation identity'); need(identity is None or identity==binding,'mixed validation identity'); identity=binding
        current={d:(m['n'],m['baseline'],m['baseline_errors']) for d,m in r['directions'].items()}; need(baseline is None or current==baseline,'mixed baseline'); baseline=current; ok=True
        for d,m in r['directions'].items():
            need(type(m['n']) is int and m['n']>0 and all(valid(m[x]) and m[x]<=100 for x in ('baseline','candidate')),'invalid score'); need(all(isinstance(m[x],dict) and set(m[x])=={'empty','truncated','number_mismatch'} and all(type(v) is int and 0<=v<=m['n'] for v in m[x].values()) for x in ('baseline_errors','candidate_errors')),'invalid errors'); ok &= m['candidate']-m['baseline']>=-max_drop and all(m['candidate_errors'][x]<=m['baseline_errors'][x] for x in m['baseline_errors'])
        if ok: eligible.append((r['bytes'],name))
    return {'status':'VALIDATION_CANDIDATE_ONLY' if eligible else 'FALLBACK_TO_TEACHER','candidate':min(eligible)[1] if eligible else None,'test_accessed':False,'production_acceptance':False}
