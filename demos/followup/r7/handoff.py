"""Offline D01b validation-only handoff; never accepts a candidate test record."""
import argparse, hashlib, json, math, re, unicodedata
from pathlib import Path
from collections import Counter

DIRS = ('zh-my','my-zh','zh-en','en-zh','en-my','my-en')
SPLITS = ('calibration','validation','test')


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def corpus(manifest_path, manifest_sha, corpus_path):
    need(sha(manifest_path) == manifest_sha, 'manifest hash mismatch')
    m = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    need(m['corpus'] == 'FLORES-200' and tuple(m['directions']) == DIRS, 'unsupported corpus or directions')
    need(sha(corpus_path) == m['corpus_sha256'], 'corpus hash mismatch')
    sizes = m['sizes_per_direction']
    need(set(sizes) == set(SPLITS) and all(type(v) is int and v > 0 for v in sizes.values()), 'bad split sizes')
    need(set(m['indices']) == {'zh','en','my'}, 'bad languages')
    expected = {}
    line_groups = {s:set() for s in SPLITS}
    for s in SPLITS:
        fs = 'dev' if s == 'calibration' else 'devtest'
        for d in DIRS:
            src, tgt = d.split('-')
            ids = m['indices'][src][s]
            need(len(ids) == sizes[s] and len(set(ids)) == len(ids) and all(type(i) is int and 0 <= i < m['split_sizes'][fs] for i in ids), 'bad manifest indices')
            line_groups[s].update(ids)
            expected.update({f'{s}:{d}:{i}':(s,d,src,tgt,fs,i) for i in ids})
    need(not line_groups['validation'] & line_groups['test'], 'parallel devtest line crosses splits')
    records = {}; forms = {}; aligned = {}
    for r in lines(corpus_path):
        i = r['id']
        need(i in expected and i not in records, 'unexpected/duplicate corpus ID')
        need(tuple(r[k] for k in ('split','direction','src_lang','tgt_lang','flores_split','line')) == expected[i], 'corpus identity mismatch')
        need(isinstance(r['prompt_ids'],list) and r['prompt_ids'] and all(type(t) is int and t >= 0 for t in r['prompt_ids']), 'invalid prompt tokens')
        for lang, field in ((r['src_lang'],'source'),(r['tgt_lang'],'reference')):
            v = norm(r[field]); need(bool(v), 'empty corpus text')
            key = (lang,v); previous = forms.setdefault(key,r['split']); need(previous == r['split'], 'normalized text crosses splits')
            key = (r['flores_split'],r['line'],lang); need(aligned.setdefault(key,r[field]) == r[field], 'parallel line text mismatch')
        records[i] = r
    need(set(records) == set(expected) and len(records) == m['records'], 'incomplete corpus')
    return m, records


def records(path, m, contexts, keep, *, candidate):
    rr = lines(path); need(rr and rr[0].get('type') == 'run', 'missing run header')
    h = rr[0]
    need(h.get('corpus_sha256') == m['corpus_sha256'] and h.get('num_experts') == keep and 'masked_keep' in h and h['masked_keep'] is None, 'wrong run identity')
    need(isinstance(h.get('splits'),list) and 'validation' in h['splits'] and set(h['splits']) <= set(SPLITS), 'wrong split scope')
    need(type(h.get('max_new_tokens')) is int and h['max_new_tokens'] > 0 and isinstance(h.get('decoding'),str) and h['decoding'], 'bad decoding contract')
    need(not candidate or 'test' not in h['splits'], 'candidate test already exposed')
    out = {}; seen = set()
    for r in rr[1:]:
        i = r['id']; need(r.get('type') == 'sentence' and i in contexts and i not in seen, 'invalid/duplicate run record'); seen.add(i)
        c = contexts[i]
        need(all(r.get(k) == c[k] for k in ('split','direction','line')), 'run sample mismatch')
        need(c['split'] in h['splits'], 'undeclared split')
        need(not candidate or c['split'] != 'test', 'candidate test already exposed')
        if c['split'] != 'validation': continue
        need(r.get('prompt_tokens') == len(c['prompt_ids']) and isinstance(r.get('tokens'),list) and all(type(t) is int and t >= 0 for t in r['tokens']), 'bad token record')
        need(type(r.get('new_tokens')) is int and len(r['tokens']) == r['new_tokens'] <= h['max_new_tokens'] and r.get('finish') in ('eos','length') and isinstance(r.get('text'),str), 'bad generation record')
        need(r['finish'] != 'length' or r['new_tokens'] == h['max_new_tokens'], 'length finish inconsistent')
        out[i] = r
    need(set(out) == {i for i,c in contexts.items() if c['split']=='validation'}, 'incomplete validation records')
    return h,out


def digits(s):
    fold = str.maketrans('၀၁၂၃၄၅၆၇၈၉０１２３４５６７８９','0123456789'*2)
    return sorted(re.findall(r'\d+',s.translate(fold)))


def diagnostic(rs, cs):
    # Same Myanmar/full-width digit-fold and sorted digit-run heuristic as D01b; not a semantic quality judgment.
    return {'empty':sum(not r['text'].strip() for r in rs), 'truncated':sum(r['finish']!='eos' for r in rs), 'digits':sum(digits(r['text'])!=digits(c['reference']) for r,c in zip(rs,cs))}


def evaluate(manifest_path, manifest_sha, corpus_path, baseline, candidates, metric, max_drop=1.0):
    """Metric callable accepts hypotheses and references. CLI requires pinned sacrebleu."""
    need(type(max_drop) in (int,float) and math.isfinite(max_drop) and max_drop >= 0, 'invalid quality threshold')
    m,ctx = corpus(manifest_path,manifest_sha,corpus_path)
    need(candidates and all(type(k) is int and 8 <= k < 128 for k in candidates), 'invalid candidate keep')
    h,b = records(baseline,m,ctx,128,candidate=False)
    result = {}
    for k,p in sorted(candidates.items()):
        ch,c = records(p,m,ctx,k,candidate=True)
        need(all(ch[x] == h[x] for x in ('max_new_tokens','decoding')), 'decoding mismatch')
        directions = {}
        for d in DIRS:
            ids = sorted(i for i in b if ctx[i]['direction']==d)
            refs = [ctx[i]['reference'] for i in ids]; cs = [ctx[i] for i in ids]
            a = [b[i] for i in ids]; z = [c[i] for i in ids]
            scores = [metric([r['text'] for r in rs],refs) for rs in (a,z)]
            need(all(type(v) in (int,float) and math.isfinite(v) and 0 <= v <= 100 for v in scores), 'invalid score')
            da,dz = diagnostic(a,cs),diagnostic(z,cs); delta = scores[1]-scores[0]
            directions[d] = {'n':len(ids),'baseline':scores[0],'candidate':scores[1],'delta':delta,'baseline_flags':da,'candidate_flags':dz,'pass':delta >= -max_drop and all(dz[f] <= da[f] for f in da)}
        result[str(k)] = {'records_sha256':sha(p),'directions':directions,'pass':all(v['pass'] for v in directions.values())}
    passed = [int(k) for k,v in result.items() if v['pass']]
    return {'status':'PROVISIONAL_VALIDATION_SELECTION_ONLY','chosen_keep':min(passed) if passed else None,'fallback_to_teacher':not passed,'candidates':result,'manifest_sha256':manifest_sha,'corpus_sha256':m['corpus_sha256'],'baseline_sha256':sha(baseline),'max_drop':max_drop,'checkpoint_provenance_verified':False,'test_execution_authorized':False,'production_acceptance':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest','manifest-sha','corpus','baseline','out'): p.add_argument('--'+flag,required=True)
    p.add_argument('--candidate',action='append',required=True,metavar='KEEP=PATH'); p.add_argument('--max-drop',type=float,default=1.0)
    a=p.parse_args(); c={}
    for item in a.candidate:
        k,v=item.split('=',1); k=int(k); need(k not in c,'duplicate candidate keep'); c[k]=v
    import sacrebleu
    need(sacrebleu.__version__=='2.6.0','expected pinned sacrebleu 2.6.0')
    metric=sacrebleu.metrics.CHRF(word_order=2)
    result=evaluate(a.manifest,a.manifest_sha,a.corpus,a.baseline,c,lambda hy,ref:float(metric.corpus_score(hy,[ref]).score),a.max_drop)
    result['metric']='sacrebleu 2.6.0 CHRF(word_order=2)'; result['metric_signature']=str(metric.get_signature())
    with Path(a.out).open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)


if __name__=='__main__': main()
