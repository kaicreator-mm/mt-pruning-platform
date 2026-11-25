"""Read-only D01b aggregate adapter and quality gate; no model execution."""
import hashlib, json, math, re
from collections import Counter
from pathlib import Path
import numpy as np

DIRS = ('zh-my', 'my-zh', 'zh-en', 'en-zh', 'en-my', 'my-en')
PHASES = ('prefill', 'decode')
FAILURE_FLAGS = frozenset(('empty','truncated','untranslated_copy','wrong_script','digits_differ_from_reference','length_ratio_outlier','degenerate_repetition'))

def need(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()

def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def by_id(records):
    result = {}
    for r in records:
        i = r['id']
        need(isinstance(i, str) and i and i not in result, 'duplicate/invalid sample ID')
        result[i] = r
    return result

def corpus_group(corpus, split, expected_n):
    need(split in ('calibration', 'validation', 'test') and type(expected_n) is int and expected_n > 0, 'bad split/size')
    by_id(corpus)
    group = by_id([r for r in corpus if r['split'] == split])
    need(Counter(r['direction'] for r in group.values()) == Counter(dict.fromkeys(DIRS, expected_n)), 'corpus coverage mismatch')
    return group

def adapt(corpus_path, records_path, stats_base, *, corpus_sha256, layer_ids, n_experts, top_k, expected_n):
    """Convert complete original-teacher aggregates into R1 plan input. Caller must prove teacher/run identity.

    Detects missing calibration rows and missing/overwritten aggregates by counts;
    does not prove sample-level provenance or make a quality decision.
    """
    need(sha(corpus_path) == corpus_sha256, 'corpus hash mismatch')
    need(type(n_experts) is int and type(top_k) is int and 1 <= top_k <= n_experts, 'bad expert counts')
    need(layer_ids and all(type(i) is int and i >= 0 for i in layer_ids) and sorted(set(layer_ids)) == layer_ids, 'invalid sparse layer list')
    expected = corpus_group(rows(corpus_path), 'calibration', expected_n)
    records = rows(records_path)
    need(records and records[0].get('type') == 'run' and all(r.get('type') == 'sentence' for r in records[1:]), 'invalid run header/records')
    h = records[0]
    need(h.get('corpus_sha256') == corpus_sha256 and h.get('num_experts') == n_experts and 'masked_keep' in h and h['masked_keep'] is None and 'calibration' in h.get('splits', []), 'not original unmasked calibration run')
    all_actual = by_id(records[1:]); actual = {i:r for i,r in all_actual.items() if r['split'] == 'calibration'}
    need(set(actual) == set(expected), 'missing/extra calibration records')
    tokens = {p: np.zeros(len(DIRS), dtype=np.int64) for p in PHASES}
    for i, r in actual.items():
        e = expected[i]; d = e['direction']; j = DIRS.index(d)
        need(r['direction'] == d and r.get('line') == e.get('line'), 'record direction/line mismatch')
        need(type(r['prompt_tokens']) is int and r['prompt_tokens'] == len(e['prompt_ids']) > 0, 'prompt token mismatch')
        need(type(r['new_tokens']) is int and r['new_tokens'] >= 0 and isinstance(r['tokens'], list) and len(r['tokens']) == r['new_tokens'], 'generated token mismatch')
        tokens['prefill'][j] += r['prompt_tokens']; tokens['decode'][j] += r['new_tokens']
    jp = Path(str(stats_base) + '.json'); npz = Path(str(stats_base) + '.npz')
    summary = json.loads(jp.read_text(encoding='utf-8'))
    need(summary['directions'] == list(DIRS) and summary['layers'] == layer_ids, 'stats axis identity mismatch')
    need(summary['sentences'] == dict.fromkeys(DIRS, expected_n), 'stats sentence coverage mismatch')
    need(summary['tokens'] == {d:{p:int(tokens[p][j]) for p in PHASES} for j,d in enumerate(DIRS)}, 'stats summary token mismatch')
    shape = (len(DIRS), len(layer_ids), n_experts); result = {str(l):{d:{} for d in DIRS} for l in layer_ids}
    with np.load(npz, allow_pickle=False) as data:
        need(set(data.files) == {'sentences', *(f'{k}_{p}' for k in ('count','mass','tokens') for p in PHASES)}, 'unknown/missing arrays')
        sent = data['sentences']; need(sent.dtype.kind in 'iu' and sent.shape == (len(DIRS),) and np.array_equal(sent, np.full(len(DIRS), expected_n)), 'stats array sentence mismatch')
        for p in PHASES:
            t, c, m = (data[f'{k}_{p}'] for k in ('tokens','count','mass'))
            need(t.dtype.kind in 'iu' and np.array_equal(t, tokens[p]) and np.all(t > 0), 'stats array token mismatch or empty phase')
            need(c.dtype.kind in 'iu' and c.shape == shape and np.all(c >= 0) and np.all(c <= t[:,None,None]) and np.all(c.sum(-1) == t[:,None] * top_k), 'invalid expert count coverage')
            need(m.dtype.kind == 'f' and m.shape == shape and np.isfinite(m).all() and np.all(m >= 0) and np.isfinite(m.sum(-1)).all() and np.all(m.sum(-1) > 0) and np.all(m[c == 0] == 0), 'invalid routing mass')
            for j,d in enumerate(DIRS):
                for k,l in enumerate(layer_ids):
                    result[str(l)][d][p] = m[j,k].tolist()
    return {'status':'AGGREGATE_COVERAGE_CHECKED_NOT_PROVENANCE', 'profiles':result, 'expert_count':n_experts, 'top_k':top_k, 'hashes':{'corpus':corpus_sha256, 'records':sha(records_path), 'stats_json':sha(jp), 'stats_npz':sha(npz)}}

def quality_gate(table, split, candidate, expected_n, max_drop=1.0):
    """Independent gate for D01b report rows; never trusts report's gate_pass field."""
    need(split in ('validation','test') and isinstance(candidate,str) and candidate and type(expected_n) is int and expected_n > 0, 'invalid gate scope')
    need(type(max_drop) in (float,int) and math.isfinite(max_drop) and max_drop >= 0, 'invalid drop threshold')
    group = [r for r in table if r['split'] == split]
    need(Counter(r['direction'] for r in group) == Counter(dict.fromkeys(DIRS,1)), 'missing/duplicate quality directions')
    verdict = {}
    for r in group:
        need(r['n'] == expected_n and candidate in r['candidates'], 'incomplete candidate group')
        c = r['candidates'][candidate]; a, b = r['baseline_chrf_pp'], c['chrf_pp']
        need(all(type(v) in (float,int) and math.isfinite(v) and 0 <= v <= 100 for v in (a,b)), 'invalid chrF++')
        delta = b-a
        need(type(c['delta']) in (float,int) and math.isfinite(c['delta']) and math.isclose(c['delta'], delta, abs_tol=1e-9), 'inconsistent score delta')
        flags = ('empty','truncated','digits_differ_from_reference')
        errors = (r.get('baseline_flags'), c.get('flags'))
        need(all(isinstance(f,dict) and set(f) <= FAILURE_FLAGS and all(type(v) is int and 0 <= v <= expected_n for v in f.values()) for f in errors), 'invalid or missing diagnostic flags/counts')
        diagnostic = all(errors[1].get(f,0) <= errors[0].get(f,0) for f in flags)
        verdict[r['direction']] = {'delta':delta,'quality_pass':delta >= -max_drop,'diagnostics_pass':diagnostic}
    return {'status':'REPORT_GATE_PASS' if all(v['quality_pass'] and v['diagnostics_pass'] for v in verdict.values()) else 'REPORT_GATE_FAIL','split':split,'candidate':candidate,'directions':verdict,'raw_sample_coverage_verified':False,'production_acceptance':False}
