"""D01a: synthetic HY-V3-style sparse expert mechanism, not translation."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

DIRECTIONS = ('zh-en', 'en-zh', 'zh-my', 'my-zh', 'en-my', 'my-en')


def route(x, state, top_k, allowed=None):
    scores = (x @ state['router'].T).sigmoid()
    ranked = scores + state['bias']
    if allowed is not None:
        mask = torch.ones(ranked.shape[-1], dtype=torch.bool)
        mask[allowed] = False
        ranked = ranked.masked_fill(mask, -torch.inf)
    ids = ranked.topk(top_k, dim=-1).indices
    weights = scores.gather(-1, ids)
    return ids, weights / weights.sum(-1, keepdim=True) * 2.826


def expert(x, gate_up, down):
    gate, up = (x @ gate_up.T).chunk(2, -1)
    return (torch.nn.functional.silu(gate) * up) @ down.T


def forward(x, state, top_k, allowed=None):
    ids, weights = route(x, state, top_k, allowed)
    result = expert(x, state['shared_gate_up'], state['shared_down'])
    for token, row in enumerate(ids):
        for slot, idx in enumerate(row.tolist()):
            result[token] += weights[token, slot] * expert(
                x[token], state['gate_up'][idx], state['down'][idx])
    return result


def profile(x, state, top_k):
    ids, weights = route(x, state, top_k)
    n = state['router'].shape[0]
    counts = torch.bincount(ids.flatten(), minlength=n)
    mass = torch.zeros(n).scatter_add_(0, ids.flatten(), weights.flatten())
    return counts, mass


def select(profiles, keep, top_k):
    if set(profiles) != set(DIRECTIONS):
        raise ValueError('all six directions required')
    values = torch.stack([profiles[d].double() for d in DIRECTIONS])
    if not torch.isfinite(values).all() or (values < 0).any() or (values.sum(1) <= 0).any():
        raise ValueError('invalid profile')
    if not top_k <= keep <= values.shape[1]:
        raise ValueError('invalid expert count')
    # Normalize independently: a large corpus must not dominate by sample count.
    normalized = values / values.sum(1, keepdim=True)
    scores = normalized.mean(0)
    return sorted(sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:keep])


def prune(state, kept, top_k):
    n = state['router'].shape[0]
    keys = ('router', 'bias', 'gate_up', 'down')
    if set(state) != set(keys) | {'shared_gate_up', 'shared_down'}:
        raise ValueError('unknown tensor layout')
    if any(state[k].shape[0] != n for k in keys):
        raise ValueError('expert axis mismatch')
    if len(set(kept)) != len(kept) or not top_k <= len(kept) <= n or any(type(i) is not int or not 0 <= i < n for i in kept):
        raise ValueError('invalid expert mapping')
    return {k: (v[kept] if k in keys else v).clone().contiguous() for k, v in state.items()}


def size(state):
    return sum(v.numel() * v.element_size() for v in state.values())


def run():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    n, h, m, top_k = 16, 12, 8, 4
    state = {k: torch.randn(*shape) * .2 for k, shape in {
        'router': (n, h), 'bias': (n,), 'gate_up': (n, 2*m, h),
        'down': (n, h, m), 'shared_gate_up': (2*m, h), 'shared_down': (h, m)
    }.items()}
    profiles = {}
    tests = {}
    for i, d in enumerate(DIRECTIONS):
        center = torch.eye(h)[i] * 2
        _, profiles[d] = profile(torch.randn(32+i*8, h)+center, state, top_k)
        tests[d] = torch.randn(24, h)+center
    report = {'scope': 'SYNTHETIC_ONLY', 'torch': torch.__version__, 'seed': 42,
              'original_bytes': size(state), 'candidates': []}
    for keep in (16, 12, 8):
        kept = select(profiles, keep, top_k)
        compact = prune(state, kept, top_k)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'model.safetensors'
            save_file(compact, str(path))
            disk_bytes = path.stat().st_size
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            reloaded = load_file(str(path))
        assert all(torch.equal(compact[k], reloaded[k]) for k in compact)
        assert all(torch.equal(state[k], reloaded[k]) for k in ('shared_gate_up', 'shared_down'))
        direction_report = {}
        for d, x in tests.items():
            original = forward(x, state, top_k)
            masked = forward(x, state, top_k, kept)
            actual = forward(x, reloaded, top_k)
            torch.testing.assert_close(actual, masked, atol=1e-6, rtol=1e-5)
            if keep == n:
                torch.testing.assert_close(actual, original, atol=1e-6, rtol=1e-5)
            ids, weights = route(x, state, top_k)
            coverage = (weights * torch.isin(ids, torch.tensor(kept))).sum()/weights.sum()
            direction_report[d] = {'retained_routing_mass': round(float(coverage), 6),
                'masked_max_abs_error': float((actual-masked).abs().max()),
                'original_max_abs_error': float((actual-original).abs().max())}
        report['candidates'].append({'keep': keep, 'ids': kept, 'tensor_bytes': size(compact),
            'file_bytes': disk_bytes, 'sha256': digest, 'directions': direction_report})
    negatives = [lambda: prune(state, [0,0,1,2], top_k),
                 lambda: prune(state, [0,1,2], top_k),
                 lambda: prune(state, [0,1,2,16], top_k),
                 lambda: select({}, 8, top_k),
                 lambda: prune({**state, 'unknown': torch.zeros(1)}, list(range(8)), top_k)]
    for test in negatives:
        try:
            test()
        except ValueError:
            continue
        raise AssertionError('negative test accepted')
    report['negative_tests_passed'] = len(negatives)
    report['result'] = 'PASS_SYNTHETIC_MECHANISM_ONLY'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
