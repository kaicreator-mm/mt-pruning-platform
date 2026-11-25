"""D19: router-only recovery on trained synthetic classifier, no language claims."""
import copy, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'r4'))
import torch
from torch import nn
from toy import Net

def experiment(seed):
    torch.manual_seed(seed); torch.set_num_threads(1)
    x=torch.randn(1280,12); y=(x@torch.randn(12,4)).argmax(-1); m=Net(); opt=torch.optim.AdamW(m.parameters(),lr=.015)
    for _ in range(100):
        ix=torch.randint(0,640,(64,)); loss=nn.functional.cross_entropy(m(x[ix])[0],y[ix]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _,cal=m(x[640:832]); p=Net(4); p.out.load_state_dict(m.out.state_dict())
        for b,c,(i,w) in zip(m.blocks,p.blocks,cal):
            a=torch.zeros(len(i),8).scatter_add_(1,i,w); ids=sorted(a.sum(0).topk(4).indices.tolist()); c.router.weight.copy_(b.router.weight[ids]); c.experts.copy_(b.experts[ids])
        teacher=m(x[832:1088])[0].detach()
    def metrics(model):
        with torch.no_grad():
            t=m(x[1088:])[0]; z=model(x[1088:])[0]
            return {'correct':int((z.argmax(-1)==y[1088:]).sum()),'teacher_agreement':int((z.argmax(-1)==t.argmax(-1)).sum()),'kl':float(nn.functional.kl_div(z.log_softmax(-1),t.softmax(-1),reduction='batchmean'))}
    q=copy.deepcopy(p); before={n:v.detach().clone() for n,v in q.named_parameters()}
    for n,v in q.named_parameters(): v.requires_grad_('router' in n)
    opt=torch.optim.AdamW([v for v in q.parameters() if v.requires_grad],lr=.005)
    for _ in range(100):
        ix=torch.randint(0,256,(64,)); z=q(x[832:1088][ix])[0]; loss=nn.functional.kl_div(z.log_softmax(-1),teacher[ix].softmax(-1),reduction='batchmean'); opt.zero_grad(); loss.backward(); opt.step()
    changed=[n for n,v in q.named_parameters() if not torch.equal(v,before[n])]
    assert changed and all('router' in n for n in changed)
    return {'seed':seed,'teacher':metrics(m),'pruned':metrics(p),'recovered':metrics(q),'changed':changed,'test_n':192}

if __name__=='__main__': print(json.dumps([experiment(s) for s in [41,42,43]],indent=2))
