"""Position-marginalised (aggregate) second-order attack on C2, plus C1 and C2-oracle controls.
Exact Bayes attacker for the paper's leakage model (HW/12 + N(0,s^2)); no learned model."""
import numpy as np, sys, json
q=3329; rng=np.random.default_rng(int(sys.argv[2]) if len(sys.argv)>2 else 1)
sigma=float(sys.argv[1]); NK=100; NT=2000; SL=640
hw=np.array([bin(v).count("1") for v in range(4096)])
vals=np.arange(-3,4); pv=np.array([1,6,15,20,15,6,1])/64
cls_of=lambda v: hw[v%q]
classes=sorted(set(cls_of(v) for v in vals)); C=len(classes)
pi=np.array([pv[[cls_of(v)==c for v in vals]].sum() for c in classes])
# exact pair-HW distribution P_c[h0,h1]
P=np.zeros((C,13,13)); s0=np.arange(q)
for ci,c in enumerate(classes):
    for v,p in zip(vals,pv):
        if cls_of(v)!=c: continue
        s1=(v-s0)%q
        np.add.at(P[ci],(hw[s0],hw[s1]),p/ q)
P/=pi[:,None,None]
H=np.arange(13)/12
def g(x): return np.exp(-(x[...,None]-H)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))   # (...,13)
def fcls(x0,x1):   # (...,C)
    return np.einsum('...i,cij,...j->...c',g(x0),P,g(x1))
def sample_key():
    s=rng.choice(vals,size=512,p=pv); return s
def pair_hw(sv):
    a=rng.integers(0,q,size=sv.shape); return hw[a],hw[(sv-a)%q]
def trace_slots(key,cfg):
    """returns pair-slot observations (SL,2) and slot index of coefficient 0 (or 0)"""
    h0,h1=pair_hw(key)
    if cfg=="C1":
        sl=np.stack([h0,h1],1)[:SL]; pos=0
    else:
        order=rng.permutation(512); rows=[]; pos=None
        for k in order:
            if rng.random()<0.25:
                d=rng.integers(0,q,2); rows.append((hw[d[0]],hw[d[1]]))
            if k==0: pos=len(rows)
            rows.append((h0[k],h1[k]))
        rows=np.array(rows)
        asc=hw[rng.integers(0,q,size=(SL,2))]   # ASCON/tail: secret-independent random HW
        full=np.concatenate([rows,asc])[:SL]; sl=full
        if pos>=SL: pos=0
    obs=sl/12+rng.normal(0,sigma,sl.shape)
    return obs,pos
def run(cfg,mode):
    out=np.zeros((NK,NT)); ranks=np.zeros((NK,NT),dtype=int)
    for k in range(NK):
        key=sample_key(); true=classes.index(cls_of(key[0]))
        ll=np.zeros(C); 
        for t in range(NT):
            obs,pos=trace_slots(key,cfg)
            f=fcls(obs[:,0],obs[:,1])           # (SL,C)
            if mode=="oracle": s=np.log(f[pos]/ (f[pos]@pi))
            else:
                fm=f@pi; s=np.log((f/fm[:,None]).mean(0))
            ll+=s; post=np.log(pi)+ll
            ranks[k,t]=(post>post[true]).sum()
    return ranks
res={}
for name,cfg,mode in [("C1 position known","C1","oracle"),("C2 oracle (location given)","C2","oracle"),("C2 aggregate over all positions","C2","blind")]:
    r=run(cfg,mode)
    sr=(r==0).mean(0); ge=r.mean(0)
    n50=int(np.argmax(sr>=0.5))+1 if (sr>=0.5).any() else None
    n90=int(np.argmax(sr>=0.9))+1 if (sr>=0.9).any() else None
    res[name]=dict(GE_2000=float(ge[-1]),SR_2000=float(sr[-1]),n50=n50,n90=n90,SR_1=float(sr[0]),SR_300=float(sr[299]))
    print(sigma,name,res[name],flush=True)
prior=np.sort(pi)[::-1]; 
print("prior-only GE",float((np.arange(C)*prior).sum()))
json.dump(res,open(f"agg_s{sigma}.json","w"))
