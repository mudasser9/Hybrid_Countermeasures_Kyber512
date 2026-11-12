import numpy as np
Q=3329
HW=np.array([bin(i).count('1') for i in range(4096)])
# CBD eta=3 pmf over s in [-3,3]
from math import comb
pm={}
for a in range(4):
    for b in range(4):
        pm[a-b]=pm.get(a-b,0)+comb(3,a)*comb(3,b)/64
pm={k:v/ sum(pm.values()) for k,v in pm.items()}
print('CBD3 pmf',{k:round(v,4) for k,v in sorted(pm.items())})
lab={}
for s,p in pm.items(): lab[HW[s%Q]]=lab.get(HW[s%Q],0)+p
print('label HW(s) pmf',{k:round(v,4) for k,v in sorted(lab.items())}, 'majority',round(max(lab.values()),4))
H=lambda p: -sum(x*np.log2(x) for x in p if x>0)
print('H(label) bits',round(H(lab.values()),3))
# noise-free mutual information I(s ; (HW(s0),HW(s1))) with s0=(s-r)%Q, s1=r
joint={}   # (s) -> dict (h0,h1)->count
r=np.arange(Q)
for s in pm:
    h0=HW[(s-r)%Q]; h1=HW[r]
    d={}
    for a,b in zip(h0,h1): d[(a,b)]=d.get((a,b),0)+1/Q
    joint[s]=d
marg={}
for s in pm:
    for k,v in joint[s].items(): marg[k]=marg.get(k,0)+pm[s]*v
I=0
for s in pm:
    for k,v in joint[s].items():
        I+=pm[s]*v*np.log2(v/marg[k])
print('noise-free I(s; HW(s0),HW(s1)) bits =',round(I,5),' of H(s)=',round(H(pm.values()),3))
# single share
for name,idx in (('share0',0),('share1',1)):
    m={}
    for s in pm:
        for k,v in joint[s].items(): m[k[idx]]=m.get(k[idx],0)+pm[s]*v
    I1=0
    for s in pm:
        ds={}
        for k,v in joint[s].items(): ds[k[idx]]=ds.get(k[idx],0)+v
        for h,v in ds.items(): I1+=pm[s]*v*np.log2(v/m[h])
    print(f'noise-free I(s; HW({name})) bits =',round(I1,6))
