import csv, math, itertools
from scipy import stats
D="/mnt/user-data/uploads/Desktop/binding-vit/"
def load(c,s):
    rows=list(csv.DictReader(open(D+f"e_measure_full_c{c}_s{s}.csv")))
    return rows
R={(c,s):load(c,s) for c in (1,2,3) for s in range(4)}
def ep(c,s,e):
    return [r for r in R[(c,s)] if r['point']==f"ep{e:02d}"][0]
f=lambda x: float(x)
print("=== epoch 40 per seed (far bg left/upper %, ratio, d_lr, d_ud, CLS left/upper)")
for c in (1,2,3):
    for s in range(4):
        r=ep(c,s,40)
        print(c,s, f"L {100*f(r['left']):.2f} U {100*f(r['upper']):.2f} gap {100*f(r['gap']):.2f} ratio {f(r['ratio']):.3f} [{f(r['ratio_lo']):.3f},{f(r['ratio_hi']):.3f}] dlr {f(r['d_lr']):.3e} dud {f(r['d_ud']):.3e} CLS L {100*f(r['cls_left']):.2f} U {100*f(r['cls_upper']):.2f} q_lr {f(r['q_lr']):.3f} q_ud {f(r['q_ud']):.3f}")
def vals(c,k): return [f(ep(c,s,40)[k]) for s in range(4)]
# H1
u1=[100*x for x in vals(1,'upper')]; u3=[100*x for x in vals(3,'upper')]
sep=all(a<b for a in u3 for b in u1); diff=sum(u1)/4-sum(u3)/4
print("\nH1: upper c1",[round(x,2) for x in u1],"c3",[round(x,2) for x in u3],"all c3<c1:",sep,"diff means",round(diff,2),"->", "confirmed" if sep and diff>3 else "not shown")
# H2 Welch 90% CI
def welch(a,b):
    ma,mb=sum(a)/len(a),sum(b)/len(b)
    va=sum((x-ma)**2 for x in a)/(len(a)-1); vb=sum((x-mb)**2 for x in b)/(len(b)-1)
    se=math.sqrt(va/len(a)+vb/len(b)); df=(va/len(a)+vb/len(b))**2/((va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1))
    t=stats.t.ppf(0.95,df); d=mb-ma
    return d,d-t*se,d+t*se,df
res={}
for k,s,scale in (('ratio',0.15,1),('left',3,100)):
    a=[scale*x for x in vals(1,k)]; b=[scale*x for x in vals(2,k)]
    d,lo,hi,df=welch(a,b); res[k]=(d,lo,hi,s)
    print(f"H2 {k}: c1 {[round(x,4) for x in a]} c2 {[round(x,4) for x in b]} diff(c2-c1) {d:.4f} 90%CI [{lo:.4f},{hi:.4f}] df {df:.2f} SESOI ±{s}")
eq=all(-s<lo and hi<s for d,lo,hi,s in res.values())
diffr=any(lo>s or hi<-s for d,lo,hi,s in res.values())
print("H2 ->","equivalent" if eq else ("different" if diffr else "not resolved"))
# H4
rnd=[r for r in R[(1,0)] if r['point']=='random']
print("\nrandom nets ratio [lo,hi]:",[(r['seed'],round(f(r['ratio']),3),round(f(r['ratio_lo']),3),round(f(r['ratio_hi']),3)) for r in rnd])
# check random rows identical across files
for key in R:
    rr=[(r['seed'],r['ratio_hi']) for r in R[key] if r['point']=='random']
    assert rr==[(r['seed'],r['ratio_hi']) for r in rnd], key
hi_min=min(f(r['ratio_hi']) for r in rnd); lo_min=min(f(r['ratio_lo']) for r in rnd)
n=0
for c in (1,2):
    for s in range(4):
        early={e:f(ep(c,s,e)['ratio']) for e in (1,2,3,5)}
        r40=f(ep(c,s,40)['ratio'])
        peak=any(v>1.2 for v in early.values()); below=all(r40<f(r['ratio_hi']) for r in rnd)
        ok=peak and below; n+=ok
        print(f"H4 c{c}s{s}: early {', '.join(f'{e}:{v:.3f}' for e,v in early.items())}  ep0 {f(ep(c,s,0)['ratio']):.3f} ep40 {r40:.3f} peak>1.2 {peak} below_rand_hi {below} below_rand_lo {all(r40<f(r['ratio_lo']) for r in rnd)} -> {ok}")
print("H4 ->", "confirmed" if n==8 else f"not confirmed ({n} of 8)")
