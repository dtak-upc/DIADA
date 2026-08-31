import math
from typing import Tuple

# Memoized polygamma approximations used for Beta distribution statistics in log-odds space.
# y1(n) ≈ digamma(n)   = ψ(n),       y1(1) = -γ (Euler-Mascheroni)
# y2(n) ≈ trigamma(n)  = ψ₁(n),      y2(1) = π²/6
# y3(n) ≈ tetragamma(n)= ψ₂(n),      y3(1) ≈ -2.404
# y4(n) ≈ pentagamma(n)= ψ₃(n),      y4(1) ≈ 6.494
# Each uses the asymptotic tail once n is large enough to avoid table growth.
yconst=0.5772156649
def y1f():
    mem=[-yconst]
    def f(n):
        if n>=5000:
            return math.log(n)
        while len(mem)<n:
            mem.append(mem[-1]+1/len(mem))
        return mem[n-1]
    return f
y1=y1f()


def y2f():
    mem=[math.pi**2/6]
    def f(n):
        if n>=1000:
            return 1/n
        while len(mem)<n:
            mem.append(mem[-1]-1/len(mem)**2)
        return mem[n-1]
    return f
y2=y2f()


def y3f():
    mem=[-2.404114]
    def f(n):
        if n>=200:
            return -1/n**2
        while len(mem)<n:
            mem.append(mem[-1]+2/(len(mem)**3))
        return mem[n-1]
    return f
y3=y3f()

def y4f():
    mem=[6.493939]
    def f(n):
        if n>=50:
            return 2/n**3
        while len(mem)<n:
            mem.append(mem[-1]-6/(len(mem)**4))
        return mem[n-1]
    return f
y4=y4f()




class BetaDist:
    # Tracks a Beta(a,b) distribution representing selectivity p ∈ (0,1).
    # Sufficient statistics in log-odds space:
    #   meanLO = E[logit(p)] ≈ ψ(a) - ψ(b)
    #   varLO  = Var[logit(p)] ≈ ψ₁(a) + ψ₁(b)
    # Gradients dmda/dmdb and dvda/dvdb are with respect to a and b.

    def __init__(self,a:int,b:int,mean:float,meanLO:float,varLO:float,sdLO:float,dmda:float,dmdb:float,dvda:float,dvdb:float) -> None:
        self.a=a
        self.b=b
        self.mean=mean
        self.meanLO=meanLO
        self.varLO=varLO
        self.sdLO=sdLO
        self.dmda=dmda
        self.dmdb=dmdb
        self.dvda=dvda
        self.dvdb=dvdb

    @staticmethod
    def empty()->'BetaDist':
        # Uninformative prior Beta(1,1) = Uniform[0,1]
        res= BetaDist(0,0,0,0,0,0,0,0,0,0)
        res.add(1,1)
        return res


    def updateStats(self):
        self.mean=self.a/(self.a+self.b)
        #self.meanLO=y1(self.a)-y1(self.b)
        #self.varLO=y2(self.a)+y2(self.b)
        self.meanLO=y1(self.a)-y1(self.a+self.b)
        self.varLO=y2(self.a)-y2(self.a+self.b)
        self.sdLO=math.sqrt(self.varLO)
        self.dmda=1/self.a
        self.dmdb=-1/self.b
        self.dvda=-self.dmda*self.dmda
        self.dvdb=-self.dmdb*self.dmdb
        
    def add(self,a:int,b:int):
        self.a+=a
        self.b+=b
        self.updateStats()
    def copy(self)->'BetaDist':
        return BetaDist(self.a,self.b,self.mean,self.meanLO,self.varLO,self.sdLO,self.dmda,self.dmdb,self.dvda,self.dvdb)
    
    def __repr__(self) -> str:
        return f"({self.a},{self.b})"

        
class BBDist:
        # Bundles three Beta distributions needed for finite-difference gradient estimation:
        #   d  = Beta(a, b)
        #   da = Beta(a+1, b)  — one extra success, for ∂/∂a
        #   db = Beta(a, b+1)  — one extra failure, for ∂/∂b
        def __init__(self,d:BetaDist,da:BetaDist,db:BetaDist) -> None:
            self.d=d
            self.da=da
            self.db=db

        @staticmethod
        def empty()->'BBDist':
            d=BetaDist.empty()
            da=d.copy()
            da.add(1,0)
            db=d.copy()
            db.add(0,1)
            return BBDist(d,da,db)
        def add(self,a,b):
            self.d.add(a,b)
            self.da.add(a,b)
            self.db.add(a,b)
            
        def copy(self)->'BBDist':
            return BBDist(self.d.copy(),self.da.copy(),self.db.copy())
            
    



# nodeGrad: sensitivity of the mean to one additional observation (∂mean/∂a = b/(a+b)²).
# High gradient → mean will shift more per sample → worth scheduling.
def nodeGrad(d: 'BetaDist') -> float:
    s = d.a + d.b
    return d.b / (d.a * s * s)


# dist1: z-score of log-odds of a single node; large positive = node is selective
def dist1(a:'BetaDist')->float:
        m=a.meanLO
        v=a.varLO
        sd=math.sqrt(v)
        return m/sd

# nodeDist4: z-score of the interaction between two edges sharing the same added predicate.
# from1→to1 is the original edge (e.g. (A,B)→(A,B,C));
# from2→to2 is a reference edge with one fewer parent predicate (e.g. A→(A,C)).
# Negative return means the original edge adds more selectivity than the reference.
def nodeDist4(from1: 'BetaDist', to1: 'BetaDist', from2: 'BetaDist', to2: 'BetaDist') -> float:
    m = (to1.meanLO - from1.meanLO) - (to2.meanLO - from2.meanLO)
    v = to1.varLO + from1.varLO + to2.varLO + from2.varLO
    return m / math.sqrt(v)


# dist2: z-score of difference in log-odds between two nodes; measures how much one
# edge predicate shifts selectivity relative to uncertainty
def dist2(a:'BetaDist',b:'BetaDist')->float:
        m1=a.meanLO
        m2=b.meanLO
        v1=a.varLO
        v2=b.varLO

        m=m1-m2
        v=v1+v2
        sd=math.sqrt(v)
        return m/sd


# grad2: finite-difference gradient of dist2 w.r.t. adding one observation to a or b
def grad2(a:'BetaDist',b:'BetaDist')->Tuple[float,float]:
        m1=a.mean
        m2=b.mean

        a1=a.copy()
        a1.add(1,0)
        b1=b.copy()
        b1.add(0,1)

        d0=dist2(a,b)
        da=dist2(a1,b)
        db=dist2(a,b1)

        ga=da-d0
        gb=db-d0

        return ga,gb

# gradsq: gradient of the squared 2x2 interaction signal across a lattice square.
# The four nodes form: ff (neither pred), ft (pred B only), tf (pred A only), tt (both).
# Interaction = tt - tf - ft + ff  in log-odds space; measures non-additivity.
def gradsq(tf:'BetaDist',tt:'BetaDist',ff:'BetaDist',ft:'BetaDist')->Tuple[float,float,float,float]:
    
    utf=tf.mean
    vtf=tf.varLO
    
    utt=tt.mean
    vtt=tt.varLO
    
    uff=ff.mean
    vff=ff.varLO
    
    uft=ft.mean
    vft=ft.varLO
    
    u=utt-utf-uft+uff
    v=vtf+vtt+vff+vft
    
    u2=u*u+v
    v2=2*v*v+4*u*u*v
    
    dutfda=-1/tf.a
    duttda=1/tt.a
    duffda=1/ff.a
    duftda=-1/ft.a
    
    dutfdb=1/tf.b
    duttdb=-1/tt.b
    duffdb=-1/ff.b
    duftdb=1/ft.b
    
    
    dvtfda=-1/(tf.a*tf.a)
    dvttda=-1/(tt.a*tt.a)
    dvffda=-1/(ff.a*ff.a)
    dvftda=-1/(ft.a*ft.a)
    
    dvtfdb=-1/(tf.b*tf.b)
    dvttdb=-1/(tt.b*tt.b)
    dvffdb=-1/(ff.b*ff.b)
    dvftdb=-1/(ft.b*ft.b)
    
    # TODO: incomplete — intermediate u,v,u2,v2 are computed but the full gradient
    # (which should use dutfda etc.) is not assembled; currently only returns ∂v terms
    return dvtfda,dvttdb,dvffdb,dvftda


# distNode: z-score of the 2x2 interaction effect (same as interaction numerator / sum-of-SDs).
# A large absolute value means the two predicates interact non-additively in selectivity.
def distNode(ff:'BetaDist',ft:'BetaDist',tf:'BetaDist',tt:'BetaDist'):
    m=tt.meanLO-tf.meanLO-ft.meanLO+ff.meanLO
    sd=tt.sdLO+tf.sdLO+ft.sdLO+ff.sdLO
    return m/sd
    
    


        