from typing import List,Tuple,Dict,Generic,TypeVar
from collections.abc import ItemsView


# A lattice whose nodes are subsets of active predicates (at most one per predicate group).
# Each node is a conjunction of predicates; edges add/remove one predicate from one group.
# The root (empty node) represents no active predicates (selectivity = 1.0).
class Lattice:

    def __init__(self,pgs:List[int]) -> None:
        # pgs[i] = number of predicates in group i (0-indexed)
        self.pgs=pgs
        self.npgs=len(pgs)
        # pgscores is a mixed-radix positional weight for each group, used for hashing.
        # pgscores[i] = product of (pgs[j]+1) for j<i, so each group occupies pgs[i]+1 "digits".
        self.pgscores=[1]*(self.npgs+1);
        self.mask=0xffffffff
        for i in range(self.npgs):
            self.pgscores[i+1]=(self.pgscores[i]*(self.pgs[i]+1))&self.mask;
        self.maxLatticeSize=self.pgscores[-1]

    class Node:
        # A node encodes a partial predicate assignment: preds = {pg -> predicate_index}.
        # hash = sum over active (pg,p) of pgscores[pg]*(p+1); digit 0 means group absent.

        def __init__(self,lattice:'Lattice') -> None:
            self.lattice=lattice
            self.preds:Dict[int,int]={}
            self.hash=0

        def to(self,pg:int,p:int) -> 'Lattice.Edge':
            # Returns the edge going UP: self -> self + (pg,p)
            t=self.copy()
            t._add(pg,p)
            return Lattice.Edge(self,t,pg,p)


        def fr(self,pg:int) -> 'Lattice.Edge':
            # Returns the edge going DOWN: self - pg -> self
            n=self.copy()
            p=n._remove(pg)
            return Lattice.Edge(n,self,pg,p)

            
        def _add(self,pg:int,p:int) -> None:
            assert pg not in self.preds
            self.preds[pg]=p
            self.hash+=self.lattice.pgscores[pg]*(p+1)

        def _remove(self,pg:int) -> int:
            assert pg in self.preds
            p=self.preds[pg]
            del self.preds[pg]
            self.hash-=self.lattice.pgscores[pg]*(p+1)
            return p

        def get(self,pg:int) -> int:
            return self.preds[pg]
        
        def __contains__(self, item):
            return item in self.preds
        
        def copy(self):
            res=Lattice.Node(self.lattice)
            res.hash=self.hash
            res.preds=self.preds.copy()
            return res
        
        def __len__(self):
            return len(self.preds)
        
        def __hash__(self) -> int:
            return self.hash
        
        def __eq__(self, value) -> bool:
            if self.hash!=value.hash:
                return False
            return self.preds==value.preds
        
        def __repr__(self) -> str:
            return self.preds.__repr__()


    class Edge:

        def __init__(self,f:'Lattice.Node',t:'Lattice.Node',pg:int,p:int) -> None:
            self.f=f
            self.t=t
            self.pg=pg
            self.p=p
            self.hash=t.hash*len(t)+pg

        def to(self) -> 'Lattice.Node':
            return self.t
        
        def fr(self) -> 'Lattice.Node':
            return self.f
        
        def __hash__(self) -> int:
            return self.hash
        
        def __eq__(self, value) -> bool:
            if self.hash!=value.hash:
                return False
            return self.f==value.f and self.p==value.p
        
        def __repr__(self) -> str:
            return self.f.__repr__()+f' + ({self.pg}: {self.p})'


    def getRoot(self) -> Node:
        return self.Node(self)
    
    


N=TypeVar("N")
E=TypeVar("E")

class LatticeMap(Generic[N,E]):
    def __init__(self) -> None:
        super().__init__()
        if N is not None:
            self.nodeMap:Dict[Lattice.Node,N]={}
        if E is not None:
            self.edgeMap:Dict[Lattice.Edge,E]={}
            
    def getNode(self,n:Lattice.Node)->N:
        return self.nodeMap[n]
    
    def getEdge(self,e:Lattice.Edge)->E:
        return self.edgeMap[e]
    
    def setNode(self,n:Lattice.Node,nn:N):
        self.nodeMap[n]=nn
    
    def setEdge(self,e:Lattice.Edge,ee:E):
        self.edgeMap[e]=ee
        
    def nodes(self)->ItemsView[Lattice.Node,N]:
        return self.nodeMap.items()
    
    def edges(self)->ItemsView[Lattice.Edge,E]:
        return self.edgeMap.items()
        