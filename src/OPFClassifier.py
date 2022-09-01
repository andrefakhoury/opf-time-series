import numpy as np
from collections import defaultdict
from dtaidistance import dtw
import heapq

"""Union Find structure with Path Compression + Merge by Size"""
class DSU:
    def __init__(self, n):
        self.par = np.arange(n)
        self.siz = np.ones(n)
        
    def find(self, x):
        if self.par[x] == x:
            return x
        self.par[x] = self.find(self.par[x])
        return self.par[x]
    
    def same(self, u, v):
        return self.find(u) == self.find(v)

    def merge(self, u, v):
        u = self.find(u)
        v = self.find(v)
        if u == v:
            return
        if self.siz[u] > self.siz[v]:
            u, v = v, u
        self.par[u] = v
        self.siz[v] += self.siz[u]
        
"""Optimum Path Forest Classifier"""
class OptimumPathForestClassifier:
    def __init__(self, cost='euclidean-distance'):
        available_cost_functions = {
            'euclidean-distance': lambda x, y: np.linalg.norm(x - y),
            'manhattan-distance': lambda x, y: np.sum(np.abs(x - y)),
            'dtw-distance': lambda x, y: dtw.distance_fast(x, y, use_pruning=True)
        }
        assert cost in available_cost_functions.keys(), f"Invalid cost function. Should be one of {available_cost_functions.keys()}"
        self.F = available_cost_functions[cost]            
    
    def fit(self, X_, Y_):
        n = len(Y_)
        self.X = np.array(X_, copy=True)
        self.labels = np.ones(n) * -1
        Y = np.array(Y_, copy=True)
        
        # First of all, builds the graph
        self.adj = defaultdict(list)
        self.edges = []
        for u in range(n):
            self.adj[u] = [(v, self.F(self.X[u], self.X[v])) for v in range(n)]
            self.edges += [(w, u, v) for v, w in self.adj[u]]
        
        # Runs MST to choose PROTOTYPES (seed vertices)
        self.prototypes = []        
        self.edges.sort()
        dsu = DSU(n)
        for w, u, v in self.edges:
            if not dsu.same(u, v):
                dsu.merge(u, v)
                if Y[u] != Y[v]:
                    self.prototypes += [u, v]
        self.prototypes = np.unique(self.prototypes)                
        
        # Run multisourced dijkstra on prototypes to get the cost
        self.cost = np.ones(n) * np.inf
        self.cost[self.prototypes] = 0
        self.labels[self.prototypes] = Y[self.prototypes]
        
        pq = [[0., u] for u in self.prototypes]
        heapq.heapify(pq)
        while pq:
            u_w, u = heapq.heappop(pq)
            if self.cost[u] < u_w:
                continue
            for v, w in self.adj[u]:
                if self.cost[v] > max(u_w, w):
                    self.cost[v] = max(u_w, w)
                    self.labels[v] = self.labels[u]
                    heapq.heappush(pq, [self.cost[v], v])
                    
    def _classify_one_vertex(self, x):
        b, y = np.inf, -1
        for i in range(len(self.X)):
            nb = max(self.F(self.X[i], x), self.cost[i])
            if nb < b:
                b, y = nb, self.labels[i]
        assert(y != -1)
        return y
    
    def classify(self, X_):
        X_train = np.array(X_, copy=True)
        Y_pred = []
        for x in X_train:
            Y_pred.append(self._classify_one_vertex(x))
        return Y_pred