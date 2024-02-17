import numpy as np
def normalize1 (x) :
    sm = sum(x)
    for i in range(len(x)) :
        x[i] /= sm
    return x
class Graph :
    adj = []
    def page_rank (self, alpha) :
        n = len(self.adj)
        D = []
        for i in range(n) :
            D.append([0] * n)
            D[i][i] = max(sum(self.adj[i]), 1)
        arr = np.subtract(np.identity(n), alpha * np.matmul(np.transpose(self.adj), np.linalg.inv(D)))
        return np.matmul(np.linalg.inv(arr), np.transpose(np.ones(n)))
    
    def eigenvector_centrality (self) :
        eigenvalues, eigenvectors = np.linalg.eig(np.transpose(self.adj))
        ind = 0
        for i in range(len(eigenvalues)) :
            if eigenvalues[i] >= eigenvalues[ind] :
               ind = i
        ans = []
        for i in range(len(eigenvectors)) :
            ans.append(eigenvectors[i][ind])
        return normalize1(ans)
    
    def katz_centrality (self, alpha) :
        n = len(self.adj)
        arr = np.subtract(np.identity(n), alpha * np.transpose(self.adj))
        return np.matmul(np.linalg.inv(arr), np.transpose(np.ones(n)))
    
    def add_edge (self, u: int, v: int) : 
        self.adj[u][v] = 1
    
    def initialize (self, mat: list[list[int]]) :
        assert len(mat) == len(mat[0])
        for i in range(len(mat)) :
            assert len(mat) == len(mat[i])
            for j in range(len(mat[i])) :
                assert 0 <= mat[i][j] <= 1
        self.adj = mat
    
    def __init__ (self, n: int) : 
        self.adj = []
        for i in range(n) :
            self.adj.append([0] * n)
    
gr = Graph(3)
"""
gr.add_edge(0, 1)
gr.add_edge(1, 0)
gr.add_edge(0, 2)
gr.add_edge(2, 0)
gr.add_edge(1, 2)
"""
gr.initialize([[0, 1, 1], [1, 0, 1], [1, 0, 0]])
print(gr.page_rank(0.5))
