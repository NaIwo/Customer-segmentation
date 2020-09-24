from copy import deepcopy
import numpy as np

class MyDBSCAN():
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        
    def clear(self):
        self.core_instances_ = np.array([])
        self.components_ = list()
        self.labels_ = np.array([])

    #Recursion /DFS - graph/ - search for groups of nodes  
    def set_labels_recursion(self, index, belongings, marked):
        marked[index] = True
        try:
            index = list(self.core_instances_).index(index)
            for node in belongings[index]:
                if not marked[node]:
                    self.set_labels_recursion(int(node), belongings, marked)
        except ValueError:
            pass
        
    def set_labels(self, belongings, X):
        labels = dict()
        marked = [False] * X.shape[0]
        counter = 0
        for core in self.core_instances_:
            last_marked = deepcopy(marked)
            self.set_labels_recursion(int(core), belongings, marked)
            
            if last_marked != marked:
                changed = np.bitwise_xor(last_marked, marked)
                labels[counter] = np.where(changed == True)[0].tolist()
                counter += 1
        del changed
        del last_marked
        del marked
        
        self.labels_ = np.array([-1] * X.shape[0])
        for key in labels.keys():
            self.labels_[labels[key]] = key
            
        del labels
        
        
    def fit(self, X):
        self.clear()
        
        belongings = list()
        for index, data in enumerate(X):
            temp = self.compute_distances(data, X, index)
            if np.count_nonzero(temp < self.eps) >= self.min_samples:
                self.core_instances_ = np.append(self.core_instances_, index)
                self.components_.append(list(data))
                belongings.append(np.where(temp < self.eps)[0].tolist())
                
        self.set_labels(belongings, X)
        del belongings
        
        self.components_ = np.array(self.components_)
        return self