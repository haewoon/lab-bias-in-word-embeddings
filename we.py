# This code is largely based on debias library from github.com/tolga-b/debiaswe

import numpy as np

class WordEmbedding:
    def __init__(self, filename):
        vecs = []
        words = []

        with open(filename, "r", encoding='utf8') as f:
            for line in f:
                s = [t.strip() for t in line.split()]
                v = np.array([float(x) for x in s[1:]])
                words.append(s[0])
                vecs.append(v)

        self.vecs = np.array(vecs, dtype='float32')
        print(self.vecs.shape)
        self.words = words            

        self.reindex()

    def reindex(self):
        self.index = {w: i for i, w in enumerate(self.words)}
        self.n, self.d = self.vecs.shape
        assert self.n == len(self.words) == len(self.index)
        self._neighbors = None
        print(self.n, "words of dimension", self.d, ":", ", ".join(self.words[:4] + ["..."] + self.words[-4:]))

    def v(self, word):
        return self.vecs[self.index[word]]

    def diff(self, word1, word2):
        v = self.vecs[self.index[word1]] - self.vecs[self.index[word2]]
        return v/np.linalg.norm(v)
