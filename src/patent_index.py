import os

import faiss
import numpy as np

from tqdm import tqdm

class PatentIndex:
    """Patent Index."""

    def __init__(self, dim=512):
        self.index = faiss.IndexFlatIP(dim)
        self.key2idx, self.idx2key = {}, {}
        self.patent2idx, self.idx2patent = {}, {}
        self.hashcode2idx, self.idx2hashcode = {}, {}

    @property
    def size(self):
        return self.index.ntotal

    def add(self, embeddings, keys, patents, hashcodes):
        """Add embeddings."""
        idx = self.index.ntotal
        self.index.add(embeddings)

        for i, k in enumerate(keys):
            self.idx2key[idx + i] = k
            self.key2idx[k] = idx + i

        for i, patent in enumerate(patents):
            self.patent2idx.setdefault(patent, []).append(idx + i)
            self.idx2patent[idx + i] = patent

        for i, hashcode in enumerate(hashcodes):
            self.idx2hashcode[idx + i] = hashcode
            self.hashcode2idx[hashcode] = idx + i

    def search_top_k(self, query_embedding, top_k=None):
        """Search for similar embeddings"""
        K = top_k if top_k else self.size
        D, I = self.index.search(query_embedding.reshape(1, -1), k=K)
        return [{
                'id': int(i),
                'key': self.idx2key[i],
                'patent': self.idx2patent[i],
                'hashcode': self.idx2hashcode[i],
                'score': round(float(d), 6)
            } for i, d in zip(I[0], D[0])
        ]
    
    def search_top_k_within_threshold(
        self, query_embedding, min_threshold=0.0, max_threshold=float('inf')
    ):
        """Search for similar embeddings"""
        
        query_embedding = query_embedding.reshape(1, -1)
        
        lims, D, I = self.index.range_search(
            query_embedding.reshape(1, -1), min_threshold
        )

        filtered_results = []
        for i in range(lims[0], lims[1]):
            if D[i] <= max_threshold:
                filtered_results.append({
                    'id': int(I[i]),
                    'key': str(self.idx2key[I[i]]),
                    'patent': str(self.idx2patent[I[i]]),
                    'hashcode': str(self.idx2hashcode[I[i]]),
                    'score': round(float(D[i]), 6)
                })

        return sorted(filtered_results, key=lambda x: -x['score'])

    def start_index(self, load_dir, metadata):
        """Index embeddings."""

        shards = sorted(os.listdir(load_dir))

        for shard in shards:
            keys = np.load(f'{load_dir}/{shard}/keys.npy')
            patents = [''.join(key.split('_')[:-2]) for key in keys]
            embeddings = np.load(f'{load_dir}/{shard}/{metadata}.npy')
            hashcodes = np.load(f'{load_dir}/{shard}/hashcodes.npy')

            if len(embeddings) == 0: continue
            assert len(keys) == len(patents)
            assert len(keys) == embeddings.shape[0]

            self.add(embeddings, keys, patents, hashcodes)

        print(f'Index has {self.size} embeddings')