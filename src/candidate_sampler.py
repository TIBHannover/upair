import os
import random
from PIL import Image

import torch
import numpy as np

import faiss
from tqdm import tqdm

from encoders import (
    PatentNetEncoder, PaecterEncoder, CLIPEncoder, ResNext101Encoder
)

from patent_index import PatentIndex

from utils import (
    load_config, load_json, save_json,
    clean_text, preprocess
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GROUPS = [('easy','positive'),('easy','negative'),('hard','positive'),('hard','negative')]

class CandidateSampler:
    """Candidate Sampler class"""
    def __init__(self, config_path):
        self.config = load_config(path=config_path)
        self.group_config = self.config['groups']
        self.text_encoder = PaecterEncoder()
        self.clip_encoder = CLIPEncoder()
        self.patentnet_encoder = PatentNetEncoder()
        self.resnext_encoder = ResNext101Encoder()
        self.gt, self.gt_pat = self.load_ground_truth()
        self.current_cited_candidates = None
        self.citations_data = load_json(self.config['citations_data'])
    
    def load_query_dict(self):
        """Prepare query data"""

        query_dict = {}
        query_image_dict = load_json(self.config['query_images'])
 
        for section in tqdm(query_image_dict.keys(), desc='Preparing queries'):

            query_dict.setdefault(section, {})

            for patent, image in query_image_dict[section].items():
                query_patent_path = os.path.join(self.config['query_dir'], section, patent)
                query_image_metadata = load_json(f'{query_patent_path}/{image.split(".")[0]}.json')
                
                query_dict[section][image] = {
                    'patent':  patent,
                    'section': section,
                    'title': self.text_encoder.encode(
                        query_image_metadata['title']
                    ),
                    'desc': self.text_encoder.encode(
                        clean_text(query_image_metadata['fig_desc'])
                    ),
                    'label': self.text_encoder.encode(
                        ' '.join(query_image_metadata['component_labels'])
                    ),
                    'clip': self.clip_encoder.encode(
                        preprocess(
                            image=Image.open(f'{query_patent_path}/{image}'),
                            size=(378,378),
                            num_output_channels=3
                        ).unsqueeze(0)
                    ),
                    'patentnet': self.patentnet_encoder.encode(
                        preprocess(
                            image=Image.open(f'{query_patent_path}/{image}'),
                            size=(384,384),
                            num_output_channels=1
                        ).unsqueeze(0)
                    ),
                    'resnext101': self.resnext_encoder.encode(
                        preprocess(
                            image=Image.open(f'{query_patent_path}/{image}'),
                            size=(384,384),
                            num_output_channels=3
                        ).unsqueeze(0)
                    ),
                }

        return query_dict

    def prepare_indices(self):
        """Load indicies"""

        self.indices = {}

        for index, config in tqdm(self.config['indices'].items(), desc='Indexing'):
            
            patent_index = PatentIndex(dim=config['dim'])
            patent_index.start_index(
                load_dir=config['dir'], metadata=config['metadata']
            )
            self.indices[index] = patent_index

    def set_thresholds(self, force=False):
        """Save thresholds for each query"""

        if not force and os.path.exists(self.config['thresholds_dict']):
            print('Thresholds already set!')
            self.thresholds_dict = load_json(self.config['thresholds_dict'])
            return

        thresholds_dict = {}

        query_dict = self.load_query_dict()
        percentile_values = np.arange(100.0, 80.0, -0.001)

        for index in self.config['indices'].keys():
            
            patent_index = self.indices[index]
            thresholds_dict[index] = {}

            for section in query_dict.keys():

                for key in tqdm(query_dict[section].keys(), f'Saving thresholds for {index}'):

                    results = patent_index.search_top_k(query_embedding=query_dict[section][key][index])
                    scores = [result['score'] for result in results]
                    score_percentiles = np.percentile(scores, percentile_values)
                    thresholds_dict[index][key] = {
                        f"{p:.3f}": f"{s}" for p, s in zip(percentile_values, score_percentiles)
                    }
        
        save_json(thresholds_dict, self.config['thresholds_dict'])
        self.thresholds_dict = thresholds_dict

    def is_duplicate(self, index, vector, threshold):
        """Check if vector is duplicate"""

        if index.ntotal == 0:
            return False
        D, I = index.search(vector.reshape(1, -1), k=1)
        return D[0][0] >= threshold
    
    def is_cited(self, qpatent, cpatent):
        """Check if a patent is cited"""
        return True if cpatent in self.citations_data[qpatent] else False

    def sample_and_deduplicate(
            self, candidate_dict, n, threshold=0.90, sample_factor=2
        ):
        """Sample unique candidates"""
        all_hashcodes = set(candidate_dict.keys())

        unique_candidates = {}
        temp_index = faiss.IndexFlatIP(512)

        if self.current_cited_candidates:

            cited_hashcodes = set(self.current_cited_candidates.keys())
            all_hashcodes = all_hashcodes - cited_hashcodes

            for hashcode in self.current_cited_candidates.keys():
                vector = self.indices['patentnet'].index.reconstruct(
                    self.current_cited_candidates[hashcode]['id']
                )
                temp_index.add(vector.reshape(1, -1).astype('float32'))

        while len(unique_candidates) < n:
            needed = n - len(unique_candidates)
            sample_size = min(len(all_hashcodes), needed * sample_factor)
            if sample_size == 0: break
            sampled_hashcodes = random.sample(all_hashcodes, sample_size)

            for hashcode in sampled_hashcodes:
                if hashcode in unique_candidates: continue
                
                vector = self.indices['patentnet'].index.reconstruct(
                    candidate_dict[hashcode]['id']
                )

                if vector is None:
                    unique_candidates[hashcode] = candidate_dict[hashcode]
                    continue
                
                if self.is_duplicate(temp_index, vector, threshold):
                    continue

                unique_candidates[hashcode] = candidate_dict[hashcode]
                temp_index.add(vector.reshape(1, -1).astype('float32'))
                
                if len(unique_candidates) >= n: break

            all_hashcodes -= set(unique_candidates.keys())

            if len(unique_candidates) < n and len(unique_candidates) == len(all_hashcodes):
                print("Not enough unique candidates to sample n without duplicates.")
                break
        
        unique_candidates = dict(list(unique_candidates.items())[:n])

        return unique_candidates
    
    def fetch_cited_patents(self, query, index='patentnet'):
        """Fetch all cited patents"""
        patent_index = self.indices[index]
        cited_results = {}

        results = patent_index.search_top_k(
            query_embedding=query[index]
        )

        for result in results:
            if self.is_cited(qpatent=query['patent'], cpatent=result['patent']):
                cited_results[result['hashcode']] =  {
                    'id': result['id'],
                    'key': result['key'],
                    'patent': result['patent'],
                    'hashcode': result['hashcode'],
                }

        return cited_results
    
    def select_cited_candidates(self, query):
        """Sample candidates from citations data"""
        candidate_dict = self.fetch_cited_patents(query=query)

        if len(candidate_dict.keys()) > 5:
            samples = self.sample_and_deduplicate(
                candidate_dict=candidate_dict, n=self.config['cited_sample_size']
            )
        else:
            samples = candidate_dict
        self.current_cited_candidates = samples

        return samples

    def merge_results(self, results):
        """Combine results from all indices"""
        hashcode_sets = [set(index_data.keys()) for index_data in results.values()]
        candidate_dict = {}

        def add_hashcode(hashcode, index_data):
            data = index_data[hashcode]['metadata']
            if hashcode not in candidate_dict:
                candidate_dict[hashcode] = {
                    'id': data['id'],
                    'key': data['key'],
                    'patent': data['patent'],
                    'hashcode': data['hashcode'],
                    'sum_score': 0.0
                }
            candidate_dict[hashcode]['sum_score'] += data['score']

        if hashcode_sets:
            common_hashcodes = set.intersection(*hashcode_sets)
        else:
            common_hashcodes = set()

        desc_hashcodes = set(results.get('desc', {}).keys())
        patentnet_hashcodes = set(results.get('patentnet', {}).keys())
        desc_and_patent_hashcodes = desc_hashcodes & patentnet_hashcodes

        if len(common_hashcodes) >= 5: 
            # All intersection case
            for hashcode in common_hashcodes:
                for index in results.keys():
                    add_hashcode(hashcode, results[index])

        elif 0 < len(common_hashcodes) < 5:
            # All intersection + desc_patentnet intersection

            for hashcode in common_hashcodes:
                for index in results.keys():
                    add_hashcode(hashcode, results[index])

            for hashcode in desc_and_patent_hashcodes - common_hashcodes:
                if hashcode in results.get('desc', {}):
                    add_hashcode(hashcode, results['desc'])
                elif hashcode in results.get('patentnet', {}):
                    add_hashcode(hashcode, results['patentnet'])

        else:  # Union case
            for index in results.keys():
                selected = random.sample(
                    results[index].keys(), min(1000, len(results[index]))
                )
                for hashcode in selected:
                    add_hashcode(hashcode, results[index])

        return candidate_dict
     
    def select_candidates(self, key, query, group):
        """Sample candiditates for a query per sample group"""
        difficulty, relevance = group
        percentiles_n_size = self.group_config[relevance][difficulty]
        text_percentile_range = percentiles_n_size['text_percentile_range']
        image_percentile_range = percentiles_n_size['image_percentile_range']
        size = percentiles_n_size['size']

        index_results = {index_name: {} for index_name in self.config['indices']}

        for index in self.config['indices'].keys():
            
            patent_index = self.indices[index]
            
            percentile_range = image_percentile_range
            if index in ['title','desc','label']:
                percentile_range = text_percentile_range

            max_threshold = self.thresholds_dict[index][key][percentile_range[0]]
            min_threshold = self.thresholds_dict[index][key][percentile_range[1]]

            results = patent_index.search_top_k_within_threshold(
                query_embedding=query[index],
                max_threshold=float(max_threshold),
                min_threshold=float(min_threshold)
            )

            for result in results:
                index_results[index][result['hashcode']] = {
                    'metadata': result,
                    'score': float(result['score'])
                }

        candidate_dict = self.merge_results(index_results)

        if len(candidate_dict.keys()) > 5:
            samples = self.sample_and_deduplicate(
                candidate_dict, n=size
            )
        else:
            samples = candidate_dict

        return samples
   
    def sample(self):
        """Sampling process"""

        query_dict = self.load_query_dict()

        for section in query_dict.keys():
            
            os.makedirs(f'data/candidates/{section}', exist_ok=True)

            for key in tqdm(list(query_dict[section].keys()), desc=f'Sampling candidates'):

                if os.path.exists(f'data/candidates/{section}/{key}.json'):
                    print(f'Skipping {key}')
                    continue

                candidates = {}

                candidates.setdefault('cited', {})
                candidates['cited'] = self.select_cited_candidates(
                    query=query_dict[section][key]
                )
                
                for group in GROUPS:

                    candidates.setdefault(f'{group[0]}_{group[1]}', {})
                    candidates[f'{group[0]}_{group[1]}'] = self.select_candidates(
                        key=key, query=query_dict[section][key], group=group
                    )

                save_json(candidates, f'data/candidates/{section}/{key}.json')
