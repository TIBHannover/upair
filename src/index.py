import io
import os

import torch
import numpy as np

from PIL import Image

np.random.seed(42)
torch.manual_seed(42)

import torch.nn as nn

from tqdm import tqdm
import webdataset as wds

from dataset import PatentDataset

from encoders import (
    PaecterEncoder, PatentNetEncoder, CLIPEncoder, ResNext101Encoder
)

from utils import load_config, load_json, clean_text, preprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = load_config(path='src/config.yaml')

pub2title = load_json('data/pub2title.json')

text_encoder = PaecterEncoder()
clip_encoder = CLIPEncoder()
patentnet_encoder = PatentNetEncoder()
resnext_encoder = ResNext101Encoder()

def fetch_title(_key):
    try:
        return pub2title[''.join(_key.split('_')[:-2])]['title']
    except KeyError:
        return None

def save_numpys(
    index_dir, shard_idx, keys, hashcodes,
    titles, descs, labels,
    clip_embs, patenet_embs, resnext_embs
):
    """Save numpy files for each data array in the specified shard directory."""

    shard_dir = os.path.join(index_dir, f'shard-{shard_idx}')
    os.makedirs(shard_dir, exist_ok=True)

    data_dict = {
        'keys': keys,
        'hashcodes': hashcodes,
        'titles': titles,
        'descs': descs,
        'labels': labels,
        'clip': clip_embs,
        'patentnet': patenet_embs,
        'resnext': resnext_embs
    }

    for var_name, data in data_dict.items():
        if data:
            file_path = os.path.join(shard_dir, f'{var_name}.npy')
            np.save(file_path, np.array(data))

    return True

def index_dataset():
    """Save embeddings."""

    ROOT_DIR = config['index']['root_dir']
    BATCH_SIZE = config['index']['batch_size']
    NUM_WORKERS = config['index']['num_workers']
    SAVE_EVERY = config['index']['save_every']
    INDEX_DIR = config['index_dir']

    os.makedirs(INDEX_DIR, exist_ok=True)

    # Load index dataset
    dataset = PatentDataset(
        root=f'{ROOT_DIR}/',
        split='index',
        figure_types='drawing',
        reference_dirs='w-ref',
        outputs={
            '__key__': lambda x: x,
            'hashcode.txt': lambda x: x.decode('utf-8'),
            'image.png': lambda x: Image.open(io.BytesIO(x)), 
            'desc.txt': lambda x: x.decode('utf-8'),
            'labels.txt': lambda x: x.decode('utf-8')
        },
    )

    dataloader = wds.WebLoader(
        dataset, num_workers=NUM_WORKERS, shuffle=False, batch_size=BATCH_SIZE
    )

    # Save embeddings
    shard_idx = 0
    keys, hashcodes, titles, descs, labels = [],[],[],[],[]
    clip_embs, patenet_embs, resnext_embs = [], [], []

    for i, batch in tqdm(enumerate(dataloader, start=1)):

        batch_keys, batch_hashcodes, batch_images, batch_descs, batch_labels = batch

        batch_titles = [fetch_title(x) for x in batch_keys]
        batch_descs = [clean_text(x) for x in batch_descs]
        batch_labels = [
            ' '.join([y for y in x.split('__SEP__') if y != 'None'])
            for x in batch_labels
        ]

        batch_data = list(
            zip(batch_keys, batch_hashcodes, batch_images, batch_titles, batch_descs, batch_labels,)
        )

        filtered_batch = [
            (key, hashcode, image, title, desc, label)
            for key, hashcode, image, title, desc, label in batch_data
            if key and hashcode and title and desc != '' and label != ''
        ]

        if len(filtered_batch) > 0:

            batch_keys, batch_hashcodes, batch_images, batch_titles, batch_descs, batch_labels = zip(*filtered_batch)

            keys.extend(batch_keys)
            hashcodes.extend(batch_hashcodes)
            
            titles.extend(text_encoder.encode(texts=batch_titles, batch_size=BATCH_SIZE))
            descs.extend(text_encoder.encode(texts=batch_descs, batch_size=BATCH_SIZE))
            labels.extend(text_encoder.encode(texts=batch_labels, batch_size=BATCH_SIZE))

            clip_embs.extend(
                clip_encoder.encode(
                    images=[preprocess(img, size=(378,378)) for img in batch_images]
                )
            )
            patenet_embs.extend(
                patentnet_encoder.encode(
                    images=[
                        preprocess(img, size=(384,384), num_output_channels=3)
                        for img in batch_images
                    ]
                )
            )
            resnext_embs.extend(
                clip_encoder.encode(
                    images=[preprocess(img, size=(384,384)) for img in batch_images]
                )
            )

            if i % SAVE_EVERY == 0:
                save_numpys(
                    INDEX_DIR, shard_idx, keys, hashcodes,
                    titles, descs, labels,
                    clip_embs, patenet_embs, resnext_embs
                )
                shard_idx += 1
                keys, hashcodes, titles, descs, labels = [],[],[],[],[]
                clip_embs, patenet_embs, resnext_embs = [], [], []

    # Save last batch
    save_numpys(
        INDEX_DIR, shard_idx, keys, hashcodes,
        titles, descs, labels,
        clip_embs, patenet_embs, resnext_embs
    )

    print("Done!")

if __name__ == '__main__':
    index_dataset()
