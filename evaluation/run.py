import os

import torch
import numpy as np

from PIL import Image

import open_clip
from tqdm import tqdm

import ir_measures
from ir_measures import MAP, nDCG, MRR, Recall

from model import load_model
from utils import clean_text, preprocess, load_json

METADATA_FILE = "metadata.json"
IMAGE_ROOT = "dataset/test/images"
MODEL_NAME = "ViT-B-16"
PRETRAINED = "laion400m_e32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = open_clip.get_tokenizer(MODEL_NAME)

def load_image(hashcode):
    img_path = os.path.join(IMAGE_ROOT, f"{hashcode}.png")
    img = Image.open(img_path)
    return preprocess(img).unsqueeze(0).to(DEVICE)

def encode_image(model, hashcode):
    img = load_image(hashcode)
    with torch.no_grad():
        feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat

def encode_text(model, text):
    text = clean_text(text)
    tokens = tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat

def cosine_sim(a, b):
    return (a @ b.T).item()

def save_trec_run(run_dict, filename, run_name):
    with open(filename, "w") as f:
        for qid, docs in run_dict.items():
            ranked = sorted(docs, key=lambda x: -x[1])
            for rank, (cid, score) in enumerate(ranked, 1):
                f.write(f"{qid} Q0 {cid} {rank} {score} {run_name}\n")
    print(f"Saved TREC run file: {filename}")

def minmax_norm(arr):
    arr = np.array(arr)
    if arr.size == 0:
        return arr
    min_val = arr.min()
    max_val = arr.max()
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

evaluation_dataset = load_json(METADATA_FILE)

def run(model_name):

    run_image = {}
    run_img_desc = {}
    run_img_labels = {}

    model = load_model(model_name)
    model.to(DEVICE)
    model.eval()

    for qhash, qdata in tqdm(evaluation_dataset.items()):
        query_img_feat = encode_image(model, qhash)
        if query_img_feat is None:
            continue
        query_desc_feat = encode_text(model, qdata["query"]["description"])
        query_labels_feat = encode_text(model, qdata["query"]["component_labels"])

        cands = []
        for cand in qdata["candidates"]:
            chash = cand["hashcode"]
            cand_img_feat = encode_image(model, chash)
            if cand_img_feat is None:
                continue
            cand_desc_feat = encode_text(model, cand["description"])
            cand_labels_feat = encode_text(model, cand["component_labels"])

            img_score = cosine_sim(query_img_feat, cand_img_feat)
            desc_score = cosine_sim(query_desc_feat, cand_desc_feat)
            labels_score = cosine_sim(query_labels_feat, cand_labels_feat)

            cands.append({
                "hashcode": chash,
                "img_score": img_score,
                "desc_score": desc_score,
                "labels_score": labels_score
            })

        img_scores = [c["img_score"] for c in cands]
        desc_scores = [c["desc_score"] for c in cands]
        labels_scores = [c["labels_score"] for c in cands]

        img_scores_norm = minmax_norm(img_scores)
        desc_scores_norm = minmax_norm(desc_scores)
        labels_scores_norm = minmax_norm(labels_scores)

        run_image[qhash] = []
        run_img_desc[qhash] = []
        run_img_labels[qhash] = []

        for i, cand in enumerate(cands):
            chash = cand["hashcode"]
            run_image[qhash].append((chash, cand["img_score"]))
            fused_img_desc = img_scores_norm[i] + desc_scores_norm[i]
            run_img_desc[qhash].append((chash, fused_img_desc))
            fused_img_labels = img_scores_norm[i] + labels_scores_norm[i]
            run_img_labels[qhash].append((chash, fused_img_labels))

    save_trec_run(run_image, f"results/{model_name}_image.run", f"{model_name}_image")
    save_trec_run(run_img_desc, f"results/{model_name}_imgdesc_norm.run", f"{model_name}_imgdesc_norm")
    save_trec_run(run_img_labels, f"results/{model_name}_imglabels_norm.run", f"{model_name}_imglabels_norm")

def evaluate(model_name):
    qrels_files = [
        ("dataset/test/qrels_visual.txt", "Visual"),
        ("dataset/test/qrels_semantic.txt", "Semantic"),
        ("dataset/test/qrels_part_whole.txt", "Part-Whole"),
    ]
    run_files = [
        (f"results/{model_name}_image.run", f"CLIP Image ({model_name})"),
        (f"results/{model_name}_imgdesc_norm.run", f"CLIP Image+Desc (Norm Late Fusion) ({model_name})"),
        (f"results/{model_name}_imglabels_norm.run", f"CLIP Image+Labels (Norm Late Fusion) ({model_name})"),
    ]
    metrics = [MAP, nDCG@10, MRR, Recall@1]

    for qrels_file, rel_name in qrels_files:
        print(f"\n=== Evaluating for {rel_name} relevance ({qrels_file}) ===")
        for run_file, run_name in run_files:
            print(f"\n--- Results for {run_name} ---")
            results = ir_measures.calc_aggregate(
                metrics,
                ir_measures.read_trec_qrels(qrels_file),
                ir_measures.read_trec_run(run_file)
            )
            for metric, value in results.items():
                print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    
    for model_name in ['clip_base', 'clip_desc', 'clip_labels']:
        run(model_name)
        evaluate(model_name)