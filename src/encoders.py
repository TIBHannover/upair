import timm

import torch
import torch.nn as nn

import open_clip
import numpy as np

import warnings

from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PatentNet(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, embedding_size: int):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=1)
        in_features = 1024 # swin_base

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(),
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
    
class PatentNetEncoder:
    """PatentNet Encoder"""
    def __init__(
            self, model_name = 'swinv2_base_window12to24_192to384_22kft1k'
        ):
        super().__init__()
        self.model = PatentNet(model_name, pretrained=False, embedding_size=512)
        self.model.to(device)
        self.model = torch.nn.DataParallel(self.model)

        state_dict = torch.load('upair/swin-arcface-test_epoch19.pth', map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def encode(self, images):
        with torch.inference_mode():
            if isinstance(images, list): images = torch.stack(images)
            features = self.model(images.to(device))
            features /= features.norm(dim=-1, keepdim=True)
            return features.detach().cpu().numpy()
        
class CLIPEncoder:
    """CLIP Encoder"""
    def __init__(self, model_name='ViT-H-14-378-quickgelu', pretrained='dfn5b'):
        self.encoder = open_clip.create_model(
            model_name=model_name,
            pretrained=pretrained,
            device=device
        )
    
    def encode(self, images):
        with torch.inference_mode():
            if isinstance(images, list): images = torch.stack(images)
            image_features = self.encoder.encode_image(images.to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.detach().cpu().numpy()

class ResNext101Encoder:
    """ResNext101 Encoder"""
    def __init__(self, model_name='resnext101_32x8d'):
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0
        )
        self.model.to(device)
        self.model.eval()

    def encode(self, images):
        with torch.inference_mode():
            if isinstance(images, list): images = torch.stack(images)
            features = self.model(images.to(device))
            features /= features.norm(dim=-1, keepdim=True)
            return features.detach().cpu().numpy()
    
class PaecterEncoder:
    """Paecter Encoder"""
    def __init__(self, model_name='mpi-inno-comp/paecter'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(
            self, texts, batch_size=1, max_length=512, convert_to_numpy=True, normalize_embeddings=True, precision='float32'
        ):

        if isinstance(texts, str): texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            
            batch_texts = texts[i:i+batch_size]
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length
            ).to(device)

            with torch.no_grad():
                
                model_output = self.model(**encoded_input)
                embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                if precision == 'float32': embeddings = embeddings.float()
                elif precision == 'float16': embeddings = embeddings.half()
                elif precision == 'bfloat16': embeddings = embeddings.bfloat16()
                else: raise ValueError(f"Unsupported precision: {precision}")
                
                if convert_to_numpy: embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)

        if convert_to_numpy: return np.vstack(all_embeddings)
        else: return torch.cat(all_embeddings, dim=0)

