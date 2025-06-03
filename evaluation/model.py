import fsspec

import torch
import open_clip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELNAME = 'ViT-B-16'

MODELS = {
    'clip_base': None,
    'clip_desc': 'patent_clip_desc_aug.pt',
    'clip_labels': 'patent_clip_labels_aug.pt'
}

def pt_load(file_path, map_location=None):
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out

def load_model(modelname):
    
    checkpoint_path = MODELS[modelname]

    if checkpoint_path:
        model = open_clip.create_model(model_name=MODELNAME, pretrained=False)
        checkpoint = pt_load(checkpoint_path, map_location='cpu')
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        model = open_clip.create_model(model_name=MODELNAME, pretrained='laion400m_e32')

    return model