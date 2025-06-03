import re
import json
import yaml
import csv

from PIL import Image

from torchvision import transforms as T

from constants import (
    IMAGENET_MEAN, IMAGENET_STD,
    PATENTNET_MEAN, PATENTNET_STD
)

BICUBIC = T.InterpolationMode.BICUBIC

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_json(filepath):
    with open(filepath, 'r') as rf:
        return json.load(rf)

def save_json(data, filepath):
    with open(f'{filepath}', 'w') as wf:
        json.dump(data, wf, indent=4)

def save_csv(data, filepath):
    with open(f'{filepath}.csv', 'w') as wf:
        writer = csv.writer(wf)
        writer.writerows(data)

def remove_figure_text(text):
    figure_text = (
        r'(?:abb|fig)[.]*[ure]*[s|n]*[. ]+[a-z]*[0-9]*[,0-9]*[(]*[a-z|0-9.,]*[)]*[-]*'
        + r'[ and | to | through | und ]*[0-9]*[(]*[a-z|0-9.,]*[)]*[:]*'
    )   # Regex for figure text
    return re.sub(
        figure_text, '<image>', text, flags=re.IGNORECASE
    )   # Remove figure text


def clean_text(text):

    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = remove_figure_text(text)
    return text

def pad(image):
    """Pad image to square"""
    width, height = image.size
    if width == height:
        return image
    elif width > height:
        result = Image.new('RGB', (width, width), color=(255, 255, 255))
        result.paste(image, (0, (width - height) // 2))
    else:
        result = Image.new('RGB', (height, height), color=(255, 255, 255))
        result.paste(image, ((height - width) // 2, 0))

    return result.convert('RGB')

def resize(image, size):
    return T.Resize(size=size, interpolation=BICUBIC, antialias=True)(image)

def convert_to_grayscale(image, num_output_channels):
    return T.Grayscale(num_output_channels=num_output_channels)(image)

def convert_to_rgb(image):
    return image.convert('RGB')

def normalize(tensor):
    mean = IMAGENET_MEAN if tensor.shape[0] == 3 else PATENTNET_MEAN
    std = IMAGENET_STD if tensor.shape[0] == 3 else PATENTNET_STD
    return T.Normalize(mean=mean, std=std)(tensor)

def to_tensor(image):
    return T.ToTensor()(image)

def preprocess(image, size=(224, 224), num_output_channels=3):

    return T.Compose([
        pad,
        lambda x: resize(x, size),
        lambda x: convert_to_grayscale(x, num_output_channels),
        to_tensor,
        normalize
    ])(image)