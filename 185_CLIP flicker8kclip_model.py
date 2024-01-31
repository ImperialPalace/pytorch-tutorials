
#%%
#copy from https://github.com/RustamyF/clip-multimodal-ml
# CLIP Model and The Importance of Multimodal Embeddings - https://towardsdatascience.com/clip-model-and-the-importance-of-multimodal-embeddings-1c8f6b13bf72#:~:text=The%20CLIP%20loss%20aims%20to,the%20N%C2%B2%20%E2%88%92%20N%20incorrect%20pairings.

# %%
import torch
import os
import subprocess
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel, AutoTokenizer, BertTokenizer

import json
from PIL import Image
from torch.utils.data import Dataset
import collections
from torchvision import transforms
from datasets import load_dataset
from dataclasses import dataclass
import numpy as np
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#%%
@dataclass
class Config:
    """
    Configuration class for the CLIP training script.
    """

    embed_dim: int = 512  # Embedding dimension
    transformer_embed_dim: int = 768  # Transformer embedding dimension
    max_len: int = 32  # Maximum text length
    text_model: str = "distilbert-base-multilingual-cased"  # Text model name
    epochs: int = 5  # Number of training epochs
    batch_size: int = 128  # Batch size
    image_path = "./dataset/flickr8k/Images/"
    captions_path = "./dataset/flickr8k/"
    
#%%
class CocoDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        annotations_dir = os.path.join(root_dir, "annotations")
        annotation_file = os.path.join(
            annotations_dir, "annotations", "captions_train2017.json"
        )

        self.caption_list, self.image_path_list = self.load_annotations(annotation_file)

    def load_annotations(self, annotation_file):
        with open(annotation_file, "r") as f:
            annotations = json.load(f)["annotations"]

        image_path_to_caption = collections.defaultdict(list)
        for element in annotations:
            caption = f"{element['caption'].lower().rstrip('.')}"
            image_path = os.path.join(
                self.root_dir,
                "train2017",
                "train2017",
                "%012d.jpg" % (element["image_id"]),
            )
            image_path_to_caption[image_path].append(caption)
        image_paths = list(image_path_to_caption.keys())
        caption_list, image_path_list = self.training_list(
            image_paths, image_path_to_caption
        )

        return caption_list, image_path_list

    def training_list(self, image_paths, image_path_to_caption):
        captions_per_image = 2
        caption_list = []
        image_path_list = []
        for image_path in image_paths:
            captions = image_path_to_caption[image_path][:captions_per_image]
            caption_list.extend(captions)
            image_path_list.extend([image_path] * len(captions))

        return caption_list, image_path_list

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        caption = self.caption_list[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"image": image, "caption": caption}


class Flickr30kDataset(Dataset):
    def __init__(self):
        self.dataset = load_dataset("nlphuji/flickr30k", cache_dir="./huggingface_data")
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.cap_per_image = 2

    def __len__(self):
        return self.dataset.num_rows["test"] * self.cap_per_image

    def __getitem__(self, idx):
        original_idx = idx // self.cap_per_image
        # image_path = self.dataset[idx]["image_path"]
        image = self.dataset["test"][original_idx]["image"].convert("RGB")
        image = self.transform(image)

        # You might need to adjust the labels based on your task
        caption = self.dataset["test"][original_idx]["caption"][
            idx % self.cap_per_image
        ]

        return {"image": image, "caption": caption}

class Flickr8kDataset(torch.utils.data.Dataset):
    def __init__(self):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        
        self.image_filenames, self.captions = self.load_data()
        
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        item = {}
      
        img_path = f"{Config.image_path}/{self.image_filenames[idx]}"
        img = Image.open(img_path)
        
        item['image'] = self.transform(img)
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)
    
    def load_data(self):
        dataframe = pd.read_csv(f"{Config.captions_path}/captions1000.csv")
        max_id = dataframe["id"].max() + 1
        image_ids = np.arange(0, max_id)
        np.random.seed(42)
        valid_ids = np.random.choice(
            image_ids, size=int(0.2 * len(image_ids)), replace=False
        )
        train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
        train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
        
        # valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    
        return train_dataframe["image"].values, list(train_dataframe["caption"].values)
    
#%%
class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class VisionEncoder(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        base = models.resnet34(pretrained=True)
        d_in = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.projection = Projection(d_in, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        projected_vec = self.projection(self.base(x))
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class TextEncoder(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained(Config.text_model)
        self.projection = Projection(Config.transformer_embed_dim, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(x)[0]
        out = out[:, 0, :]  # get CLS token output
        projected_vec = self.projection(out)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class CustomModel(nn.Module):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder(Config.embed_dim)
        self.caption_encoder = TextEncoder(Config.embed_dim)
        self.tokenizer = Tokenizer(
            # AutoTokenizer.from_pretrained(Config.text_model), use_fast=False
            AutoTokenizer.from_pretrained(Config.text_model)
        )
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, images, text):
        text = self.tokenizer(text).to(self.device)

        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text["input_ids"])
        similarity = caption_embed @ image_embed.T

        loss = CLIP_loss(similarity)
        img_acc, cap_acc = metrics(similarity)
        return loss, img_acc, cap_acc


class Tokenizer:
    def __init__(self, tokenizer: BertTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, x: str) -> AutoTokenizer:
        return self.tokenizer(
            x,
            max_length=Config.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

#%%
def CLIP_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate a custom cross-entropy loss.

    Args:
    - logits (torch.Tensor): The input tensor containing unnormalized logits.

    Returns:
    - torch.Tensor: The computed custom cross-entropy loss.

    Example:
    >>> logits = torch.rand((batch_size, num_classes))
    >>> loss = CLIP_loss(logits)
    """

    n = logits.shape[1]

    # Create labels tensor
    labels = torch.arange(n)

    # bring logits to cpu
    logits = logits.to("cpu")

    # Calculate cross entropy losses along axis 0 and 1
    loss_i = F.cross_entropy(logits.transpose(0, 1), labels, reduction="mean")
    loss_t = F.cross_entropy(logits, labels, reduction="mean")

    # Calculate the final loss
    loss = (loss_i + loss_t) / 2

    return loss


def metrics(similarity: torch.Tensor):
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc

#%%
dataset = "Flickr8k"
# Create the CLIP dataset
if dataset == "coco":
    if not "datasets" in os.listdir():
        print("coco dataset is not downloaded! running the downloading script ....")
        subprocess.run(["python", "src/download_coco_data.py"])

    clip_dataset = CocoDataset(root_dir="datasets")
elif dataset == "Flickr30k":
    clip_dataset = Flickr30kDataset()
    
elif dataset == "Flickr8k":
    clip_dataset = Flickr8kDataset()
    
else:
    print("dataset is not exist!")

# Create the DataLoader
clip_dataloader = DataLoader(
    clip_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4
)

device = "cuda" if torch.cuda.is_available() else "cpu"
# Create an instance of your model
model = CustomModel().to(device)

# Define optimizer
optimizer = torch.optim.Adam(
    [
        {"params": model.vision_encoder.parameters()},
        {"params": model.caption_encoder.parameters()},
    ],
    lr=model.lr,
)


# Dummy training and validation loops
num_epochs = 5
batch_zero = True
for epoch in range(num_epochs):
    model.train()
    for batch in clip_dataloader:
        image = batch["image"].to(device)
        text = batch["caption"]
        # images, text = batch
        loss, img_acc, cap_acc = model(image, text)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_zero:
            print(f"Epoch [{0}/{num_epochs}], Batch Loss: {loss.item()}")
            batch_zero = False

    # Print training statistics
    print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item()}")

print("Training complete.")