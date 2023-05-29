import numpy as np
import torch
import clip as openai_clip
import pandas as pd
from PIL import Image
import streamlit as st
import faiss

# OpenAI CLIP model
class CLIPOpenAIFaiss:
    def build_index(self):
        if self.image_paths is None:
            return

        st.info("Re-building FAISS index for new data...", icon="ℹ️")

        # IndexFlatIP: Exact Search for Inner Product
        self.index = faiss.IndexFlatIP(512)

        # Use image indexing. Loading image_features from saved torch tensors.
        df = pd.read_csv(self.index_lookup_file)
        image_features = []
        for image_path in self.image_paths:
            selected_row = df.loc[df["image_path"] == image_path].iloc[0]
            saved_path = selected_row["index_path"]
            img_feature = torch.load(saved_path, map_location=self.device)
            image_features.append(img_feature)
        # Shape: (N, 512)
        image_features = (
            torch.stack(image_features).squeeze(1).to(self.device).cpu().numpy()
        )
        self.index.add(image_features)

    def __init__(self, index_lookup_file, image_paths, k_neighbors=5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = openai_clip.load("ViT-B/32", device=self.device)
        self.index_lookup_file = index_lookup_file
        self.k_neighbors = k_neighbors
        self.image_paths = image_paths

        self.build_index()

    def get_similarity_scores(self, image_paths, query):
        if not np.array_equal(image_paths, self.image_paths):
            self.image_paths = image_paths
            self.build_index()

        text_tokens = openai_clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            # torch.Size([1, 512])
            text_features = self.model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # Shape (1, 512)
        text_features = text_features.cpu().numpy()

        # Find nearest k_neighbors. similarity, indexes has Shape (1, k_neighbors)
        similarity, indexes = self.index.search(text_features, self.k_neighbors)

        df = pd.DataFrame()
        df["image_path"] = np.take(self.image_paths, indexes[0], axis=0)
        df["score"] = similarity[0] * 100
        return df

    def get_similarity(self, image_paths, input_image):
        if not np.array_equal(image_paths, self.image_paths):
            self.image_paths = image_paths
            self.build_index()
        
        image = self.preprocess(Image.open(input_image)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # torch.Size([1, 512])
            image_features = self.model.encode_image(image).float()
            
        image_features /= image_features.norm(dim=-1, keepdim=True)
        # Shape (1, 512)
        image_features = image_features.cpu().numpy()

        # Find nearest k_neighbors. similarity, indexes has Shape (1, k_neighbors)
        similarity, indexes = self.index.search(image_features, self.k_neighbors)

        df = pd.DataFrame()
        df["image_path"] = np.take(self.image_paths, indexes[0], axis=0)
        df["score"] = similarity[0] * 100
        return df