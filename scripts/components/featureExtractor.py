import numpy as np
import os
import pandas as pd
# import torchvision
import torch
from torch import nn
from torchvision import models, transforms
import cv2
from torch.utils.data import DataLoader, Dataset
import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random

class FeatureExtractor(nn.Module):
	def __init__(self):
		super(FeatureExtractor, self).__init__()
		self.feature_extractor = models.resnet50(pretrained=True)
		self.feature_extractor.fc = nn.Identity()
		self.features = nn.Flatten()
	
	def forward(self,x):
		out = self.feature_extractor(x)
		# features = self.features(out)
		return out