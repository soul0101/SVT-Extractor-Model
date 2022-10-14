
import os
import numpy as np
from signver.extractor import MetricExtractor

DIR = os.path.dirname(__file__)
EXTRACTOR_MODEL_PATH = "models/extractor/metric"
extractor = MetricExtractor() 
extractor_model = extractor.load(os.path.join(DIR, EXTRACTOR_MODEL_PATH))

def extract(image_np):
    return extractor_model.extract(image_np)

def signature_feature_extractor(signatures):
    if isinstance(signatures, list):
        sign_features = extract(np.array(signatures) / 255)
    else:
        sign_features = extract(signatures)
    
    return sign_features