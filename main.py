import os
import numpy as np
from signver.extractor import MetricExtractor

DIR = os.path.dirname(__file__)
EXTRACTOR_MODEL_PATH = "models/extractor/metric"
extractor_model = MetricExtractor() 
extractor_model.load(os.path.join(DIR, EXTRACTOR_MODEL_PATH))

def extract(image_np):
    """
    Extracts the features from the given image.
    
    Parameters
    ----------
    image_np : numpy.ndarray
        The image to extract features from.
    
    Returns
    -------
    The extracted features.
    """
    return extractor_model.extract(image_np)

def signature_feature_extractor(signatures):
    """
    This function accepts a batch of signatures in the form of numpy arrays.
    It extracts features from each signature and returns the feature vectors.
    
    Parameters
    ----------
    signatures : list or numpy array
        A batch of signatures in the form of numpy arrays.
        
    Returns
    -------
    sign_features : numpy array
        A batch of feature vectors extracted from the signatures.
    """
    if isinstance(signatures, list):
        sign_features = extract(np.array(signatures) / 255)
    else:
        sign_features = extract(signatures)
    
    return sign_features