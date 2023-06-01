import cv2
import numpy as np

from .utils import check_has_nan
from scipy.stats import entropy
from skimage.filters import gabor
from skimage.feature import canny
from skimage.feature import local_binary_pattern


EPSYLON=0.001


def extract_features(image, use_gabor=True, use_lbp=False):
    if use_gabor:
        gabor_feat = np.array(extract_gabor_features(image))
        if check_has_nan(gabor_feat):
            print("piece has nan in gabor")
    
    hist_feat = np.array(extract_histogram_features(image))
    if check_has_nan(hist_feat):
        print("piece has nan in hist")

    if use_lbp:
        lbp_feat = np.array(extract_lbp_features(image))
        if check_has_nan(lbp_feat):
            print("piece has nan in lbp")
    
    if use_gabor and use_lbp:
        full_feat = np.concatenate((gabor_feat,hist_feat,lbp_feat))
    
    elif not use_gabor and use_lbp:
        full_feat = np.concatenate((hist_feat,lbp_feat))
    
    else:
        full_feat = np.concatenate((gabor_feat,hist_feat))

    return full_feat


def extract_gabor_features(image, orientations=4, blur=False, scales=(3, 5), lambdas=np.arange(EPSYLON, np.pi+EPSYLON, np.pi / 2), gammas=([0.1])):
    """
    Extract Gabor features from an image using Gabor filters.
    
    Parameters:
        - image: Input image for feature extraction.
        - orientations: Number of orientations for the Gabor filters.
        - scales: Scales of the Gabor filters.
        - lambdas: Spatial frequencies (wavelengths) of the Gabor filters.
        - gammas: Spatial aspect ratios of the Gabor filters.
    
    Returns:
        - features: Extracted Gabor features as a 1D numpy array.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    if blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    feature_lists = []
    
    for theta in range(orientations):
        theta_radians = theta / orientations * np.pi
        for sigma in scales:
            for lambd in lambdas:
                for gamma in gammas:
                    try:
                        # Create the Gabor kernel
                        kernel_real, kernel_imag = gabor(gray, frequency=1/lambd, theta=theta_radians, sigma_x=sigma, sigma_y=sigma)
                    
                        # Apply the Gabor filter to the image and get the filtered response
                        filtered = np.abs(cv2.filter2D(gray, cv2.CV_64F, kernel_real))  # Magnitude response
                        
                        # Calculate statistical measures from the filtered response
                        energy = np.sum(filtered**2)
                        ent=entropy(filtered.flatten())
                        feature_lists.append([ent, energy])
                        
                    except (ZeroDivisionError, ValueError, RuntimeWarning) as e:
                        # Handle the specific error or warning here
                        # For example, you can skip the iteration or assign a default value
                        print("Error encountered with parameters:")
                        print("Theta:", theta)
                        print("Sigma:", sigma)
                        print("Lambda:", lambd)
                        print("Gamma:", gamma)
                        print("Error message:", str(e))
    
    # Concatenate the feature lists into a single feature vector
    features = np.concatenate(feature_lists)
    
    # Normalize the feature vector
    features /= np.max(features)
    return features


def extract_histogram_features(image, color_space='RGB', normalize='total', preprocessing_method=None):
    if preprocessing_method is not None:
        # Apply the selected preprocessing method
        if preprocessing_method == 'denoise':
            # Apply denoising using cv2.fastNlMeansDenoisingColored()
            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        elif preprocessing_method == 'equalize':
            # Apply histogram equalization using cv2.equalizeHist()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.equalizeHist(image)
        elif preprocessing_method == 'blur':
            # Apply Gaussian blur using cv2.GaussianBlur()
            image = cv2.GaussianBlur(image, (5, 5), 0)
        else:
            raise ValueError('Invalid preprocessing method')
    
    if color_space == 'RGB':
        # Split RGB channels
        r, g, b = cv2.split(image)
        # Compute the histogram for each RGB channel
        hist_r = np.histogram(r.ravel(), bins=256, range=(0, 255))[0]
        hist_g = np.histogram(g.ravel(), bins=256, range=(0, 255))[0]
        hist_b = np.histogram(b.ravel(), bins=256, range=(0, 255))[0]
        
        # Concatenate the histograms to create a feature vector
        hist = np.concatenate([hist_r, hist_g, hist_b])
    
    elif color_space == 'LAB':
        # Convert to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split LAB channels
        l, a, b = cv2.split(lab_image)
        
        # Compute the histogram for each LAB channel
        hist_l = np.histogram(l.ravel(), bins=256, range=(0, 255))[0]
        hist_a = np.histogram(a.ravel(), bins=256, range=(0, 255))[0]
        hist_b = np.histogram(b.ravel(), bins=256, range=(0, 255))[0]
        
        # Concatenate the histograms to create a feature vector
        hist = np.concatenate([hist_l, hist_a, hist_b])
    
    elif color_space == 'HSV':
        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Split HSV channels
        h, s, v = cv2.split(hsv_image)
        
        # Compute the histogram for each HSV channel
        hist_h = np.histogram(h.ravel(), bins=256, range=(0, 255))[0]
        hist_s = np.histogram(s.ravel(), bins=256, range=(0, 255))[0]
        hist_v = np.histogram(v.ravel(), bins=256, range=(0, 255))[0]
        
        # Concatenate the histograms to create a feature vector
        hist = np.concatenate([hist_h, hist_s, hist_v])
    
    # Normalize the histograms
    if normalize == 'individual':
        # Normalize each channel individually
        if color_space == 'RGB':
            hist_r = hist_r / np.sum(hist_r)
            hist_g = hist_g / np.sum(hist_g)
            hist_b = hist_b / np.sum(hist_b)
            hist_normalized = np.concatenate([hist_r, hist_g, hist_b])
        elif color_space == 'LAB':
            hist_l = hist_l / np.sum(hist_l)
            hist_a = hist_a / np.sum(hist_a)
            hist_b = hist_b / np.sum(hist_b)
            hist_normalized = np.concatenate([hist_l, hist_a, hist_b])
        elif color_space == 'HSV':
            hist_h = hist_h / np.sum(hist_h)
            hist_s = hist_s / np.sum(hist_s)
            hist_v = hist_v / np.sum(hist_v)
            hist_normalized = np.concatenate([hist_h, hist_s, hist_v])
    
    else:
        # Normalize the concatenated histogram
        hist_normalized = hist / np.sum(hist)
    
    return hist_normalized


def extract_lbp_features(image, use_canny=True, radius=3, n_points=8):
    """
    Extract Local Binary Patterns (LBP) features from an image.
    
    Parameters:
        - image: Input image for feature extraction.
        - radius: Radius of the circular neighborhood around each pixel.
        - n_points: Number of sampling points to use in the circular neighborhood.
    
    Returns:
        - features: Extracted LBP features as a 1D numpy array.
    """
    # Convert the image to gray-scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if use_canny:
        preprocessed_img = canny(gray)
    else:
        preprocessed_img = gray
    
    # Extract LBP features
    lbp = local_binary_pattern(preprocessed_img, n_points, radius, method='uniform')
    
    # Calculate the histogram of LBP values
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    # Flatten the histogram and return as features
    features = hist.flatten()
    
    return features
