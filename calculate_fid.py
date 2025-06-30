import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy.stats import multivariate_normal
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from scipy import linalg

# 1. Load Inception Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception_model = inception_v3(pretrained=True, transform_input=False).to(device)  # Set transform_input=False
inception_model.eval()

# 2. Image Preprocessing
img_size = 299  # InceptionV3 expects 299x299 images
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

def get_activations(image_batch):
  """Calculates activations for a batch of images."""
  with torch.no_grad():
      image_batch = image_batch.to(device)
      # InceptionV3 returns a tuple in "eval" mode. We are interested in the first element
      # which contains the pool5 activations.
      pred = inception_model(image_batch)

      if isinstance(pred, tuple):
        pred = pred[0]

      # If the model has an auxiliary logits layer, take only the primary output.
      if isinstance(pred, tuple):
        pred = pred[0]

      return pred.cpu().numpy()

def calculate_fid(real_features, generated_features):
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    mu2 = np.mean(generated_features, axis=0)
    sigma2 = np.cov(generated_features, rowvar=False)

    try:
        sqrt_sigma = linalg.sqrtm(sigma1 @ sigma2) # Correct: Use scipy.linalg.sqrtm
    except linalg.LinAlgError:
        print("Square root of matrix singular. Using trace approximation.")
        offset = np.eye(sigma1.shape[0]) * 1e-6 # Add small offset to diagonal
        sqrt_sigma = linalg.sqrtm(sigma1 @ sigma2 + offset)

    fid_score = np.real(np.sum(np.square(mu1 - mu2)) + np.trace(sigma1 + sigma2 - 2 * sqrt_sigma)) # Use np.real

    return fid_score

def process_images(image_dir):
    """Loads and preprocesses images from a directory."""
    image_files = [f for i, f in enumerate(os.listdir(image_dir)) if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg')) and i < 1000]

    num_images = len(image_files)
    all_activations = []

    batch_size = 64 # Adjust batch size as needed depending on available memory

    for i in tqdm(range(0, num_images, batch_size), desc=f"Processing {image_dir}"):
        image_batch = []
        for j in range(i, min(i + batch_size, num_images)):
            image_path = os.path.join(image_dir, image_files[j])
            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = transform(image)
                image_batch.append(image_tensor)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

        if image_batch: # Check if image_batch is not empty
            image_batch = torch.stack(image_batch)
            activations = get_activations(image_batch)
            all_activations.extend(activations)

    return np.array(all_activations)

# Example Usage:
real_image_dir = "genome"  # Replace with your real image directory
generated_image_dir = "Results/genome_inpainted"  # Replace with your generated image directory
generated_image_dir_baseline = "Results/genomeinpainted_baseline"  # Replace with your generated image directory

real_activations = process_images(real_image_dir)
generated_activations = process_images(generated_image_dir)
generated_activations_baseline = process_images(generated_image_dir_baseline)

fid_value = calculate_fid(real_activations, generated_activations)
fid_value_baseline = calculate_fid(real_activations, generated_activations_baseline)
print(f"FID new method: {fid_value}\n\n")
print(f"FID  baseline: {fid_value_baseline}")