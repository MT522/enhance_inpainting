import torch
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import logging
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_fid import fid_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os

class InpaintPipeline():
    def __init__(self, blip_model_dir, clip_model_dir, diffusion_model_dir, device, dataset_path):
        self.blip_model_dir = blip_model_dir
        self.clip_model_dir = clip_model_dir
        self.diffusion_model_dir = diffusion_model_dir
        
        self.device = device

        # Load model directly
        self.blip_processor = AutoProcessor.from_pretrained(blip_model_dir)
        self.blip_model = AutoModelForImageTextToText.from_pretrained(f"{blip_model_dir}-v2").to(self.device)

        # Load CLIP model
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_dir)
        self.clip_model = CLIPModel.from_pretrained(clip_model_dir).to(self.device)

        # Load Stable Diffusion pipeline
        self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(diffusion_model_dir,
            torch_dtype=torch.float16).to(self.device)
        logging.disable_progress_bar()

        self.dataset_path = dataset_path        
        self._load_dataset_paths()

        self.metrics = {
            "filename": [],
            "ssim_baseline": [],
            "ssim_new": [],
            "psnr_baseline": [],
            "psnr_new": [],
            "clip_score_baseline": [],
            "clip_score_new": [],
        }

    def _load_dataset_paths(self):
        self.image_files = [os.path.join(self.dataset_path, f) for i,f in enumerate(os.listdir(self.dataset_path)) if (f.endswith(".jpg") and i < 1000)]

    def generate_caption(self, image):
        """Generate a caption for the image using the BLIP model."""
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        out = self.blip_model.generate(**inputs)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def inpaint_image(self, image, mask, prompt):
        """Inpaint the masked region of the image using the Stable Diffusion model."""
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))
        inpainted_image = self.inpaint_pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=50,
        ).images[0]
        return inpainted_image

    def create_mask(self, image, bbox):
        """Create a binary mask for the image based on the bounding box."""
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(bbox, fill=255)
        return mask

    def create_random_mask(self, image, mask_size=(200, 200)):
        """
        Create a binary mask with a random position and fixed size.
        Args:
            image: PIL Image to create the mask for.
            mask_size: Tuple (width, height) for the size of the mask.
        Returns:
            mask: PIL Image with a random rectangular mask.
            bbox: Tuple (x1, y1, x2, y2) representing the bounding box of the mask.
        """
        width, height = image.size
        mask_width, mask_height = mask_size

        # Ensure the mask fits within the image
        x1 = random.randint(0, width - mask_width)
        y1 = random.randint(0, height - mask_height)
        x2 = x1 + mask_width
        y2 = y1 + mask_height

        # Create the mask
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((x1, y1, x2, y2), fill=255)

        return mask, (x1, y1, x2, y2)

    def overlay_mask_on_image(self, image, mask, bbox):
        """Overlay the mask on the image as a black rectangle."""
        image_with_mask = image.copy()
        draw = ImageDraw.Draw(image_with_mask)
        draw.rectangle(bbox, fill="black")
        return image_with_mask

    def calculate_ssim(self, ground_truth, inpainted):
        """Calculate SSIM between the ground truth and inpainted image."""
        ground_truth_gray = np.array(ground_truth.convert("L"))
        inpainted_gray = np.array(inpainted.convert("L"))
        return ssim(ground_truth_gray, inpainted_gray, data_range=255)

    def calculate_psnr(self, ground_truth, inpainted):
        """Calculate PSNR between the ground truth and inpainted image."""
        ground_truth_array = np.array(ground_truth)
        inpainted_array = np.array(inpainted)
        return psnr(ground_truth_array, inpainted_array, data_range=255)

    def calculate_fid(self, ground_truth_dir, inpainted_dir):
        """
        Calculate FID between two directories of images (ground truth and inpainted).
        """
        fid_value = fid_score.calculate_fid_given_paths(
            [ground_truth_dir, inpainted_dir],
            batch_size=32,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dims=2048,
        )
        return fid_value

    def calculate_clip_score(self, image, caption):
        """Calculate CLIP score between an image and a caption."""
        inputs = self.clip_processor(text=[caption], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return outputs.logits_per_image.item()

    def create_collage(self, image1, image2, image3, image4):
        """Create a 2x2 collage of the four images."""
        # Ensure all images are the same size
        image1 = image1.resize((512, 512))
        image2 = image2.resize((512, 512))
        image3 = image3.resize((512, 512))
        image4 = image4.resize((512, 512))

        # Create a blank canvas for the collage
        collage = Image.new("RGB", (1034, 1034))

        # Paste the images into the collage
        collage.paste(image1, (0, 0))  # Top-left
        collage.paste(image2, (522, 0))  # Top-right
        collage.paste(image3, (0, 522))  # Bottom-left
        collage.paste(image4, (522, 522))  # Bottom-right

        return collage

    def process_image(self, image_path, save_dir_inpainted, save_dir_inpainted_baseline, save_dir_collage):
        """Process a single image: inpainting, saving, and calculating metrics."""
        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512))
        mask, bbox = self.create_random_mask(image)
        image_with_mask = self.overlay_mask_on_image(image, mask, bbox)

        # Generate caption
        caption = self.generate_caption(image)
        enhanced_prompt = f"A high-quality image of {caption}, with the missing region filled in seamlessly."

        # Inpaint using both methods
        inpainted_image = self.inpaint_image(image, mask, enhanced_prompt)
        inpainted_image_baseline = self.inpaint_image(image, mask, "Fill the missing region.")
        collage = self.create_collage(image, image_with_mask, inpainted_image, inpainted_image_baseline)

        # Save images
        filename = os.path.basename(image_path)
        inpainted_image.save(os.path.join(save_dir_inpainted, filename))
        inpainted_image_baseline.save(os.path.join(save_dir_inpainted_baseline, filename))
        collage.save(os.path.join(save_dir_collage, filename))

        # Calculate metrics
        ssim_baseline = self.calculate_ssim(image, inpainted_image_baseline)
        ssim_new = self.calculate_ssim(image, inpainted_image)
        psnr_baseline = self.calculate_psnr(image, inpainted_image_baseline)
        psnr_new = self.calculate_psnr(image, inpainted_image)
        clip_score_baseline = self.calculate_clip_score(inpainted_image_baseline, caption)
        clip_score_new = self.calculate_clip_score(inpainted_image, enhanced_prompt)

        # Store metrics
        self.metrics["ssim_baseline"].append(ssim_baseline)
        self.metrics["filename"].append(filename)
        self.metrics["ssim_new"].append(ssim_new)
        self.metrics["psnr_baseline"].append(psnr_baseline)
        self.metrics["psnr_new"].append(psnr_new)
        self.metrics["clip_score_baseline"].append(clip_score_baseline)
        self.metrics["clip_score_new"].append(clip_score_new)

    def main(self):
        save_dir_inpainted = f"Results/{self.dataset_path}_inpainted"  # Directory to save new method inpainted images
        save_dir_inpainted_baseline = f"Results/{self.dataset_path}inpainted_baseline"  # Directory to save baseline inpainted images
        save_dir_collage = f"Results/{self.dataset_path}_collage"  # Directory to save collages
        os.makedirs(save_dir_inpainted, exist_ok=True)
        os.makedirs(save_dir_inpainted_baseline, exist_ok=True)
        os.makedirs(save_dir_collage, exist_ok=True)

        # Process all images
        for image_path in tqdm(self.image_files, desc="Processing images"):
            self.process_image(image_path, save_dir_inpainted, save_dir_inpainted_baseline, save_dir_collage)
        
        # fid_value_baseline = self.calculate_fid(self.dataset_path, save_dir_inpainted_baseline)
        # fid_value_new = self.calculate_fid(self.dataset_path, save_dir_inpainted)
        # print(f"Baseline fid value: {fid_value_baseline} vs new method fid distance: {fid_value_new}")

        # Save metrics to a CSV file
        metrics_df = pd.DataFrame(self.metrics)
        metrics_df.to_csv(f"Results/inpainting_metrics_{DATASET_PATH}.csv", index=False)

        # Calculate average performance
        numeric_cols = metrics_df.select_dtypes(include=np.number).columns  # Select numeric columns
        avg_metrics = metrics_df[numeric_cols].mean() # Calculate the mean of the numeric columns only

        # Plot average performance
        labels = ["SSIM", "PSNR", "CLIP Score"]
        baseline_avgs = [avg_metrics["ssim_baseline"], avg_metrics["psnr_baseline"], avg_metrics["clip_score_baseline"]]
        new_method_avgs = [avg_metrics["ssim_new"], avg_metrics["psnr_new"], avg_metrics["clip_score_new"]]

        x = np.arange(len(labels))
        width = 0.35

        _, ax = plt.subplots()
        _ = ax.bar(x - width/2, baseline_avgs, width, label="Baseline")
        _ = ax.bar(x + width/2, new_method_avgs, width, label="New Method")

        ax.set_ylabel("Scores")
        ax.set_title("Average Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.savefig(f"Results/average_performance_{DATASET_PATH}.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BLIP_MODEL_DIR = "blip"
CLIP_MODEL_DIR = "clip"
DIFFUSON_MODEL_DIR = "stabilityai/stable-diffusion-2-inpainting"

DATASET_PATH = "test2014"
# DATASET_PATH = "pascal-voc2012"
# DATASET_PATH = "genome"

if __name__ == "__main__":
    # login()

    inpaint = InpaintPipeline(blip_model_dir=BLIP_MODEL_DIR, clip_model_dir=CLIP_MODEL_DIR, diffusion_model_dir=DIFFUSON_MODEL_DIR,
                              device=DEVICE, dataset_path=DATASET_PATH)
    
    inpaint.main()
