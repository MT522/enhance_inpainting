import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm
import pandas as pd
import os
from PIL import Image

# 1. Modified BLIP Model with MLP Head
class BLIPWithMetricsHead(nn.Module):
    def __init__(self, blip_model, num_layers_to_train=2, hidden_size=768):  # Added num_layers_to_train
        super().__init__()
        self.blip_model = blip_model

        # Freeze BLIP's original layers initially
        for param in self.blip_model.parameters():
            param.requires_grad = False

        self.mlp_head = nn.Sequential( # MLP Head
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # Output: SSIM, PSNR, CLIP
        )

        self.max_text_length = self.blip_model.config.text_config.max_position_embeddings

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        device = pixel_values.device

        # Create dummy input_ids
        input_ids = torch.zeros((batch_size, self.max_text_length), dtype=torch.long).to(device)  # Dummy input_ids

        outputs = self.blip_model(pixel_values=pixel_values, input_ids=input_ids) # Provide input_ids
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # CLS token
        metrics_pred = self.mlp_head(last_hidden_state)
        return metrics_pred

# 2. Dataset for Fine-tuning
class MetricsDataset(Dataset):
    def __init__(self, csv_file, blip_processor, dataset_path):
        self.df = pd.read_csv(csv_file)
        self.blip_processor = blip_processor
        self.image_dir = dataset_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.df.iloc[idx]['filename'])
        image = Image.open(image_path).convert("RGB")
        inputs = self.blip_processor(image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)  # Remove batch dimension
        ssim = torch.tensor(self.df.iloc[idx]['ssim_new'], dtype=torch.float32)
        psnr = torch.tensor(self.df.iloc[idx]['psnr_new'], dtype=torch.float32)
        clip = torch.tensor(self.df.iloc[idx]['clip_score_new'], dtype=torch.float32)

        metrics_true = torch.stack([ssim, psnr, clip])
        return pixel_values, metrics_true

def weighted_loss(ssim_diff, psnr_diff, clip_diff, weights=(0.4, 0.5, 0.3)):
    """
    Compute a weighted loss based on SSIM, PSNR, and CLIP Score differences.
    weights: Tuple of weights for SSIM, PSNR, and CLIP Score (default: 0.4, 0.3, 0.3).
    """
    return (weights[0] * ssim_diff + weights[1] * psnr_diff + weights[2] * clip_diff).mean()

# 3. Fine-tuning Function
def fine_tune_blip_with_metrics(blip_model, blip_processor, csv_file, dataset_path, device, epochs_mlp=50, epochs_blip=5, batch_size=8, lr=1e-5):  # Added learning rate
    dataset = MetricsDataset(csv_file, blip_processor, dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer (only fine-tune the MLP head initially)
    optimizer = optim.Adam(blip_model.mlp_head.parameters(), lr=lr)
    # Cosine Annealing LR Scheduler
    # LambdaLR Scheduler (decrease LR after epoch 20)
    def lr_lambda(epoch):
        if epoch < 20:
            return 1.0  # Keep LR the same before epoch 20
        else:
            return 0.1  # Reduce LR by a factor of 10 after epoch 20

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    criterion = nn.MSELoss() # Use Mean Squared Error Loss

    for epoch in range(epochs_mlp):
        blip_model.train()
        epoch_loss = 0.0
        for pixel_values, metrics_true in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs_mlp}"):
            pixel_values = pixel_values.to(device)
            metrics_true = metrics_true.to(device)

            metrics_pred = blip_model(pixel_values) # Forward pass

            loss = criterion(metrics_pred, metrics_true) # Calculate Loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs_mlp}, Loss: {epoch_loss / len(dataloader)}")

    print("Regressor training completed... Proceeding to finetuning Blip...")

    for epoch in range(epochs_blip):
        blip_model.train()
        epoch_loss = 0.0

        # # Now freeze the MLP head
        # for param in blip_model.mlp_head.parameters():
        #     param.requires_grad = False

        # Train the last num_layers_to_train layers of the text decoder + text_projection
        for param in blip_model.blip_model.text_decoder.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(blip_model.blip_model.text_decoder.parameters(), lr=lr/100)

        for pixel_values, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs_blip}"):
            pixel_values = pixel_values.to(device)
            metrics_pred = blip_model(pixel_values)

            ssim_pred = metrics_pred[:, 0]
            psnr_pred = metrics_pred[:, 1]
            clip_pred = metrics_pred[:, 2]

            ssim_diff = 1 - ssim_pred
            psnr_diff = 60 - psnr_pred
            clip_diff = -clip_pred

            loss = weighted_loss(ssim_diff=ssim_diff, psnr_diff=psnr_diff, clip_diff=clip_diff)

            # Unfreeze BLIP's original layers for the last epoch
            # if epoch == epochs_blip -1:
            #     for param in blip_model.blip_model.parameters():
            #         param.requires_grad = True

            #     optimizer = optim.Adam(blip_model.parameters(), lr=lr*10) # Reduce learning rate when unfreezing

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs_blip}, Loss: {epoch_loss / len(dataloader)}")

    # Save the fine-tuned model
    torch.save(blip_model.state_dict(), "fine_tuned_blip_with_metrics.pth") # Save just the state dictionary
    blip_model.blip_model.save_pretrained("blip-v2")
    blip_processor.save_pretrained("bip-v2") # Save processor

# Example Usage (in your main function):
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BLIP_MODEL_DIR = "blip"

    DATASET_PATH = "test2014"
    # DATASET_PATH = "pascal-voc2012"
    # DATASET_PATH = "genome"

    blip_model = AutoModelForImageTextToText.from_pretrained(BLIP_MODEL_DIR).to(DEVICE)
    blip_processor = AutoProcessor.from_pretrained(BLIP_MODEL_DIR)
    blip_model_with_head = BLIPWithMetricsHead(blip_model).to(DEVICE) # Wrap BLIP with MLP
    # blip_state_dict = torch.load("fine_tuned_blip_with_metrics.pth", map_location=DEVICE)
    # blip_model_with_head.load_state_dict(blip_state_dict, strict=False)

    csv_file = f"inpainting_metrics_{DATASET_PATH}.csv"  # Path to your CSV file
    fine_tune_blip_with_metrics(blip_model_with_head, blip_processor, csv_file, DATASET_PATH, DEVICE, epochs_mlp=40, epochs_blip=3, batch_size=8, lr = 1e-4)