# 🔧 Fine-Tuning BLIP for Image Quality Metrics Regression (SSIM, PSNR, CLIP)

This project fine-tunes a [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) model to **generate better inpainting captions**.

The preprint of the paper can be reviewed in [this link](https://github.com/MT522/enhance_inpainting/blob/main/A%20Transformer-Based%20Generalization%20Pipeline%20for%20Inpainting%20Models.pdf).

## 📦 Requirements

Install dependencies (preferably in a virtual environment):

```bash
pip install torch torchvision transformers pandas tqdm pillow
Optional (for GPU support):
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 🧠 Model Overview
Base: BLIP, StableDiffusion-Inpaint

Head: MLP regressor that outputs 3 values: [SSIM, PSNR, CLIP Score]

Loss: Weighted MSE / custom weighted difference loss


## 🚀 Running the Script
1. Place your data
Place images in a directory like test2014/

Download a pretrained BLIP model (e.g., from Hugging Face) into a blip/ folder

2. Run main
```bash
python main.py
```

The script will generate a csv file of all the losses. This will be used to train the MLP head and finetune BLIP.

3. Run training
```bash
python finetune_blip.py
```

The script will:

Train the MLP head for epochs_mlp epochs

Fine-tune select layers of BLIP for epochs_blip epochs

Save the final model to:

`blip-v2/fine_tuned_blip_with_metrics.pth`

4. Run main with updated BLIP model
