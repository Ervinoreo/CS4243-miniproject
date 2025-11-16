# CAPTCHA Generation Models

This directory contains several models for generating CAPTCHA images. Each model uses a different approach, from single-character generation to full-image synthesis.

## 1. Single-Character Conditional GAN (TensorFlow)

This is the foundational model that generates individual characters based on a given label. The final CAPTCHA is then assembled by combining these generated characters.

- **Training:** `train.py`
- **Generation:** `generate.py`

### How it Works

1.  **`train.py`**: Trains a Conditional GAN (CGAN) on a dataset of single characters. The generator learns to produce a 64x64 grayscale image of a character given a one-hot encoded label. The trained generator is saved as `captcha_char_generator.h5`.
2.  **`generate.py`**: Loads the trained generator and, for a given string, generates each character individually. It then composites these characters into a single CAPTCHA image with added noise and random transformations.

### Usage

1.  **Train the model:**
    ```bash
    python train.py
    ```
2.  **Generate a CAPTCHA:**
    ```bash
    python generate.py
    ```

## 2. Full-Image Conditional GAN (`gen2`)

This model generates the entire CAPTCHA image in a single pass, conditioned on the full text label.

- **Training & Generation:** `gen2/gen.py`

### How it Works

The `gen2/gen.py` script defines a Conditional GAN (CGAN) in PyTorch. The generator takes a latent vector and a one-hot encoded representation of the entire CAPTCHA text to produce a 64x200 RGB image.

### Usage

1.  **Train the model:**
    ```bash
    python gen2/gen.py
    ```
2.  The script will automatically save generated samples and model checkpoints in the `gen2/generated_samples` and `gen2/models` directories, respectively.

## 3. Full-Image WGAN-GP (`gen3`)

This model is an improved version of the full-image CGAN, using a Wasserstein GAN with Gradient Penalty (WGAN-GP) for more stable training and higher-quality results.

- **Training & Generation:** `gen3/gen.py`

### How it Works

The `gen3/gen.py` script implements a WGAN-GP in PyTorch. It uses a "critic" instead of a discriminator and a different loss function to guide the generator. The overall architecture is similar to the `gen2` model but with architectural improvements for stability.

### Usage

1.  **Train the model:**
    ```bash
    python gen3/gen.py
    ```
2.  Generated samples and model checkpoints will be saved in the `gen3/generated_samples_1` and `gen3/models` directories.

## 4. Single-Character WGAN-GP (`gen4`)

This model combines the single-character generation approach with the advanced WGAN-GP architecture. It generates high-quality individual characters, which are then composed into a full CAPTCHA.

- **Training:** `gen4/train.py`
- **Generation:** `gen4/generate.py`

### How it Works

1.  **`gen4/train.py`**: Trains a WGAN-GP on a dataset of single characters. The generator learns to produce a 64x64 RGB image of a character given a one-hot encoded label.
2.  **`gen4/generate.py`**: Loads the trained single-character generator and composes a full CAPTCHA by generating each character and concatenating them horizontally.

### Usage

1.  **Train the model:**
    ```bash
    python gen4/train.py
    ```
2.  **Generate a CAPTCHA:**
    ```bash
    python gen4/generate.py --model_path <path_to_model.pth> --text "yourtext"
    ```