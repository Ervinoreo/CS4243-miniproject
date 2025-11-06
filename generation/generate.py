import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt

print("Loading model...")
MODEL_PATH = 'captcha_char_generator.h5'
MAPPINGS_PATH = 'class_mappings.json'

if not os.path.exists(MODEL_PATH) or not os.path.exists(MAPPINGS_PATH):
    print(f"Error: Model or class mappings not found.")
    print(f"Please run 'train.py' first to create '{MODEL_PATH}' and '{MAPPINGS_PATH}'.")
    exit()

generator = tf.keras.models.load_model(MODEL_PATH)

print("Loading class mappings...")
with open(MAPPINGS_PATH, 'r') as f:
    class_data = json.load(f)
class_names = class_data['class_names']

char_to_int = {name: i for i, name in enumerate(class_names)}
int_to_char = {i: name for i, name in enumerate(class_names)}
print(f"Loaded {len(class_names)} classes.")

LATENT_DIM = 100
IMG_HEIGHT = 64
IMG_WIDTH = 64


def get_random_color():
    return (random.randint(0, 150), random.randint(0, 150), random.randint(0, 150))

def generate_captcha(text_label, num_noise_lines=5):
    char_images_pil = []

    # generate BnW character images from GAN
    for char in text_label:
        if char not in char_to_int:
            print(f"Warning: Character '{char}' not in training data. Skipping.")
            continue

        label_int = char_to_int[char]
        label_input = tf.constant([[label_int]])
        noise_input = tf.random.normal([1, LATENT_DIM])

        # shape (1, 64, 64, 1)
        char_image = generator([noise_input, label_input], training=False)

        # denormalize [-1, 1] to [0, 255]
        # convert to PIL Image
        char_image_np = (char_image[0].numpy() * 127.5 + 127.5).astype(np.uint8)
        # Squeeze the channel dimension to make it a 2D (64, 64) array
        char_image_pil = Image.fromarray(np.squeeze(char_image_np), mode='L')
        char_images_pil.append(char_image_pil)


    total_width = 0
    positions = []
    # random overlap
    for i in range(len(char_images_pil)):
        positions.append(total_width)
        total_width += (IMG_WIDTH - random.randint(10, 25))

    # WHITE bg
    full_image = Image.new('RGB', (total_width, IMG_HEIGHT), color=(255, 255, 255))

    for bw_mask, x_pos in zip(char_images_pil, positions):
        char_color = get_random_color()

        char_canvas = Image.new('RGBA', (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0, 0))

        char_canvas.paste(char_color, (0, 0), mask=bw_mask)

        # random rotation
        y_pos = random.randint(-5, 5)
        rotated_char = char_canvas.rotate(random.randint(-15, 15),
                                          resample=Image.BICUBIC,
                                          expand=False)

        full_image.paste(rotated_char, (x_pos, y_pos), rotated_char)

    # noise lines
    draw = ImageDraw.Draw(full_image)
    for _ in range(num_noise_lines):
        start = (random.randint(0, total_width), random.randint(0, IMG_HEIGHT))
        end = (random.randint(0, total_width), random.randint(0, IMG_HEIGHT))
        draw.line([start, end], fill=(0, 0, 0), width=random.randint(1, 2)) #line color

    return full_image


new_captcha_label = "8dfv0"  # WHAT STRINGGGG
my_captcha = generate_captcha(new_captcha_label, num_noise_lines=6)

if not os.path.exists('generated_captchas'):
    os.makedirs('generated_captchas')

save_path = f"generated_captchas/{new_captcha_label}.png"
my_captcha.save(save_path)

print(f"Generated CAPTCHA saved as '{save_path}'")
plt.imshow(my_captcha)
plt.axis('off')
plt.show()