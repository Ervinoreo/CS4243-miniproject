import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json

DATA_DIR = '../../labeled_data_train/'
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 1
BATCH_SIZE = 128
LATENT_DIM = 100
RESUME_TRAINING = True
START_EPOCH = 300
TOTAL_EPOCHS = 310
SAVE_EVERY = 10

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode='int',
    color_mode='grayscale',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
NUM_CLASSES = len(class_names)
char_to_int = {name: i for i, name in enumerate(class_names)}
int_to_char = {i: name for i, name in enumerate(class_names)}

print(f"Found {NUM_CLASSES} classes: {class_names}")

def normalize(image, label):
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, label

train_dataset = train_dataset.map(normalize).cache().shuffle(10000).prefetch(buffer_size=tf.data.AUTOTUNE)
print("Data pipeline built.")

def build_generator():
    label_input = Input(shape=(1,), dtype='int32', name='label_input')
    label_embedding = layers.Embedding(NUM_CLASSES, 50)(label_input)
    label_embedding = layers.Dense(8 * 8)(label_embedding)
    label_embedding = layers.Reshape((8, 8, 1))(label_embedding)

    noise_input = Input(shape=(LATENT_DIM,), name='noise_input')
    noise_path = layers.Dense(8 * 8 * 256, use_bias=False)(noise_input)
    noise_path = layers.BatchNormalization()(noise_path)
    noise_path = layers.LeakyReLU()(noise_path)
    noise_path = layers.Reshape((8, 8, 256))(noise_path)

    merged = layers.Concatenate()([noise_path, label_embedding])

    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(merged)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(CHANNELS, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
    assert x.shape == (None, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    return Model([noise_input, label_input], x, name="Generator")

def build_discriminator():
    label_input = Input(shape=(1,), dtype='int32', name='label_input')
    label_embedding = layers.Embedding(NUM_CLASSES, 50)(label_input)
    label_embedding = layers.Dense(IMG_HEIGHT * IMG_WIDTH)(label_embedding)
    label_embedding = layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 1))(label_embedding)

    image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), name='image_input')
    merged = layers.Concatenate()([image_input, label_embedding])

    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(merged)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    return Model([image_input, label_input], x, name="Discriminator")

generator = build_generator()
discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(real_images, real_labels):
    current_batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([current_batch_size, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator([noise, real_labels], training=True)
        real_output = discriminator([real_images, real_labels], training=True)
        fake_output = discriminator([fake_images, real_labels], training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    return gen_loss, disc_loss

# --- 8. Setup Checkpoints & Sample Generation ---
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)

if RESUME_TRAINING and tf.train.latest_checkpoint(checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print(f"✅ Restored from checkpoint: {tf.train.latest_checkpoint(checkpoint_dir)}")
else:
    print("🆕 Starting fresh training")

NUM_EXAMPLES_TO_GENERATE = 16
seed_labels = tf.constant(np.random.randint(0, NUM_CLASSES, size=NUM_EXAMPLES_TO_GENERATE))
seed_noise = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, LATENT_DIM])

def save_sample_images(model, epoch, noise_seed, label_seed):
    predictions = model([noise_seed, label_seed], training=False)
    predictions = (predictions + 1) / 2.0
    plt.figure(figsize=(10, 10))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        label_int = label_seed[i].numpy()
        plt.title(f"Label: {int_to_char[label_int]}")
        plt.axis('off')
    if not os.path.exists('image_samples'):
        os.makedirs('image_samples')
    plt.savefig(f'image_samples/image_at_epoch_{epoch:04d}.png')
    # Close the plot to save memory
    plt.close()

# --- 9. The Main Loop ---
start_epoch = START_EPOCH if RESUME_TRAINING else 0
print(f"Starting training from epoch {start_epoch + 1} to {TOTAL_EPOCHS}...")

for epoch in range(start_epoch, TOTAL_EPOCHS):
    for image_batch, label_batch in train_dataset:
        g_loss, d_loss = train_step(image_batch, label_batch)

    print(f"Epoch {epoch + 1}/{TOTAL_EPOCHS} --- Gen Loss: {g_loss.numpy():.4f} | Disc Loss: {d_loss.numpy():.4f}")

    if (epoch + 1) % SAVE_EVERY == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
        save_sample_images(generator, epoch + 1, seed_noise, seed_labels)

print("Training finished.")

# --- 10. Save Final Model and Class Mappings ---
generator.save('captcha_char_generator.h5')

# THIS IS THE CRITICAL FIX:
class_data = {'class_names': class_names}
with open('class_mappings.json', 'w') as f:
    json.dump(class_data, f)

print("Model and class mappings saved.")