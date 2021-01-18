# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb
# https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Resizing
import time

#from IPython import display
from PIL import Image

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets.mnist import load_data
#conda install scikit-image
from skimage.transform import resize
from scipy.linalg import sqrtm
#from numba import njit

FIDmodel = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

# Load and prepare the dataset
#(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
#train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
#train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
# BEGAN = 16, DCGAN = 64
BATCH_SIZE = 32
FID_BATCH = 1000

# Batch and shuffle the data
#train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# scale an array of images to a new size
#@njit
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0).astype('float16')
		# store
		images_list.append(new_image)
	return np.asarray(images_list)

def calculate_fid(model, act1, images2):
	# calculate activations
	#act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def _extract_fn(tfrecord):
  ximg = 64
  yimg = 48
  # Extract features
  features = {
    'fpath': tf.io.FixedLenFeature([1], tf.string),
    'image': tf.io.FixedLenFeature([ximg * yimg], tf.int64),
    'label': tf.io.FixedLenFeature([6], tf.float32)
  }

  # Extract the data record
  sample = tf.io.parse_single_example(tfrecord, features)
  fpath = sample['fpath']
  image = sample['image']
  label = sample['label']

  fpath = tf.cast(fpath, tf.string)

  image = tf.reshape(image, [yimg, ximg, 1])
  #image = tf.reshape(image, [64, 64, 1])
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  #image = tf.cast(image, 'float32')

  coords = tf.cast(label, 'float32')

  #return fpath, image, coords
  attrs = coords
  return image, attrs

# Prepare CelebA data
#tfrecord_file = "G:\\VIR2020\\began_tf20\\Data\\celeba\\began64.tfrecord"
xres = 64; yres=48; dstype = 'train'
tfrecord_file = f"E:\\NCK\\gan_{xres}{yres}{dstype}.tfrecord"
dataset = tf.data.TFRecordDataset(tfrecord_file)
dataset = dataset.map(_extract_fn)
dataset = dataset.repeat()
#dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(4)
train_dataset = dataset

fiddataset = tf.data.TFRecordDataset(tfrecord_file)
fiddataset = fiddataset.map(_extract_fn)
fiddataset = fiddataset.repeat()
fiddataset = fiddataset.batch(FID_BATCH)

for item in fiddataset.take(1):
  images_real, _ = item
images_real = (images_real + 1.) * 127.5
images_real = tf.concat((images_real,)*3, axis=3)
images_real = scale_images(images_real.numpy(), (299,299,3))
#images_real = preprocess_input(images_real).astype('float16')
images_real = preprocess_input(images_real)
act_real = FIDmodel.predict(images_real)
del images_real
print("Done reals.")

class Resize2D(tf.keras.layers.Layer):
  def __init__(self, size_x, size_y):
    super(Resize2D, self).__init__()
    self.size_x = size_x
    self.size_y = size_y

  def call(self, x):
    x = tf.compat.v1.image.resize_nearest_neighbor(x, size=(int(self.size_x), int(self.size_y)))
    return x

# The 64x48 Generator
def make_generator_model48():
  gf_dim = 64
  model = tf.keras.Sequential()
  model.add(layers.Dense(gf_dim * 8 * 3 * 4, input_shape=(100,)))
  model.add(layers.Reshape((3, 4, gf_dim * 8)))  # (4, 3, 512)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape((3, 4, gf_dim * 8)))  # (4, 4, 512)
  assert model.output_shape == (None, 3, 4, 512)  # Note: None is the batch size

  #model.add(layers.Conv2DTranspose(gf_dim * 4, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  model.add(layers.Conv2DTranspose(gf_dim * 4, (5, 5), strides=(2, 2), padding='same'))
  assert model.output_shape == (None, 6, 8, 256)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  #model.add(layers.Conv2DTranspose(gf_dim * 2, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  model.add(layers.Conv2DTranspose(gf_dim * 2, (5, 5), strides=(2, 2), padding='same'))
  assert model.output_shape == (None, 12, 16, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  #model.add(layers.Conv2DTranspose(gf_dim * 1, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  model.add(layers.Conv2DTranspose(gf_dim * 1, (5, 5), strides=(2, 2), padding='same'))
  assert model.output_shape == (None, 24, 32, 64)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  #model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
  assert model.output_shape == (None, 48, 64, 1)

  return model

# The Generator
def make_generator_model():
  gf_dim = 64
  model = tf.keras.Sequential()
  model.add(layers.Dense(gf_dim * 8 * 4 * 4, input_shape=(100,)))
  model.add(layers.Reshape((4, 4, gf_dim * 8)))  # (4, 4, 512)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape((4, 4, gf_dim * 8)))  # (4, 4, 512)
  assert model.output_shape == (None, 4, 4, 512)  # Note: None is the batch size

  #model.add(layers.Conv2DTranspose(gf_dim * 4, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  model.add(layers.Conv2DTranspose(gf_dim * 4, (5, 5), strides=(2, 2), padding='same'))
  assert model.output_shape == (None, 8, 8, 256)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  #model.add(layers.Conv2DTranspose(gf_dim * 2, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  model.add(layers.Conv2DTranspose(gf_dim * 2, (5, 5), strides=(2, 2), padding='same'))
  assert model.output_shape == (None, 16, 16, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  #model.add(layers.Conv2DTranspose(gf_dim * 1, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  model.add(layers.Conv2DTranspose(gf_dim * 1, (5, 5), strides=(2, 2), padding='same'))
  assert model.output_shape == (None, 32, 32, 64)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  #model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
  assert model.output_shape == (None, 64, 64, 1)

  return model

generator = make_generator_model48()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
#plt.imshow(generated_image[0, :, :, 0], cmap='gray')
#print(generator.summary())

# The 64x48 Discriminator
def make_discriminator_model48():
  model = tf.keras.Sequential()
  model.add(layers.InputLayer(input_shape=(48, 64, 1)))
  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Flatten())
  model.add(layers.Dense(1))

  return model

# The Discriminator
def make_discriminator_model():
  model = tf.keras.Sequential()
  model.add(layers.InputLayer(input_shape=(64, 64, 1)))
  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Flatten())
  model.add(layers.Dense(1))

  return model

discriminator = make_discriminator_model48()
decision = discriminator(generated_image)
print(decision)
#print(discriminator.summary())


# Define the loss and optimizers
#cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#def discriminator_loss(real_output, fake_output):
#  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#  total_loss = real_loss + fake_loss
#  return total_loss

#def generator_loss(fake_output):
#  return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
#generator_optimizer = tf.keras.optimizers.Adam(1e-4)
#discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
#generator_optimizer = tf.keras.optimizers.Adam(5e-5, beta_1=0.5)
#discriminator_optimizer = tf.keras.optimizers.Adam(5e-5, beta_1=0.5)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

#generator_optimizer = tf.keras.optimizers.Adam(0.0002)
#discriminator_optimizer = tf.keras.optimizers.Adam(0.0001)

#checkpoint_dir = './training_checkpoints'
#checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
#checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                 discriminator_optimizer=discriminator_optimizer,
#                                 generator=generator,
#                                 discriminator=discriminator)


# Define the training loop
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
#print(generator.summary())
#print(discriminator.summary())

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
#seed = tf.random.normal([num_examples_to_generate, noise_dim])
seed = np.random.normal(size=[num_examples_to_generate, noise_dim]).astype(np.float32)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    #noise = np.random.uniform(-1, 1, size=[BATCH_SIZE, noise_dim]).astype(np.float32)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      d_real_logits = discriminator(images, training=True)
      d_fake_logits = discriminator(generated_images, training=True)

      gen_loss = tf.reduce_mean(tf.nn.l2_loss(d_fake_logits - tf.ones_like(d_fake_logits)))
      d_loss_real = tf.reduce_mean(tf.nn.l2_loss(d_real_logits - tf.ones_like(d_real_logits)))
      d_loss_fake = tf.reduce_mean(tf.nn.l2_loss(d_fake_logits - tf.zeros_like(d_real_logits)))
      disc_loss = d_loss_real + d_loss_fake

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs):
  #for epoch in range(epochs):
  epoch = 0
  while epoch < EPOCHS:
    epoch += 1
    epoch_start = time.time()

    for image_batch in dataset.take(1000):
      g_loss, d_loss = train_step(image_batch[0])

    print(g_loss.numpy(), d_loss.numpy())

    # Produce images for the GIF as we go
    #display.clear_output(wait=True)
    generate_and_save_images(generator, epoch, seed)

    # Compute FID for 10000 example
    if epoch % 5 == 0 or epoch == 1:
    #if False:
      print("Calculating FID ... ", end="", flush=True)
      fid_noise = np.random.normal(size=[FID_BATCH, noise_dim]).astype(np.float32)
      fid_fake=np.empty([FID_BATCH, 48, 64, 1])
      dn = 100
      for i in range(FID_BATCH//dn):
        fid_batch = fid_noise[i*dn:(i+1)*dn]
        #print(i, fid_batch.shape)
        g = generator(fid_batch, training=False)
        g = (g + 1.) * 127.5
        fid_fake[i*dn:(i+1)*dn] = g.numpy()
      fid_fake = np.concatenate((fid_fake,)*3, axis=3)
      fid_fake = np.rint(fid_fake).clip(0, 255).astype(np.uint8)
      fid_fake = fid_fake.astype('float32')
      fid_fake = scale_images(fid_fake, (299, 299, 3))
      images_fake = preprocess_input(fid_fake)

      fid_start = time.time()
      fid = calculate_fid(FIDmodel, act_real, images_fake)
      fid_end = time.time()
      msg = f"{fid:6.2f}, time {fid_end - fid_start:.2f} sec"
      print(msg)
      with open("lsgan_images/fid.txt", "a+") as fidfile:
        fidfile.writelines([f"{epoch:03}_FID {msg}\n"])

    # Save the model every 15 epochs
    #if (epoch + 1) % 15 == 0:
    #  checkpoint.save(file_prefix = checkpoint_prefix)

    print (f"Time for epoch {epoch} is {time.time()-epoch_start:.2f} sec")

  # Generate after the final epoch
  #display.clear_output(wait=True)
  generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
  ximg = 64
  yimg = 48
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  #fig = plt.figure(figsize=(4,4))

  #for i in range(predictions.shape[0]):
  #    plt.subplot(4, 4, i+1)
  #    plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
  #    plt.axis('off')

  g = predictions
  g = (g + 1.) * 127.5
  canvas = np.empty((yimg * 4, ximg * 4, 1))
  c = 0
  for i in range(4):
    for j in range(4):
      c = c + 1
      canvas[j * yimg:(j + 1) * yimg, i * ximg:(i + 1) * ximg] = g[c-1]

  image = np.rint(canvas).clip(0, 255).astype(np.uint8)
  image = np.squeeze(image)
  image = Image.fromarray(image)
  image.save(f"lsgan_images/fake_{epoch*1000}.jpg")

  #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()

def save_real_images(dataset):
  ximg = 64
  yimg = 48

  for images in dataset.take(1):
    g = (images[0][:16] + 1.) * 127.5

  canvas = np.empty((yimg * 4, ximg * 4, 1))
  c = 0
  for i in range(4):
    for j in range(4):
      c = c + 1
      canvas[j * yimg:(j + 1) * yimg, i * ximg:(i + 1) * ximg] = g[c-1]

  image = np.rint(canvas).clip(0, 255).astype(np.uint8)
  image = np.squeeze(image)
  image = Image.fromarray(image)
  image.save("lsgan_images/reals.jpg")

  #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()

#===
print("Start training ... ")

f = open("lsgan_images/fid.txt", "w"); f.close()
save_real_images(train_dataset)
train(train_dataset, EPOCHS)
#===

# Display a single image using the epoch number
#def display_image(epoch_no):
#  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

anim_file = 'lsgan_images/progress.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('lsgan_images/fake*.jpg')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

# conda install git
# pip install -q git+https://github.com/tensorflow/docs
# pip install --upgrade protobuf
#import tensorflow_docs.vis.embed as embed
#embed.embed_file(anim_file)