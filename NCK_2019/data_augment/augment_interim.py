import tensorflow as tf
import numpy as np

from data import load_data
X, C, y = load_data()

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# No augmentation
datagen_orig = ImageDataGenerator()
it_orig = datagen_orig.flow(X, batch_size=1, shuffle=False)

# Augmented
datagen_augm = ImageDataGenerator(height_shift_range = 0.1,
                                  width_shift_range = 0.1,
                                  horizontal_flip = True,
                                  vertical_flip = True,
                                  rotation_range = 15,
                                  zoom_range = 0.1,
                                  shear_range = 0.02,
                                 )
it_augm = datagen_augm.flow(X, batch_size=1, shuffle=False)

# Visualize augmentations
import matplotlib.pyplot as plt
n_x = 4
n_y = 3
canvas = np.empty((234 * n_x, 366 * n_y))
for i in range(n_x):
  for j in range(n_y):
    if i % 2 == 0:
      batch = it_orig.next()
    else:
      batch = it_augm.next()
    canvas[i * 234:(i + 1) * 234, j * 366:(j + 1) * 366] = batch.reshape([234, 366])


plt.figure(figsize=(n_x, n_y))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()
exit()