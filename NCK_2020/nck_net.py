import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime

res = 64
ximg = 64
yimg = 48

GLOBAL_BATCH_SIZE = 64
TEST_DATABASE_LENGTH = 988
EPOCHS = 300
DEPTH = 8

def create_dataset(res, ximg, yimg):

  def _extract_fn(tfrecord):
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

    image = tf.reshape(image, [ximg, yimg, 1])
    image = tf.cast(image, 'float32')

    coords = tf.cast(label, 'float32')

    return fpath, image, coords

  dstype = 'train'
  tfrecord_file = f"gan_6448train.tfrecord"
  dataset = tf.data.TFRecordDataset([tfrecord_file])
  dataset = dataset.map(_extract_fn)
  train_dataset = dataset.shuffle(buffer_size=3000, reshuffle_each_iteration=False)

  dstype = 'faketrain'
  tfrecord_file = f"ganfake_644834.tfrecord"
  dataset = tf.data.TFRecordDataset([tfrecord_file])
  dataset = dataset.map(_extract_fn)
  faketrain_dataset = dataset.shuffle(buffer_size=3000, reshuffle_each_iteration=False)

  dstype = 'test'
  tfrecord_file = f"gan_6448test.tfrecord"
  dataset = tf.data.TFRecordDataset([tfrecord_file])
  test_dataset = dataset.map(_extract_fn)
  #test_dataset = dataset.shuffle(buffer_size=1000)

  return train_dataset, test_dataset, faketrain_dataset

def create_model(depth=1):
  model = tf.keras.Sequential()
  model.add(layers.InputLayer(input_shape=(64, 48, 1)))
  for i in range(depth):
    model.add(layers.Conv2D(64, 3, activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dense(6))

  return model

def compute_loss(y,yhat):
  x = y-yhat
  #norms = tf.norm(x, axis=1, ord=1)
  norms = tf.norm(x, axis=1, ord=2)
  #norms = tf.norm(x, axis=1, ord=np.inf)
  loss = tf.math.reduce_mean(norms)
  return loss

def check_validity_tf(err):
  err_rx, err_ry, err_rz, err_x, err_y, err_z = tf.unstack(err, axis=1)
  cond1 = abs(err_rx) + abs(err_ry) < 0.052
  cond2 = abs(err_x) + abs(err_y) < 0.007
  cond3_ur = abs(err_z) < 0.04
  cond3_ku = abs(err_z) + 0.075*(abs(err_rx) + abs(err_ry)) < 0.008
  valid_ur = tf.math.reduce_all(tf.stack([cond1, cond2, cond3_ur], axis=1), axis=1)
  valid_ku = tf.math.reduce_all(tf.stack([cond1, cond2, cond3_ku], axis=1), axis=1)
  #ret = tf.stack([valid_ur, cond1, cond2, cond3_ur], axis=1)
  ret = tf.stack([valid_ur, valid_ku], axis=1)
  ret = tf.cast(ret, tf.int8)
  return ret

def check_validity(err):
  err_rx, err_ry, err_rz, err_x, err_y, err_z = err.numpy()
  cond1 = abs(err_rx) + abs(err_ry) < 0.052
  cond2 = abs(err_x) + abs(err_y) < 0.007
  cond3_ur = abs(err_z) < 0.04
  cond3_ku = abs(err_z) + 0.075*(abs(err_rx) + abs(err_ry)) < 0.008
  valid_ur = cond1 and cond2 and cond3_ur
  valid_ku = cond1 and cond2 and cond3_ku
  #return int(valid_ur), int(valid_ku)
  retnp = int(valid_ur), int(valid_ku), int(cond1), int(cond2), int(cond3_ur), int(cond3_ku)
  #retnp = int(cond1)
  ret = tf.convert_to_tensor(retnp, dtype=tf.float32)
  return ret


train_dataset, test_dataset, faketrain_dataset = create_dataset(res, ximg, yimg)
train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE)
test_dataset = test_dataset.batch(GLOBAL_BATCH_SIZE)
#test_dataset = test_dataset.batch(TEST_DATABASE_LENGTH)
faketrain_dataset = faketrain_dataset.batch(GLOBAL_BATCH_SIZE)

model = create_model(DEPTH)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5)

#tensorboard --logdir logs
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
test_log_dir = 'logs/' + current_time
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


@tf.function
def train_step(inputs):
  fpath, images, y = inputs

  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = compute_loss(y, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return loss

for epoch in range(EPOCHS):
  print(f"===EPOCH {epoch}===")
  # TRAIN LOOP
  total_loss = 0.0
  nb = 0
  #for inputs in faketrain_dataset:
  for inputs in train_dataset:
    loss = train_step(inputs)
    nb = nb + 1
    #print(f"Loss {nb} ... ", loss)
    print(".", end="", flush=True)

  # TEST LOOP
  #test_dataset = test_dataset.shuffle(buffer_size=1000)
  nt = 0
  print()
  print(f"===TESTING===")

  b = 0; loss = np.zeros((TEST_DATABASE_LENGTH, 1))
  for inputs in test_dataset:
    _, images, y = inputs
    predictions = model(images)
    nb = predictions.shape[0]
    GBS = GLOBAL_BATCH_SIZE
    loss[b*GBS:b*GBS+nb] = compute_loss(y, predictions).numpy()
    b += 1
  avg_loss = tf.reduce_mean(loss).numpy()
  print(f"loss of predictions ... , {avg_loss:.8f}")
  with test_summary_writer.as_default():
    tf.summary.scalar('loss', avg_loss, step=epoch)


# final testing
print(f"===OVERALL TESTING===")
_, test_dataset, _ = create_dataset(res, ximg, yimg)
test_dataset = test_dataset.batch(TEST_DATABASE_LENGTH)
errfname = f"test_results{EPOCHS}.txt"
nf = 0
with open(errfname, "w") as file:
  head = "filename, err_rx, err_ry, err_rz, err_x, err_y, err_z, valid_ur, valid_kuka\n"
  file.writelines(head)
  b = 0
  for inputs in test_dataset:
    b += 1
    if b == 1:
      print(b, end="", flush=True)
    else:
      print(",", b, end="", flush=True)
    fpath, images, y = inputs
    predictions = model(images)
    loss = compute_loss(y, predictions)
    print(f", overall loss ... {loss:.3f}")
    #---
    err = predictions - y
    is_valid = check_validity_tf(err)
    batch_size = fpath.shape[0]
    for i in range(batch_size):
      fname = fpath[i][0].numpy().decode()
      line = fname + ", "
      perr = list(err[i].numpy())
      for pn in perr:
        pns = f"{pn:<8.6f}"
        if pn >= 0:
          pns = " " + pns
        line += pns.rjust(12, ' ') + ", "
      valid_ur, valid_ku = is_valid[i].numpy()
      line += f" {valid_ur}, {valid_ku}\n"
      pom = fname[-6:]
      if fname[-6:] == "_2.jpg":
        file.writelines(line)
        nf += 1

print(f"depth ... {DEPTH}")
