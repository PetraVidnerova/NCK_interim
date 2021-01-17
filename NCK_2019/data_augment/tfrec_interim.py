import numpy as np
import tensorflow as tf
import os
import shutil
import matplotlib.image as mpimg

from tqdm import tqdm
import time


class GenerateTFRecord:
    def __init__(self, X, C, Y):
        self.X = X
        self.C = C
        self.Y = Y
        self.numimg = X.shape[0]

    def convert_images(self, tfrecord_file_name):
        print(f"Converting {self.X.shape[0]} images ...")
        pbar = tqdm(total=self.numimg)
        with tf.compat.v1.python_io.TFRecordWriter(tfrecord_file_name) as writer:
          for c in range(self.numimg):
            image = X[c,...].squeeze()
            image = image.reshape(234*366,).tolist() #85644
            centroid = C[c,...].tolist()
            result = [Y[c]]
            example = self._convert_image(image, centroid, result)
            writer.write(example.SerializeToString())
            time.sleep(0.1)
            if c%10 == 0:
              pbar.update(10)
        pbar.close()


    def _convert_image(self, image, centroid, result):
        example = tf.train.Example(features = tf.train.Features(feature = {
            'image': tf.train.Feature(float_list=tf.train.FloatList(value=image)),
            'centroid': tf.train.Feature(float_list=tf.train.FloatList(value=centroid)),
            'result': tf.train.Feature(int64_list=tf.train.Int64List(value=result))
        }))
        return example


class TFRecordExtractor:
  def __init__(self, tfrecord_file):
    self.tfrecord_file = os.path.abspath(tfrecord_file)

  def _extract_fn(self, tfrecord):
    # Extract features
    features = {
      'image': tf.io.FixedLenFeature([85644], tf.float32),
      'centroid': tf.io.FixedLenFeature([3], tf.float32),
      'result': tf.io.FixedLenFeature([], tf.int64)
    }

    # Extract the data record
    sample = tf.io.parse_single_example(tfrecord, features)
    image = sample['image']
    centroid = sample['centroid']
    result = sample['result']
    return [image, centroid, result]

  def extract_image(self):
    XX_list = []
    CC_list = []
    YY_list = []

    dataset = tf.data.TFRecordDataset([self.tfrecord_file])
    dataset = dataset.map(self._extract_fn)
    # TF 1.x
    #iterator = dataset.make_one_shot_iterator()
    #try:
    #  # Keep extracting data till TFRecord is exhausted
    #  while True:
    #    example = iterator.get_next()
    #    XX_list.append(example[0].numpy())
    #    CC_list.append(example[1].numpy())
    #    YY_list.append(example[2].numpy())
    #except:
    #  pass

    # TF 2.0
    for next_element in dataset:
      XX_list.append(next_element[0].numpy())
      CC_list.append(next_element[1].numpy())
      YY_list.append(next_element[2].numpy())

    return XX_list, CC_list, YY_list

if __name__ == '__main__':
  # Load data
  from data import load_data
  X, C, Y = load_data()
  # Create
  tfrec_exists = os.path.exists('lego.tfrecord')
  if not(tfrec_exists):
    t = GenerateTFRecord(X, C, Y)
    t.convert_images('lego.tfrecord')
  # Extract
  t = TFRecordExtractor('lego.tfrecord')
  XX, CC, YY = t.extract_image()
  #---
  XX = [xx.reshape((234, 366)) for xx in XX]
  XX = np.stack(XX)
  XX = XX[..., np.newaxis]
  X_diff = np.linalg.norm(X-XX)
  assert(X_diff==0.0)
  #---
  CC = np.stack(CC)
  C_diff = np.linalg.norm(C-CC)
  assert (C_diff == 0.0)
  #---
  YY = np.stack(YY)
  Y_diff = np.linalg.norm(Y-YY)
  assert (Y_diff == 0.0)
  print("Enc/Dec test OK.")