import glob
import numpy as np
import tensorflow as tf
import os, re
import shutil
from PIL import Image

from tqdm import tqdm
import time


class GenerateFakeTFRecord:
  def __init__(self):
    pass

  def wrtrec(self, imgpath, fname, res):
    #write to tfrecord
    #ximg = 2592; yimg = 1944
    if res == 64:
      self.ximg = 64
      self.yimg = 48   #48 for img.thumbnail(size)
    elif res == 128:
      self.ximg = 128  #2592
      self.yimg = 96   #1944
    elif res == 256:
      self.ximg = 256
      self.yimg = 192
    resize = self.ximg, self.yimg

    fpath = imgpath + "/" + fname
    img = Image.open(fpath)
    #img.thumbnail(size)
    #img = img.resize(resize)
    npimg = np.array(img).astype('uint8')
    image = np.squeeze(npimg).flatten().tolist()
    #encfpath = [ord(char) for char in fpath]
    #encfpath.extend([32] * (50-len(encfpath)))
    #encfname = tf.compat.as_bytes(fname, encoding='utf-8')
    example = tf.train.Example(features=tf.train.Features(feature={
      #'fpath': tf.train.Feature(int64_list=tf.train.Int64List(value=encfpath)),
      'fpath': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fpath.encode()])),
      'image': tf.train.Feature(int64_list=tf.train.Int64List(value=image)),
      'label': tf.train.Feature(float_list=tf.train.FloatList(value=self.coords))
    }))
    self.writer.write(example.SerializeToString())

  def create_tfrecord(self, tfrecord_file_name, dir_path, res):
    name = dir_path + "/0_fakeattrs.txt"
    with open(name, 'r') as file:
      lines = file.readlines()
    self.numnames = len(lines)
    pbar = tqdm(total=self.numnames);

    with tf.io.TFRecordWriter(tfrecord_file_name) as self.writer:
      with open(name, 'r') as file:
        lines = file.readlines()

      cimg = 0
      while cimg < len(lines):
        line = lines[cimg]
        items = line.split(",")
        fname = items[0]
        self.coords = [float(item) for item in items[1:]]
        self.wrtrec(dir_path, fname, res)
        cimg += 1
        # pbar
        if cimg % 1 == 0:
          pbar.update(1)

    pbar.close()
    print(cimg)


class GenerateTFRecord:
  def __init__(self):
    self.root = "G:\Data\hdr_data_new_labels"
    self.label = "\exp7500"
    #self.ob = "\ob1"
    self.ob = "\*"
    ob_train = [1, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29]
    ob_train += [31, 32, 34, 35, 37, 38, 40, 41, 43, 44, 46, 48, 50, 51, 53, 54, 56, 57, 59, 60, 62, 63, 65, 66, 71, 72]
    ob_test = [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 49, 52, 55, 58, 61, 64, 67, 73]
    print(len(ob_train), len(ob_test), len(ob_train+ob_test))
    self.ob_dir = {'train': ob_train, 'test': ob_test, 'all': ob_train + ob_test}
    ob_train_exc = ["\\exp7500\\ob37\\camera_imgs\\0038_0.jpg", "\\exp7500\\ob71\\camera_imgs\\0018_1.jpg"] #1237, 1897
    ob_test_exc = ["\\exp7500\\ob39\\camera_imgs\\0033_0.jpg", "\\exp7500\\ob39\\camera_imgs\\0037_0.jpg"]
    self.ob_exc = ob_test_exc + ob_train_exc
    #self.data = self.root + self.label + self.ob + "\processed_data.yaml"
    #print(self.data)
    self.txtfileexists = False

  def wrttxtdata(self, fpath, coords):
    # nefunguje pro multiple calls t
    openstr = 'w'
    if self.txtfileexists:
      openstr = 'a+'
    with open("datacoords.txt", openstr) as file:
      line = ""
      #line = fpath + ", "
      for coord in coords[:-1]:
        line += f"{coord:.5f}" + ", "
      line += f"{coords[-1]:.5f}" + "\n"
      file.write(line)
    self.txtfileexists = True

  def rexmatch(self, dir):
    rex = re.compile('^ob([1-9]|[1-9][0-9])$')
    if rex.match(dir):
      # print(dir, 'True')
      return True
    else:
      # print(dir, 'False')
      return False

  def wrtrec(self, imgpath, fpath, res):
    #write to tfrecord
    #ximg = 2592; yimg = 1944
    if res == 64:
      self.ximg = 64
      self.yimg = 48   #48 for img.thumbnail(size)
    elif res == 128:
      self.ximg = 128  #2592
      self.yimg = 96   #1944
    elif res == 256:
      self.ximg = 256
      self.yimg = 192
    resize = self.ximg, self.yimg

    img = Image.open(imgpath)
    #img.thumbnail(size)
    img = img.resize(resize)
    npimg = np.array(img).astype('uint8')
    image = np.squeeze(npimg).flatten().tolist()
    #encfpath = [ord(char) for char in fpath]
    #encfpath.extend([32] * (50-len(encfpath)))
    #encfname = tf.compat.as_bytes(fname, encoding='utf-8')
    example = tf.train.Example(features=tf.train.Features(feature={
      #'fpath': tf.train.Feature(int64_list=tf.train.Int64List(value=encfpath)),
      'fpath': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fpath.encode()])),
      'image': tf.train.Feature(int64_list=tf.train.Int64List(value=image)),
      'label': tf.train.Feature(float_list=tf.train.FloatList(value=self.coords))
    }))
    self.writer.write(example.SerializeToString())
    #self.wrttxtdata(fpath, self.coords)

  def create_tfrecord(self, tfrecord_file_name, res, dstype):
    # Get all file names of images present in folder

    #names = glob.glob(self.data)
    #listdir = os.listdir(path=self.root + self.label)
    #listdir.sort()  # alphanumeric
    #listdir = sorted(listdir, key=len)  # length
    #listdir = [dir for dir in listdir if self.rexmatch(dir)]  # only ob dirs
    #if type == 'train':
    #  listdir = listdir[:38]
    #elif type == 'test':
    #  listdir = listdir[38:]

    # print(listdir)

    #self.numnames = len(names)
    listdir = self.ob_dir[dstype] #train/test switch
    self.numnames = len(listdir)
    pbar = tqdm(total=self.numnames);
    c = 0
    nf = 0
    Cimg = 0

    #print(tfrecord_file_name)
    with tf.io.TFRecordWriter(tfrecord_file_name) as self.writer:

      #for name in names:
      for ob in listdir:  # train/test switch
        nf += 1
        ob = 'ob' + str(ob)
        name = self.root + self.label + "\\" + ob + "\processed_data.yaml"
        #print(name)
        with open(name, 'r') as file:
          lines = file.readlines()
        #print(len(lines))
        #print(lines[0])

        i = 0; cimg = 0
        while i < len(lines):
          line = lines[i]

          if line.strip() == "- robot_to_object_tfm_descriptor:":
            self.coords = []
            for j in range(6):
              i = i + 1
              coord = lines[i].strip()[2:]
              self.coords.append(float(coord))
              # print(coord)
            i = i + 1
            fname = lines[i][12:].strip().replace("/", "\\")
            fpath = self.label + fname
            imgpath = self.root + self.label + fname
            # print(imgpath)
            #print(fpath)
            if fpath in self.ob_exc:
              print("EXC--->", fpath)
            else:
              cimg += 1
              self.wrtrec(imgpath, fpath, res)
          elif line[2:9] == "- fname":
            fname = line[12:].strip().replace("/", "\\")
            fpath = self.label + fname
            imgpath = self.root + self.label + fname
            # print(imgpath)
            i = i + 1  # robot_to_object_tfm_descriptor:
            self.coords = []
            for j in range(6):
              i = i + 1
              coord = lines[i].strip()[2:]
              self.coords.append(float(coord))
              # print(coord)
            #print(fpath)
            if fpath in self.ob_exc:
              print("EXC--->", fpath)
            else:
              cimg += 1
              self.wrtrec(imgpath, fpath, res)
            #print(fname)
          else:
            # read next line
            i = i + 1

        Cimg += cimg
        #if Cimg > 0.7 * 2962:
        #  break

        #pbar
        time.sleep(0.1)
        c = c + 1
        if c % 1 == 0:
          pbar.update(1)

    pbar.close()
    print(Cimg, nf)


class TFRecordExtractor:
  def __init__(self, tfrecord_file, res):
    self.tfrecord_file = os.path.abspath(tfrecord_file)
    if res == 64:
      self.ximg = 64
      self.yimg = 48   #48 for img.thumbnail(size)
    elif res == 128:
      self.ximg = 128  #2592
      self.yimg = 96   #1944
    elif res == 256:
      self.ximg = 256
      self.yimg = 192

  def _extract_fn(self, tfrecord):
    # Extract features
    features = {
      # 'fpath': tf.io.FixedLenFeature([50], tf.int64),
      'fpath': tf.io.FixedLenFeature([1], tf.string),
      'image': tf.io.FixedLenFeature([self.ximg*self.yimg], tf.int64),
      'label': tf.io.FixedLenFeature([6], tf.float32)
    }

    # Extract the data record
    sample = tf.io.parse_single_example(tfrecord, features)
    fpath = sample['fpath']
    image = sample['image']
    label = sample['label']

    return [fpath, image, label]

  def extract_image(self):
    # Create folder to store extracted images
    folder_path = './ExtractedImages'
    shutil.rmtree(folder_path, ignore_errors=True)
    os.mkdir(folder_path)

    # Pipeline of dataset and iterator
    dataset = tf.data.TFRecordDataset([self.tfrecord_file])
    dataset = dataset.map(self._extract_fn)
    #iterator = dataset.make_one_shot_iterator()
    #next_element = iterator.get_next()

    # TF 2.0
    with open("coords.txt", 'w') as file:
      i = 3
      for next_element in dataset:
        #fpath = next_element[0]
        #fpath = ''.join(chr(i) for i in fpath).strip()
        fpath = next_element[0].numpy()[0].decode()
        image = next_element[1].numpy()
        coords = list(next_element[2].numpy())
        image = image.reshape(self.yimg, self.ximg)
        image = image.astype('uint8')
        save_path = os.path.abspath(os.path.join(folder_path, f"img_{i}.jpg"))
        image = Image.fromarray(image)
        image.save(save_path)
        print("Save path = ", save_path)
        #print("Image path = ", fpath)
        file.write(fpath + "\n")
        file.write(save_path+"\n")
        for coord in coords:
          line = str(coord)+"\n"
          file.write(line)
        file.write("---"+"\n")
        i = i + 1
        if i>=3+5:
          break

    return


if __name__ == '__main__':
    # resolution
    xres = 64
    yres = int(0.75*64)

    # Generate/Extract Fake
    #epoch = 34
    #dir_path = f"./dcganc_images/fake_train/fake_epoch_{epoch * 1000}"
    #t = GenerateFakeTFRecord()
    #t.create_tfrecord(f"ganfake_{xres}{yres}{epoch}.tfrecord", dir_path=dir_path, res=xres)
    #t = TFRecordExtractor(f"fakegan_{xres}{yres}{epoch}.tfrecord", res=xres)
    #t.extract_image()

    # Create
    t = GenerateTFRecord()
    t.create_tfrecord(f"gan_{xres}{yres}train.tfrecord", res=xres, dstype='train')
    t.create_tfrecord(f"gan_{xres}{yres}test.tfrecord" , res=xres, dstype='test')
    #t.create_tfrecord(f"gan_{xres}{yres}all.tfrecord"  , res=xres, dstype='all')

    # Extract
    #---
    dstype = 'train'
    t = TFRecordExtractor(f"gan_{xres}{yres}{dstype}.tfrecord", res=xres)
    t.extract_image()