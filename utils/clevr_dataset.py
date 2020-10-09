# Copyright 2019-present, Kevin Yandoka Denamganai
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import cv2
import json
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

def ToLongTensor(data):
  arr = np.asarray(data, dtype=np.int32)
  tensor = torch.LongTensor(arr)
  return tensor

class Rescale(object) :
  def __init__(self, output_size) :
    assert( isinstance(output_size, (int, tuple) ) )
    self.output_size = output_size

  def __call__(self, sample) :
    image = sample
    h,w = image.shape[:2]
    new_h, new_w = self.output_size
    img = cv2.resize(image, (new_h, new_w) )
    return img

class ToTensor(object) :
  def __call__(self, sample) :
    image = sample
    #swap color axis :
    # numpy : H x W x C
    # torch : C x H x W
    image = image.transpose( (2,0,1) )
    sample =  torch.from_numpy(image)
    return sample

default_image_size = 84
default_transform = transforms.Compose([Rescale( (default_image_size,default_image_size) ),
                                       ToTensor()
                                       ])

class CLEVR_Dataset(Dataset):
  def __init__(self,  dataset_path,
                      data_path,
                      vocab_path, 
                      image_path,
                      transform=default_transform
                      ):
    self.dataset_path = dataset_path
    self.data_path = data_path
    self.vocab_path = vocab_path
    self.image_path = image_path
    self.transform = transform

    self.data = h5py.File(self.data_path, 'r')

    with open(self.vocab_path, 'r') as f:
      self.vocab = json.load(f)
    
    self.image_path_data = np.load(self.image_path)
    
    self.idx2question = ToLongTensor(self.data['questions'])
    self.idx2answer = ToLongTensor(self.data['answers'])
    self.idx2imageidx = ToLongTensor(self.data['image_idxs'])
    self.idx2question_family = np.asarray(self.data['question_families'])
    self.image_idx2path = self.image_path_data['paths']
  
  def getVocabSize(self):
    return len(self.vocab['question_vocab'])

  def getAnswerVocabSize(self):
    return len(self.vocab['answer_vocab'])

  def __getitem__(self, index):
    question = self.idx2question[index]
    image_idx = self.idx2imageidx[index]
    answer = self.idx2answer[index]
    path = os.path.join(self.dataset_path, self.image_idx2path[index])

    question_family = self.idx2question_family[index]

    image = cv2.imread(path)
    image = np.asarray(image, dtype=np.float32)
    if self.transform is not None :
      image= self.transform(image)

    return (image, question, answer, question_family)

  def __len__(self):
    return len(self.image_idx2path)


class CLEVR_DataLoader(DataLoader):
  def __init__(self, **kwargs):
    if 'dataset_path' not in kwargs:
      raise ValueError('Must give dataset_path')
    if 'data_path' not in kwargs:
      raise ValueError('Must give data_path')
    if 'vocab_path' not in kwargs:
      raise ValueError('Must give vocab_path')
    if 'image_path' not in kwargs:
      raise ValueError('Must give vocab_path')

    dataset_path = kwargs.pop("dataset_path")
    data_path = kwargs.pop("data_path")
    vocab_path = kwargs.pop("vocab_path")
    image_path = kwargs.pop("image_path")
    self.dataset = CLEVR_Dataset( dataset_path=dataset_path, data_path=data_path, vocab_path=vocab_path, image_path=image_path )
    
    kwargs['collate_fn'] = clevr_collate
    super(CLEVR_DataLoader, self).__init__(self.dataset, **kwargs)

  def getVocabSize(self):
    return self.dataset.getVocabSize()

  def getAnswerVocabSize(self):
    return self.dataset.getAnswerVocabSize()


def clevr_collate(batch):
  transposed = list(zip(*batch))
  image_batch = default_collate(transposed[0])
  question_batch = default_collate(transposed[1])
  question_lengths = default_collate( [ torch.tensor([len(q)], dtype=torch.int64) for q in transposed[1] ] )
  question_lengths = question_lengths.squeeze(1)
  answer_batch = default_collate(transposed[2])
  question_family_batch = default_collate(transposed[3])
  return [image_batch, question_batch, question_lengths, answer_batch, question_family_batch]