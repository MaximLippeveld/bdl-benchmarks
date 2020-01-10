# Copyright 2019 BDL Benchmarks Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import io
import csv

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
cv2 = tfds.core.lazy_imports.cv2


class WhiteBloodCellClassificationConfig(tfds.core.BuilderConfig):
  """BuilderConfig for DiabeticRetinopathyDiagnosis."""

  def __init__(self,
               target_height,
               target_width,
               channels=[]
               crop=False,
               scale=1,
               num_classes=8
               **kwargs):
    """BuilderConfig for DiabeticRetinopathyDiagnosis.
    
    Args:
      target_height: `int`, number of pixels in height.
      target_width: `int`, number of pixels in width.
      scale: (optional) `int`, the radius of the neighborhood to apply
        Gaussian blur filtering.
      **kwargs: keyword arguments forward to super.
    """
    super(WhiteBloodCellClassificationConfig, self).__init__(**kwargs)
    self._target_height = target_height
    self._target_width = target_width
    self._channels = channels
    self._scale = scale
    self._num_classes = num_classes

  @property
  def target_height(self):
    return self._target_height

  @property
  def target_width(self):
    return self._target_width

  @property
  def scale(self):
    return self._scale
  
  @property
  def channels(self):
    return self._channels
  
  @property
  def num_classes(self):
    return self._num_classes


class WhiteBloodCellClassification(tfds.core.DatasetBuilder):

  BUILDER_CONFIGS = [
      WhiteBloodCellClassificationConfig(
          name="realworld_stainfree",
          version="0.0.1",
          description="BF, BF2, and SSC images for RealWorld level.",
          target_height=90,
          target_width=90,
          channels=["BF", "BF2",  "SSC"]
      ),
      WhiteBloodCellClassificationConfig(
          name="realworld_onlybf",
          version="0.0.1",
          description="BF, and BF2 images for RealWorld level.",
          target_height=90,
          target_width=90,
          channels=["BF", "BF2"]
      )
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description="An imaging flow cytometry dataset of white blood cells. "
        features=tfds.features.FeaturesDict({
            "name":
            tfds.features.Text(),  # patient ID + eye. eg: "4_left".
            "image":
            tfds.features.Image(shape=(self.builder_config.target_height,
                                       self.builder_config.target_width, len(self.builder_config.channels))),
            "label":
            tfds.features.ClassLabel(num_classes=self.builder_config.num_classes),
        })
    )

  @classmethod
  def _preprocess(cls,
                  image_fobj,
                  target_height,
                  target_width,
                  crop=False,
                  scale=500):
    """Resize an image to have (roughly) the given number of target pixels.

    Args:
      image_fobj: File object containing the original image.
      target_height: `int`, number of pixels in height.
      target_width: `int`, number of pixels in width.
      crops: (optional) `bool`, if True crops the centre of the original
        image t the target size.
      scale: (optional) `int`, the radius of the neighborhood to apply
        Gaussian blur filtering.

    Returns:
      A file object.
    """
    # Decode image using OpenCV2.
    image = cv2.imdecode(np.fromstring(image_fobj.read(), dtype=np.uint8),
                         flags=3)
    try:
      a = cls._get_radius(image, scale)
      b = np.zeros(a.shape)
      cv2.circle(img=b,
                 center=(a.shape[1] // 2, a.shape[0] // 2),
                 radius=int(scale * 0.9),
                 color=(1, 1, 1),
                 thickness=-1,
                 lineType=8,
                 shift=0)
      image = cv2.addWeighted(src1=a,
                              alpha=4,
                              src2=cv2.GaussianBlur(
                                  src=a, ksize=(0, 0), sigmaX=scale // 30),
                              beta=-4,
                              gamma=128) * b + 128 * (1 - b)
    except cv2.error:
      pass
    # Reshape image to target size
    image = cv2.resize(image, (target_height, target_width))
    # Encode the image with quality=72 and store it in a BytesIO object.
    _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    return io.BytesIO(buff.tostring())
