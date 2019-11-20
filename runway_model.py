
import torch
import torchvision
import base64
import cupy
import cv2
import flask
import getopt
import gevent
import gevent.pywsgi
import h5py
import io
import math
import moviepy
import moviepy.editor
import numpy
import os
import random
import re
import scipy
import scipy.io
import shutil
import sys
import tempfile
import time
import urllib
import zipfile
import runway

@runway.setup(options={})
def setup(opts):
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    exec(open('./3d-ken-burns/common.py', 'r').read())
    exec(open('./3d-ken-burns/models/disparity-estimation.py', 'r').read())
    exec(open('./3d-ken-burns/models/disparity-adjustment.py', 'r').read())
    exec(open('./3d-ken-burns/models/disparity-refinement.py', 'r').read())
    exec(open('./3d-ken-burns/models/pointcloud-inpainting.py', 'r').read())
    return process_load

processingInput = {'image': runway.image(description="photograph")}
processingOutput = {'video': runway.array(
    item_type = runway.file, 
    description = "3d effect video"
)}

@runway.command('process', inputs = processingInput, outputs = processingOutput)
def process(model, inputs):
    numpyImage = numpy.image(inputs['image'])
    intWidth = numpyImage.shape[1]
    intHeight = numpyImage.shape[0]
    dblRatio = float(intWidth) / float(intHeight)
    intWidth = min(int(1024 * dblRatio), 1024)
    intHeight = min(int(1024 / dblRatio), 1024)
    numpyImage = cv2.resize(src=numpyImage, dsize=(intWidth, intHeight), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
    model(numpyImage, {})
    objectFrom = {
      'dblCenterU': intWidth / 2.0,
      'dblCenterV': intHeight / 2.0,
      'intCropWidth': int(math.floor(0.97 * intWidth)),
      'intCropHeight': int(math.floor(0.97 * intHeight))
    }
    objectTo = process_autozoom({
      'dblShift': 100.0,
      'dblZoom': 1.25,
      'objectFrom': objectFrom
    })
    numpyResult = process_kenburns({
      'dblSteps': numpy.linspace(0.0, 1.0, 75).tolist(),
      'objectFrom': objectFrom,
      'objectTo': objectTo,
      'boolInpaint': True
    })
    return {'video': numpyResult}

if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=9000)