###
### Group of functions that will walk the user fully through the preprocessing
### routines.
###
import os
import sys

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA

import matchedFilter as mF
import util

###############################################################################
###
### Normalization Routines
###
##############################################################################

def normalizeToStriations(img, subsectionIdxs,filterSize):
  '''
  function that will go through the subsection and find average smoothed peak 
  and valley intensity of each striation and will normalize the image 
  based on those values.
  '''

  print "Normalizing myocyte to striations"
  
  ### Load in filter that will be used to smooth the subsection
  WTfilterName = "./myoimages/singleTTFilter.png"
  WTfilter = util.ReadImg(WTfilterName,renorm=True)
  # divide by the sum so that we are averaging across the filter
  WTfilter /= np.sum(WTfilter)

  ### Perform smoothing on subsection
  smoothed = np.asarray(mF.matchedFilter(img,WTfilter,demean=False),dtype=np.uint8)

  ### Grab subsection of the smoothed image
  smoothedSubsection = smoothed.copy()[subsectionIdxs[0]:subsectionIdxs[1],
                                       subsectionIdxs[2]:subsectionIdxs[3]]
  #plt.figure()
  #plt.imshow(smoothedSubsection)
  #plt.colorbar()
  #plt.show()
  
  ### Perform Gaussian thresholding to pull out striations
  # blockSize is pixel neighborhood that each pixel is compared to
  blockSize = int(round(float(filterSize) / 3.57)) # value is empirical
  # blockSize must be odd so we have to check this
  if blockSize % 2 == 0:
    blockSize += 1
  # constant is a constant that is subtracted from each distribution for each pixel
  constant = 0
  # threshValue is the value at which super threshold pixels are marked, else px = 0
  threshValue = 1
  gaussSubsection = cv2.adaptiveThreshold(smoothedSubsection, threshValue,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, blockSize,
                                          constant)
  #plt.figure()
  #plt.imshow(gaussSubsection)
  #plt.colorbar()
  #plt.show()

  ### Calculate the peak and valley values from the segmented image
  peaks = smoothedSubsection[np.nonzero(gaussSubsection)]
  peakValue = np.mean(peaks)
  peakSTD = np.std(peaks)
  valleys = smoothedSubsection[np.where(gaussSubsection == 0)]
  valleyValue = np.mean(valleys)
  valleySTD = np.std(valleys)

  print "Average Striation Value:", peakValue
  print "Standard Deviation of Striation:", peakSTD
  print "Average Striation Gap Value:", valleyValue
  print "Stand Deviation of Striation Gap", valleySTD

  ### Calculate ceiling and floor thresholds empirically
  ceiling = peakValue + 3 * peakSTD
  floor = valleyValue - valleySTD
  if ceiling > 255:
    ceiling = 255.
  if floor < 0:
    floor = 0
  
  ceiling = int(round(ceiling))
  floor = int(round(floor))
  print "Ceiling Pixel Value:", ceiling
  print "Floor Pixel Value:", floor

  ### Threshold
  #img = img.astype(np.float64)
  #img /= np.max(img)  
  img[img>=ceiling] = ceiling
  img[img<=floor] = floor
  img -= floor
  img = img.astype(np.float64)
  img /= np.max(img)
  img *= 255
  img = img.astype(np.uint8)

  return img

###############################################################################
###
### Resizing Routines
###
###############################################################################
def resizeGivenSubsection(img,subsection,filterTwoSarcomereSize,indexes):
  '''
  Function to resize img given a subsection of the image
  '''
  ### Using this subsection, calculate the periodogram
  fBig, psd_Big = signal.periodogram(subsection)
  # finding sum, will be easier to identify striation length with singular dimensionality
  bigSum = np.sum(psd_Big,axis=0)

  ### Mask out the noise in the subsection periodogram
  # NOTE: These are imposed assumptions on the resizing routine
  maxStriationSize = 50.
  minStriationSize = 5.
  minPeriodogramValue = 1. / maxStriationSize
  maxPeriodogramValue = 1. / minStriationSize
  bigSum[fBig < minPeriodogramValue] = 0.
  bigSum[fBig > maxPeriodogramValue] = 0.

  display = False
  if display:
    plt.figure()
    plt.plot(fBig,bigSum)
    plt.title("Collapsed Periodogram of Subsection")
    plt.show()

  ### Find peak value of periodogram and calculate striation size
  striationSize = 1. / fBig[np.argmax(bigSum)]
  imgTwoSarcomereSize = int(round(2 * striationSize))
  print "Two Sarcomere size:", imgTwoSarcomereSize,"Pixels per Two Sarcomeres"

  if imgTwoSarcomereSize > 70 or imgTwoSarcomereSize < 10:
    print "WARNING: Image likely failed to be properly resized. Manual resizing",\
           "may be necessary!!!!!"

  ### Using peak value, resize the image
  scale = float(filterTwoSarcomereSize) / float(imgTwoSarcomereSize)
  resized = cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)

  ### Find new indexes in image
  newIndexes = indexes * scale
  newIndexes = np.round(newIndexes).astype(np.int32)

  return resized, scale, newIndexes



###############################################################################
###
### CLAHE Routines
###
###############################################################################

def applyCLAHE(img,filterTwoSarcomereSize):
  print "Applying CLAHE to Myocyte"

  kernel = (filterTwoSarcomereSize, filterTwoSarcomereSize)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=kernel)

  clahedImage = clahe.apply(img)

  return clahedImage

def processMask(fileName,degreesOffCenter,resizeScale):
  '''
  function to reorient and resize the mask that was generated for the original
  image.
  '''
  maskName = fileName[:-4]+"_mask"+fileName[-4:]
  mask = util.ReadImg(maskName)
  reoriented = imutils.rotate_bound(mask,degreesOffCenter)
  resized = cv2.resize(reoriented,None,fx=resizeScale,fy=resizeScale,interpolation=cv2.INTER_CUBIC)
  cv2.imwrite(fileName[:-4]+"_processed_mask"+fileName[-4:],resized)