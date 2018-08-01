'''
Function file for running the matched filter myocyte classification algorithm 
'''

import sys

import matplotlib.pylab as plt
import numpy as np
import yaml

import bankDetect as bD
import display_util as du
import matchedFilter as mf
import optimizer
import painter
import preprocessing as pp
import util


def DisplayHits(img,threshed,
                smooth=8 # px
                ):
        # smooth out image to make it easier to visualize hits 
        daround=np.ones([smooth,smooth])
        sadf=mf.matchedFilter(threshed,daround,parsevals=False,demean=False)

        # merge two fields 
        du.StackGrayRedAlpha(img,sadf,alpha=0.5)

class empty:pass    
def docalc(img,
           mf,
           lobemf=None,
           #corrThresh=0.,
           #s=1.,
           paramDict = optimizer.ParamDict(),
           debug=False,
           smooth = 8, # smoothing for final display
           iters = [-20,-10,0,10,20], # needs to be put into param dict
           fileName="corr.png"):



    ## Store info 
    inputs=empty()
    inputs.imgOrig = img
    inputs.mfOrig  = mf
    inputs.lobemf = lobemf

    print "WARNING: TOO RESTRICTIVE ANGLES" 



    results = bD.DetectFilter(inputs,paramDict,iters=iters,display=debug)

    pasteFilter = True
    if pasteFilter:

      MFy,MFx = util.measureFilterDimensions(mf)
      filterChannel = 0
      imgDim = np.shape(img)
      results.threshed = painter.doLabel(results,dx=MFx,dy=MFy,thresh=paramDict['snrThresh'])
      #coloredImageHolder[:,:,filterChannel] = filterChannelHolder
    
    print "Writing file %s"%fileName
    #plt.figure()
    DisplayHits(img,results.threshed,smooth=smooth)
    plt.gcf().savefig(fileName,dpi=300)


    return inputs,results 

###
###  Updated YAML routine to lightly preprocess image
###
def updatedSimple(imgName,mfName,filterTwoSarcomereSize,threh,debug=False,smooth=4,outName="hits.png"):
  '''
  Updated routine for the athena web server that utilizes WT punishment filter.

  INPUTS:
    - imgName: Name of image that has myocyte of interest in the middle of the image
               Myocyte major axis must be roughly parallel to x axis. There should be some conserved 
               striations in the middle of the image. 
    - thresh: Threshold that is utilized in the detection
  
  OUTPUTS:
    - None, writes image
  '''
  ### Load necessary inputs
  img = util.ReadImg(imgName)
  imgDims = np.shape(img)
  #mf = util.LoadFilter("./myoimages/newSimpleWTFilter")
  mf = util.LoadFilter(mfName)
  mfPunishment = util.LoadFilter("./myoimages/newSimpleWTPunishmentFilter.png")

  ### Lightly preprocess the image
  ## Resize the image
  cY, cX = int(round(float(imgDims[0]/2.))), int(round(float(imgDims[1]/2.)))
  xExtent = 50
  yExtent = 25
  top = cY-yExtent; bottom = cY+yExtent; left = cX-xExtent; right = cX+xExtent
  indexes = np.asarray([top,bottom,left,right])
  subsection = np.asarray(img[top:bottom,left:right],dtype=np.float64)
  subsection /= np.max(subsection)
  img, scale, newIndexes = pp.resizeGivenSubsection(img,subsection,filterTwoSarcomereSize,indexes)
  ## Apply CLAHE to image
  img = pp.applyCLAHE(img,filterTwoSarcomereSize)
  ## Intelligently threshold image using gaussian thresholding
  img = pp.normalizeToStriations(img, newIndexes, filterTwoSarcomereSize)
  img = np.asarray(img,dtype=np.float64)
  img /= np.max(img)

  ### Construct parameter dictionary
  paramDict = optimizer.ParamDict(typeDict="WT")
  paramDict['mfPunishment'] = mfPunishment
  paramDict['covarianceMatrix'] = np.ones_like(img)

  docalc(img,
         mf,
         paramDict=paramDict,
         debug=debug,
         iters=[-25,-20,-15,-10,-5,0,5,10,15,20,25],
         smooth=smooth,
         fileName = outName
         )

###
### updated yaml call
###
def updatedSimpleYaml(ymlName):
  print "Adapt to accept filter mode argument?"
  with open(ymlName) as fp:
    data = yaml.load(fp)
  print "Reading %s" % ymlName

  if 'outName' in data:
    outName = data['outName']
  else:
    outName = 'hits.png'
  
  updatedSimple(data['imgName'],
                data['mfName'],
                data['filterTwoSarcomereSize'],
                data['thresh'],
                debug=data['debug'],
                outName=outName
                )



#
# MAIN routine executed when launching this script from command line 
#
if __name__ == "__main__":
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    # validation
    if(arg=="-validation"):             
      validation() 
      quit()

    # general test
    if(arg=="-simple"):             
      imgName =sys.argv[i+1] 
      mfName =sys.argv[i+2] 
      thresh = float(sys.argv[i+3])
      simple(imgName,mfName,thresh)
      quit()

    # updated version of simple yaml
    if(arg=="-updatedSimpleYaml"):
      ymlName = sys.argv[i+1]
      updatedSimpleYaml(ymlName)
      quit()


  raise RuntimeError("Arguments not understood")
