"""
Finds all hits for rotated filter 
To do: rename painter as something understandble 
"""

class empty:pass

import cv2
import painter 
import numpy as np 
import matplotlib.pylab as plt 
### Commented out twoDtense since we aren't running GPU level analysis on webserver
#import twoDtense as tdt


##
## For a single matched filter, this function iterates over passed-in angles 
## and reports highest correlation output for each iteration 
## 
def DetectFilter(
  inputs,  # basically contains test data and matched filter 
  paramDict,  # parameter dictionary  
  iters,   # rotations over which mf will be tested
  display=False,
  label=None,
  filterMode=None,
  returnAngles=False,
):

  if inputs is None:
    raise RuntimeError("PLACEHOLDER TO REMIND ONE TO USE INPUT/PARAMDICT OBJECTS")

  # store
  result = empty()
  result.stackedDict = dict()

  # do correlations across all iter
  if paramDict['useGPU']:
    result,timeElapsed = tdt.doTFloop(
            inputs,
            paramDict,
            ziters=iters
            )
    # since routine gives correlated > 0 for snr > snrThresh then all nonzero correlated pixels are hits
    print paramDict['inverseSNR']
    if paramDict['inverseSNR']:
      result.stackedHits[result.stackedHits > paramDict['snrThresh']] = 0.
    else:
      result.stackedHits[result.stackedHits < paramDict['snrThresh']] = 0.

  else:
    result.correlated = painter.correlateThresher(
       inputs,
       paramDict,
       iters=iters,
       printer=display,
       filterMode=filterMode,
       label=label,
       )

      # stack hits to form 'total field' of hits
    if returnAngles:
      result.stackedHits, result.stackedAngles = painter.StackHits(
                  result.correlated,paramDict,iters,display=display,returnAngles=returnAngles)
    else:
      result.stackedHits= painter.StackHits(result.correlated,paramDict,iters,
        display=display)



  return result

def GetHits(aboveThresholdPoints):
  ## idenfity hits (binary)  
  mylocs =  np.zeros_like(aboveThresholdPoints.flatten())
  hits = np.argwhere(aboveThresholdPoints.flatten()>0)
  mylocs[hits] = 1
  
  dims = np.shape(aboveThresholdPoints)  
  locs = mylocs.reshape(dims)  
  return locs

# color red channel 
def ColorChannel(Img,stackedHits,chIdx=0):  
    locs = GetHits(stackedHits)   
    chFloat =np.array(Img[:,:,chIdx],dtype=np.float)
    #chFloat[10:20,10:20] += 255 
    chFloat+= 255*locs  
    chFloat[np.where(chFloat>255)]=255
    Img[:,:,chIdx] = np.array(chFloat,dtype=np.uint8)  



# red - entries where hits are to be colored (same size as rawOrig)
# will label in rawOrig detects in the 'red' channel as red, etc 
def colorHits(rawOrig,red=None,green=None,outName=None,label="",plotMe=True):
  dims = np.shape(rawOrig)  
  
  # make RGB version of data   
  Img = np.zeros([dims[0],dims[1],3],dtype=np.uint8)
  scale = 0.5  
  Img[:,:,0] = scale * rawOrig
  Img[:,:,1] = scale * rawOrig
  Img[:,:,2] = scale * rawOrig
    

  
  if isinstance(red, (list, tuple, np.ndarray)): 
    ColorChannel(Img,red,chIdx=0)
  if isinstance(green, (list, tuple, np.ndarray)): 
    ColorChannel(Img,green,chIdx=1)    

  if plotMe:
    plt.figure()  
    plt.subplot(1,2,1)
    plt.title("Raw data (%s)"%label)
    plt.imshow(rawOrig,cmap='gray')
    plt.subplot(1,2,2)
    plt.title("Marked") 
    plt.imshow(Img)  

  if outName!=None:
    plt.tight_layout()
    plt.gcf().savefig(outName,dpi=300)
    plt.close()
  else:
    plt.close()
    return Img  

# main engine 
# TODO remove scale/pass into filter itself
# tests filters1 and filters2 against a test data set
def TestFilters(
    testData, # data against which filters 1 and 2 are applied
    filter1Data, # matched filter 1
    filter2Data, # matched filter 2
    filter1Thresh=60,filter2Thresh=50,
    iters = [0,10,20,30,40,50,60,70,80,90], 
            
    display=False,
    colorHitsOutName=None,
    label="test",
    filterDict=None, thresholdDict=None,
    saveColoredFig=True,
    returnAngles=False,
    single = False,
######
    paramDict=None   # PUT ALL PARAMETERS HERE COMMON TO ALL FILTERS
):       

    #raise RuntimeError("Require Dataset object, as done for tissue validation") 
    params=paramDict
    ## perform detection 
    inputs = empty()
    inputs.imgOrig = testData

      
    daColors =dict()
    ### filter 1 
    inputs.mfOrig = filter1Data
    params['snrThresh'] = filter1Thresh
    filter1Result = DetectFilter(inputs,params,iters,display=display,
                                 filterMode="filter1",label=label,
                                 returnAngles=returnAngles)
    #plt.figure()
    #plt.imshow(filter1Result.stackedHits)
    #plt.colorbar()
    #plt.show()
    #quit()
    daColors['green']= filter1Result.stackedHits
      
    ### filter 2 
    if single is False:
      inputs.mfOrig = filter2Data
      params['snrThresh'] = filter2Thresh
      filter2Result = DetectFilter(inputs,params,iters,display=display,
                                   filterMode="filter2",label=label,
                                   returnAngles=returnAngles)
      daColors['red']= filter2Result.stackedHits
    else:
      filter2Result = None
      daColors['red'] =None
      
    if returnAngles:
      # make the actual figure
      filter1Result.coloredAngles = painter.colorAngles(testData,
                                         filter1Result.stackedAngles,
                                         iters)

    ## display results 
    if colorHitsOutName!=None: 
      colorHits(testData,
              red=daColors['red'],
              green=daColors['green'],
              label=label,
              outName=colorHitsOutName)                       

    return filter1Result, filter2Result 

  
#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -validation" % (scriptName)
  msg+="""
  
 
Notes:

"""
  return msg



