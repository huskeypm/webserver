#!/usr/bin/env python
import sys
import os
import numpy as np
##################################
#
# Revisions
#       10.08.10 inception
#
##################################

#
# ROUTINE  
#
repo = "/opt/pkh/"
thisFileRoot = '/'.join(os.path.realpath(__file__).split('/')[:-1])


def doit(allargs): 
  # ignoring args. 
  pdbFile= allargs[1]
  paramFile = allargs[2]
  outFileName = allargs[3]
  RunTest(pdbFile,paramFile,outFileName)

def RunTest(pdbFile=None,paramFile=None,outFileName=None):

  # RunTest if I can import fenics
  #if 1: 
  #  print "Attempting to load dolfin libs"
  #  from dolfin import *
  

  # for debugging 
  if pdbFile==None:
    return 

   
  # WORKS RunTest on meddling with files
  dazip =   outFileName + ".zip"

  # write a zip file 
  datouch=   outFileName + ".txt"
  import os
  # manipulate file in a scratch directory 
  os.system('echo \"sadf\" > /tmp/x') 
  # create a zip file that is stored in 'protected' directory 
  os.system('zip %s /tmp/x '%dazip)           
  os.system('echo \"sadf\" > %s'%datouch)

  # 
  f = open(outFileName, 'w')
  f.write("hello world " + pdbFile+ " " + paramFile + " " + outFileName)
  f.write("\n"+dazip) 
  f.close()

def RunODE(outFileName):
  # scipy test 
  import demo_ode
  demo_ode.RunODE()
  print "Will write to", outFileName

def RunHomogMWE():
  ### Doesn't work, std logic error 
  # First RunTest - can we import code from a user-directory 
  # RunTest() prints to stdout 
  sys.path.append(repo + "homogenizationmwe/")
  import homoglight
  homoglight.RunTest()

def RunPoissonDemo(outFileName):
  ### Doesn't work, std logic error 
  # Second - can we run a local code, stick the data somewhere? 
  # Runner() performs a fenics calculation that creates output files/temporary files 
  import demo_poisson
  demo_poisson.Runner(outFileName=outFileName)                 

def RunMatchedMyoYaml(yamlFile,outFileName,imgFileName,maskFileName):
  '''
  Function to run the MatchedMyo algorithm using the yaml functionality
  '''
  ### Use non-interactive backend for matplotlib
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  ### append matchedmyo directory to path so we can import necessary runners
  sys.path.append('/opt/webserver/matchedmyo')

  ### import matched filtering machinery
  #import detect
  import matchedmyo as mm
  import yaml
  import util

  ### Form the inputs for the classification
  inputs = mm.Inputs(
    imageName = imgFileName,
    maskName = maskFileName,
    preprocess = False
  )
  inputs.setupImages()
  if yamlFile == None: # use default yaml file if not specified
    yamlFile = '/opt/webserver/matchedmyo/YAML_files/webserver_default.yml'
  inputs.yamlFileName = yamlFile
  inputs.load_yaml()
  ## Check if classification type is specified as normal or arbitrary
  try:
    self.classificationType = self.yamlDict['classificationType']
  except:
    pass
  ## Specify webserver specific parameters
  if 'outputParams' in inputs.yamlDict.keys():
    ## opting to save the outputs manually after the job is run instead of using previous routines
    inputs.yamlDict['outputParams']['fileRoot'] = None
    inputs.yamlDict['outputParams']['fileType'] = None
    inputs.yamlDict['outputParams']['saveHitsArray'] = False
    inputs.yamlDict['outputParams']['csvFile'] = None
  initialPreprocessValue = inputs.yamlDict['preprocess']
  inputs.yamlDict['preprocess'] = False
  inputs.setupDefaultParamDicts()
  inputs.updateDefaultDict()
  inputs.updateParamDicts()

  if initialPreprocessValue:
    inputs.imgOrig = util.lightlyPreprocess(
      inputs.imgOrig,
      inputs.dic['filterTwoSarcomereSize'],
    )
    # 0.8 value here is to kill the brightness a bit
    eightBitImage = inputs.imgOrig.astype(np.float32) /np.max(inputs.imgOrig) * 255. * 0.8
    eightBitImage = eightBitImage.astype(np.uint8)
    inputs.colorImage = np.dstack((eightBitImage,eightBitImage,eightBitImage))

  if inputs.classificationType == 'myocyte':
    myResults = mm.giveMarkedMyocyte(inputs = inputs)
  elif inputs.classificationType == 'arbitrary':
    myResults = mm.arbitraryFiltering(inputs = inputs)
  else:
    raise RuntimeError("Classification Type (specified as classificationType: <type> in YAML file)"
                       +" not understood. Check to see that spelling is correct.")

  ### Create figure for outputs and save
  f, ax = plt.subplots(2,1)
  ax[0].imshow(util.switchBRChannels(myResults.markedImage))
  ax[0].set_title('Classified Image')
  ax[1].imshow(util.switchBRChannels(myResults.markedAngles))
  ax[1].set_title('Angle Analysis of Image')
  plt.gcf().savefig(outFileName,dpi=inputs.dic['outputParams']['dpi'])

#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -run " % (scriptName)
  msg+="""
  
 
Notes:

"""
  return msg

#
# MAIN routine executed when launching this script from command line 
#
if __name__ == "__main__":
  import sys
  import os
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  #if(len(sys.argv)==3):
  #  1
  #  #print "arg"

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
  #  # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-test"):
      RunTest(pdbFile="/opt/pkh/a.pdb",
           paramFile="/opt/pkh/b.txt",
           outFileName="/tmp/dummy"
      )
      sys.exit()
    if(arg=="-template"): 
      import demo_template
      demo_template.RunTemplate(sys.argv)
      quit()
    if(arg=="-poisson"):
      RunPoissonDemo(
           outFileName="/tmp/dummy"
      )
      sys.exit()
    if(arg=="-ode"):
      RunODE(outFileName=sys.argv[i+1]) 
      sys.exit()

    if(arg=="-matchedMyoYaml"):
      imgFileName = sys.argv[i+1]
      outFileName = sys.argv[i+2]
      try:
        # check if there is a second argument
        secondArg = str(sys.argv[i+3])
        if secondArg[-4:] == '.yml':
          yamlFile = secondArg
          maskFileName = None
        else:
          maskFileName = secondArg
      except:
        secondArg = None
      if isinstance(secondArg, str):
        try:
          maskFileName = sys.argv[i+4]
        except:
          maskFileName = None
      else:
        yamlFile = None
        maskFileName = None
      RunMatchedMyoYaml(yamlFile,outFileName,imgFileName,maskFileName)
      sys.exit()

  raise RuntimeError("Arguments not understood")




