#!/usr/bin/env python
import sys
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
    if(arg=="-poisson"):
      RunPoissonDemo(
           outFileName="/tmp/dummy"
      )
      sys.exit()
    if(arg=="-ode"):
      RunODE(outFileName=sys.argv[i+1]) 
      sys.exit()
  




  #doitOld(sys.argv)
  doit(sys.argv)
  #raise RuntimeError("Arguments not understood")




