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
  data = allargs[1]
  paramFile = allargs[2]
  outFileName = allargs[3]

   
  ### Doesn't work, std logic error 
  # First test - can we import code from a user-directory 
  # test() prints to stdout 
  if 0: 
    import sys
    sys.path.append(repo + "homogenizationmwe/")
    import homoglight
    homoglight.test()
  
  ### Doesn't work, std logic error 
  # Second - can we run a local code, stick the data somewhere? 
  # Runner() performs a fenics calculation that creates output files/temporary files 
  if 0: 
    import demo_poisson
    demo_poisson.Runner(outFileName=outFileName)                 

  # WORKS test on meddling with files
  dazip =   outFileName + ".zip"
  if 0: 
    datouch=   outFileName + ".txt"
    import os
    # manipulate file in a scratch directory 
    os.system('echo \"sadf\" > /tmp/x') 
    # create a zip file that is stored in 'protected' directory 
    os.system('zip %s /tmp/x '%dazip)           
    os.system('echo \"sadf\" > %s'%datouch)

  # 
  f = open(outFileName, 'w')
  f.write("hello world " + data+ " " + paramFile + " " + outFileName)
  f.write("\n"+dazip) 
  f.close()



def doitOld(allargs): 
  pdb = allargs[1]
  param= allargs[2]
  out = allargs[3]
  
  print pdb, param, out 

 
  f = open(out, 'w')
  f.write("hello world " + pdb + " " + param)
  f.close()

 


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
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  #fileIn= sys.argv[1]
  #if(len(sys.argv)==3):
  #  1
  #  #print "arg"

  # Loops over each argument in the command line 
  #for i,arg in enumerate(sys.argv):
  #  # calls 'doit' with the next argument following the argument '-validation'
  #  if(arg=="-validation"):
  #    arg1=sys.argv[i+1] 
  #    doit(arg1)
  




  #doitOld(sys.argv)
  doit(sys.argv)
  #raise RuntimeError("Arguments not understood")




