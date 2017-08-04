import numpy as np
import scipy as sp
from scipy.integrate import odeint
#import matplotlib.pyplot as plt


def g(y, x):
    y0 = y[0]
    y1 = y[1]
    y2 = ((3*x+2)*y1 + (6*x-8)*y0)/(3*x-1)
    return y1, y2

def RunODE():
  # Initial conditions on y, y' at x=0
  init = 2.0, 3.0
  # First integrate from 0 to 2
  x = np.linspace(0,2,100)
  sol=odeint(g, init, x)

  print "I ran!" 

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
  for i,arg in enumerate(sys.argv):
    # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-test"):
      #arg1=sys.argv[i+1]
      RunODE()     
      sys.exit()






  raise RuntimeError("Arguments not understood")


