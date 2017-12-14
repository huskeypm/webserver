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
def RunTemplate(allargs):
  dataFileName=allargs[1]  
  outFileName="test.txt"   
  import numpy 
  # open cv stuff 
  
  f = open(outFileName, 'w')
  f.write("hello world " + dataFileName)
  f.write("\n")
  f.close()

