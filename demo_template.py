"""
Example that uses cv2
"""
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
import numpy  as np  
repo = "/opt/pkh/"
def RunTemplate(allargs):
  pngFileName=allargs[2]  
  pngFileOutName= pngFileName.replace(".png","_out.png") 

  outFileName="test.txt"   
  import cv2
  # open cv stuff 
   
  img = cv2.imread(pngFileName)
  #img = np.zeros([10,10],dtype=np.uint) 

  import scipy.fftpack as fftp
  Img = fftp.fftn(img)
  img = fftp.ifftn(Img)

  cv2.imwrite(pngFileOutName,np.real(img))
  
  
  f = open(outFileName, 'w')
  f.write("wrote this  " + pngFileOutName) 
  f.write("\n")
  f.close()

