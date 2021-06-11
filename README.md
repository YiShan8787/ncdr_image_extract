# ncdr_image_extract
 a python script for extract Front

# How to use

 ## Windows
 
    1. modify the config.ini file (optional)
    
    2. click "multi_image_extract.exe"
    
 ## Linux 
 
    1. modify the config.ini file (optional)
    
    2. execute the python script "multi_image_extract.py"
    
# Method
   
   1. Load image
   2. Change image to HSV
   3. Use the red and blue color filter
   4. Fit the filter mask to origin image, then I called it "extract_image"
   5. Change gray scale on extracte_image
   6. Do dilation operation on the gray scale image
   7. Use the connectedcomponent to find the "max connected componet", then I called it "extract_front_image"
   8. Save the extract_front_image
    
# Source Code

  ## Requirement
  
    1. python
    2. opencv
    3. numpy
