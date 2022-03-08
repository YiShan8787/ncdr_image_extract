# ncdr_image_extract
 a python script for extract Front

# How to use

 ## exe file (for windows)
 
    1. modify the config.ini file (optional)
    
    2. click "multi_image_extract.exe"
    
 ## source code (recommend)
 
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

# Config
```
   origin_path =  weather_image        # where you put the weather image files
   destination_path =  extract_image   # where you hope to save the output images
   suffix = _extract                   # the suffix of the output images
   image_type = jpg                    # the image type of output images (recommend jpg or png)
   component_size = 700                # the component size we choose (if you want to choose shorter, make it small; Otherwise, make it bigger)
   txt_file_suffix = _txt_log          # the suffix of the output txt files

```

# Source Code

  ## Requirement
  
    1. python
    2. opencv-contrib-python     4.5.2.54  (important)
    3. numpy
