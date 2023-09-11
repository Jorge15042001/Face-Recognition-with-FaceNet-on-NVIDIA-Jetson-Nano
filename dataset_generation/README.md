# Dataset generation 

scripts used to capture audio and video


# Stereo calibration

## Chessboard pattern
Print a chessboard patter of **6 rows and 8 columns** *(You may also display the chessboard patter in the monitor)*

## Camera separation
Measure the separation (cm) between the cameras as precisely as possible
save this value in the configuration file *stereo_config.json* with the key *"separation"*

# Depth to pixel size
This is a constant that determines the size of a pixel in cm for a given depth. Accoriding to the formula
$$ px_size = depth_to_pixel_size * depth $$

This allows to correctly transform distance in px to distances in cm at many depths

**The value of this constants should be manually calculated** by placing object of known geometry at different depths; measuring their sizes in pixels at different depths and finally making a linear regression *(You might also be able to compute this value from the camera focal length and fov)*

## Calibration process

### Image acquisition process

place the chessboard pattern in front of the cameras run

```bash
    python3 calibration_images.py ./stereo_config.json ./images/
```
In the live preview you can press 
    "s" to preview the image
        "s" to save the previewed image
        "q" to ignore the previewed image
    "q" to finish taking images


### Stereo Calibration process 

Now you can run the stereo calibration script

```bash
    python3 stereo_calibration.py ./images/ ./stereoMap.xml ./stereo_config.json
```

Note that ./stereoMap.xml will be create by the program *(or overwritten if exists)*

images/ is the path containing the images from previews step
stereo_config.json will be automatically filled
