




# Change "volcano" to the name of the specific one
# Make sure that you are within the "temp_gif_image_holder" directory

# CAUTION
# If you run into memory issues you can edit the ImageMagick limits
# on Ubuntu via sudo gedit /etc/ImageMagick-6/policy.xml
# I changed the memory limit to 4 gigabytes
convert -delay 25 *.png volcano.gif
