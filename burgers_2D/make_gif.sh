#!/bin/bash
# Install imagemagick if not there
# sudo apt-get install imagemagick
# need to expand the memory
convert -resize 15% -delay 10 -loop 0 figures/analytical*.png analytical.gif
convert -resize 15% -delay 10 -loop 0 figures/ml*.png ml.gif
