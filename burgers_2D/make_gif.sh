#!/bin/bash
# Install imagemagick if not there
# sudo apt-get install imagemagick
convert -resize 20% -delay 20 -loop 0 figures/analytical*.png analytical.gif
convert -resize 20% -delay 20 -loop 0 figures/ml*.png ml.gif
