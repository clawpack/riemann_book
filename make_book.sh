#!/bin/sh

# Need to update pandoc to 1.19.x
# Need to install bookbook

rm 0?-*.ipynb
python make_chapters.py

python3 run_bookbook.py   # rjleveque uses this instead of next line
#python3 -m bookbook.latex --output-file riemann --template riemann.tplx

pdflatex riemann
bibtex riemann
pdflatex riemann
pdflatex riemann
