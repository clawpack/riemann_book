
import re
import subprocess
import os
import glob

fname = 'riemann.tex'
lines = open(fname,'r').read()

oldpat = r'\href'
newpat = r'\hreffoot'
lines = lines.replace(oldpat, newpat)

# Attempt to replace url's to other notebooks by cross-references to 
# chapters, but then I noticed this is already done by bookbook 
# for links to other notebooks that are in this directory. 
# 
# I suggest we might want to change "Section" to "Chapter" in 
#   bookbook/filter_links.py
# Never mind -- I see ketch already contributed this to bookbook.
#
# For links to notebooks that are not found in this directory, bookbook
# leaves it as \url{notebook.ipynb} and the code below changes this to
# "Chapter \ref{notebook}", which then renders as "Chapter ??".  
# We should figure out what we want to these links to point to instead,
# maybe to online versions of other notebooks?  Or eliminate all such
# cross references for the printed book?

regexp = re.compile(r"\\url{(?P<chap>[^}]*).ipynb}")
result = regexp.search(lines)
while result:
    chap = result.group('chap')
    nbook = chap + '.ipynb'
    lines = lines.replace(r'\url{%s}' % nbook, r'Chapter~\ref{%s}' % chap)
    result = regexp.search(lines)


# Write out resulting file:
outfile = open('riemann.tex','w')
outfile.write(lines)
outfile.close()