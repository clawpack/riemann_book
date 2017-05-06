"""
Set path for rjleveque since I can't figure out how to install bookbook properly.
"""

from __future__ import print_function

import sys
sys.path.append('/Users/rjl/Install/bookbook')

from bookbook import latex
from pathlib import Path

latex.combine_and_convert(source_dir=Path('.'), 
    output_file=Path('riemann.tex'),
    pdf=False, template_file=Path('riemann.tplx'))

print('Created riemann.tex')
