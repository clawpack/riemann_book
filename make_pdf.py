"""
Convert notebooks listed in `chapters` into latex and then a PDF.

Note:

 - The notebooks are first copied into the build_pdf directory (with a chapter
   number prepended).
"""
import re
import subprocess
import os

chapters = ['Preface',
            'Introduction',
            'Traffic_flow',
            'Shallow_water',
            'Approximate_solvers',
            'Euler_approximate_solvers',
            'Traffic_variable_speed',
            'Nonlinear_elasticity',
            'Euler_equations_TammannEOS',
            'Nonconvex_scalar',
            'Pressureless_flow',
            'Kitchen_sink_problem']

# To test a subset, adjust the list of chapters and
# remove the build_pdf directory before running this script.

#chapters = ['Shallow_water']

chapters = ['Preface',
            'Introduction',
            'Traffic_flow',
            'Shallow_water',
            'Approximate_solvers',
            'Traffic_variable_speed',
            'Euler_equations_TammannEOS',
            'Nonconvex_scalar']

build_dir = 'build_pdf/'
if not os.path.exists(build_dir):
    os.makedirs(build_dir)

os.system('cp -r exact_solvers '+build_dir)
os.system('cp -r utils '+build_dir)
os.system('cp *.html '+build_dir)
os.system('cp -r figures '+build_dir)
os.system('cp riemann.tplx '+build_dir)
os.system('cp *.cls '+build_dir)
os.system('cp *.css '+build_dir)
os.system('cp riemann.bib '+build_dir)

for i, chapter in enumerate(chapters):
    filename = chapter + '.ipynb'
    with open(filename, "r") as source:
        lines = source.readlines()
    output_filename = str(i).zfill(2)+'-'+filename
    with open(build_dir+output_filename, "w") as output:
        for line in lines:
            for j, chapter_name in enumerate(chapters):
                # fix cross references to other chapters
                line = re.sub(chapter_name+'.ipynb',
                              str(j).zfill(2)+'-'+chapter_name+'.ipynb', line)
            line = re.sub(r'from ipywidgets import interact',
                          'from utils.snapshot_widgets import interact', line)
            line = re.sub(r'Widget Javascript not detected.  It may not be installed or enabled properly.',
                          '', line)
            output.write(line)
    args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--ExecutePreprocessor.kernel_name=python2",
            "--output", output_filename,
            "--ExecutePreprocessor.timeout=60", build_dir+output_filename]
    subprocess.check_call(args)

os.chdir(build_dir)
os.system('python3 -m bookbook.latex --output-file riemann --template riemann.tplx')
os.system('pdflatex riemann')
os.system('bibtex riemann')
os.system('pdflatex riemann')
os.system('pdflatex riemann')
os.system('cp riemann.pdf ..')
os.chdir('..')
