"""
Convert notebooks listed in `chapters` into latex and then a PDF.

Note:

 - The notebooks are first copied into the build_pdf directory (with a chapter
   number prepended).
"""
import re
import subprocess
import os
import glob

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

part0 = ['Preface']

part1 = ['Introduction',
            'Advection',
            'Acoustics',
            'Traffic_flow',
            'Burgers',
            'Nonconvex_scalar',
            'Shallow_water',
            'Shallow_tracer',
            'Euler']

part2 = ['Approximate_solvers',
         'Burgers_approximate',
         'Shallow_water_approximate',
         'Euler_approximate',
         'Euler_compare']

chapters = part0 + part1 + part2

# For testing purposes:
if 0:
    chapters = ['Preface',
                'Introduction',
                'Advection',
                'Acoustics']


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
os.system('cp latexdefs.tex '+build_dir)

# fix interact import statements in exact_solver demo codes:
os.chdir(os.path.join(build_dir, 'exact_solvers'))
print('Fixing interacts in %s' % os.getcwd())
files = glob.glob('*.py')
for file in files:
    infile = open(file,'r')
    lines = infile.readlines()
    infile.close()
    #print('Fixing %s' % file)
    with open(file,'w') as outfile:
        for line in lines:
            line = re.sub(r"context = 'notebook'", "context = 'pdf'", line)
            line = re.sub(r'from ipywidgets import interact',
                          'from utils.snapshot_widgets import interact', line)
            outfile.write(line)
            
os.chdir('../..')

# execute all the notebooks:

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
            line = re.sub(r"context = 'notebook'", "context = 'pdf'", line)
            # The next part is deprecated
            line = re.sub(r'from ipywidgets import interact',
                          'from utils.snapshot_widgets import interact', line)
            line = re.sub(r'Widget Javascript not detected.  It may not be installed or enabled properly.',
                          '', line)
            #line = re.sub(r"#sns.set_context('paper')",
            #              r"sns.set_context('paper')", line)
            output.write(line)
    args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--ExecutePreprocessor.kernel_name=python2",
            "--output", output_filename,
            "--ExecutePreprocessor.timeout=60", build_dir+output_filename]
    subprocess.check_call(args)

os.chdir(build_dir)
os.system('python3 -m bookbook.latex --output-file riemann --template riemann.tplx')
os.system('python3 ../fix_latex_file.py')
os.system('pdflatex riemann')
os.system('bibtex riemann')
os.system('pdflatex riemann')
os.system('pdflatex riemann')
os.system('cp riemann.pdf ..')
os.chdir('..')
