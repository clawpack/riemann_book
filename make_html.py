"""
Convert notebooks listed in `chapters` into html files in the directory
`riemann_book_files`.

Note:

 - The notebooks are first copied into the directory `build_html` and pre-processed
   to use static_widgets (or jsanimation_widgets in the Introduction only),
   and cross references to other notebooks have `.ipynb` replaced by `.html`
   so the links work in the html files.

 - The directories `utils`, `exact_solvers`, and `figures` are also copied
   in before processing.

 - The list `all_chapters` is used to replace cross-reference links
   `chapter.ipynb` by `chapter.html`.

 - The resulting html files are moved to the directory `html`

"""

import re
import subprocess
import os


all_chapters = ['Preface',
            'Index',
            'Introduction',
            'Advection',
            'Acoustics',
            'Traffic_flow',
            'Shallow_water',
            'Shallow_tracer',
            'Euler_equations',
            'Euler_approximate_solvers',
            'Traffic_variable_speed',
            'Nonlinear_elasticity',
            'Euler_equations_TammannEOS',
            'Nonconvex_Scalar_Osher_Solution',
            'Pressureless_flow',
            'Kitchen_sink_problem']

chapters = all_chapters  # which chapters to process

# test on a subset:
#chapters = ['Index','Introduction','Shallow_water']

template_path = os.path.realpath('./html.tpl')


os.system('mkdir -p build_html')  # for intermediate processing
# copy some things needed for processing
os.system('cp -r exact_solvers build_html/')
os.system('cp -r utils build_html/')
os.system('mkdir -p build_html/img')  # for viewing images
os.system('cp -r figures build_html/img/')

os.system('mkdir -p html')  # for viewing final html files
os.system('cp *.html html/')

os.chdir('build_html')

for i, chapter in enumerate(chapters):
    filename = chapter + '.ipynb'
    print("Processing %s" % filename)
    input_filename = os.path.join('..',filename)
    with open(input_filename, "r") as source:
        lines = source.readlines()
    output_filename = filename
    html_filename = chapter+'.html'

    with open(output_filename, "w") as output:
        if chapter == 'Introduction':
            widget = 'from utils.jsanimate_widgets import interact'
        else:
            widget = 'from utils.snapshot_widgets import interact'

        for line in lines:
            line = re.sub(r'from ipywidgets import interact', widget, line)
            for j, chapter_name in enumerate(all_chapters):
                line = re.sub(chapter_name+'.ipynb', chapter_name+'.html', line)
            output.write(line)

    args = ["jupyter", "nbconvert", "--to", "html", "--execute",
            "--ExecutePreprocessor.kernel_name=python2",
            "--output", html_filename,
            "--template", template_path,
            "--ExecutePreprocessor.timeout=60", output_filename]
    subprocess.check_call(args)

    os.system('mv %s ../html/' % html_filename)

print("You may delete the directory build_html")
print("Open html/Index.html for the index")

