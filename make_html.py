import re
import subprocess
import os


chapters = ['Index',
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

#chapters = ['Index','Introduction','Shallow_water']


os.system('mkdir -p riemann_book_files')
os.system('ln -s exact_solvers riemann_book_files/')
os.system('ln -s utils riemann_book_files/')
os.system('cp *.html riemann_book_files/')
os.system('cp -r figures riemann_book_files/')
os.chdir('riemann_book_files')

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
            for j, chapter_name in enumerate(chapters):
                line = re.sub(chapter_name+'.ipynb', chapter_name+'.html', line)
            output.write(line)
    args = ["jupyter", "nbconvert", "--to", "html", "--execute",
            "--ExecutePreprocessor.kernel_name=python2",
            "--output", html_filename,
            "--ExecutePreprocessor.timeout=60", output_filename]
    subprocess.check_call(args)

print("Open riemann_book_files/Index.html for index")
