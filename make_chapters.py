import re
import subprocess

chapters = ['Traffic_flow',
            'Shallow_water',
            'Euler_approximate_solvers',
            'Traffic_variable_speed',
            'Nonlinear_elasticity',
            'Euler_equations_TammannEOS',
            'Pressureless_flow',
            'Kitchen_sink_problem']

for i, chapter in enumerate(chapters):
    filename = chapter + '.ipynb'
    with open(filename, "r") as source:
        lines = source.readlines()
    output_filename = str(i).zfill(2)+'-'+filename
    with open(output_filename, "w") as output:
        for line in lines:
            output.write(re.sub(r'from ipywidgets import interact', 'from utils.snapshot_widgets import interact', line))
    args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--ExecutePreprocessor.kernel_name=python2",
            "--output", output_filename,
            "--ExecutePreprocessor.timeout=60", output_filename]
    subprocess.check_call(args)
