"""
From the top directory run
    python utils/syntax_highlight.py
in order to create .html versions of all the .py files in 
    exact_solvers/ and put them in exact_solvers/pandoc/
    utils/ and put them in utils/pandoc

Then in a notebook use markdown like this to refer to them:

    To examine the Python code for this chapter, see:

     - [exact_solvers/burgers.py](exact_solvers/pandoc/burgers.html)
     - [exact_solvers/burgers_demos.py](exact_solvers/pandocburgers_demos.html)

Also makes an index.html file in each pandoc directory.
"""


import os,sys,glob

hstyle = 'tango'     # light background
#hstyle = 'zenburn'   # dark background

topdir = os.getcwd()

#------------------------
header = """...

From [https://github.com/clawpack/riemann_book](https://github.com/clawpack/riemann_book)

```python
"""  
#------------------------

for pydir in ['exact_solvers', 'utils']:

    os.chdir(pydir)

    os.system('mkdir -p pandoc')
    index_file = os.path.join('pandoc','index.html')
    index = open(index_file,'w')
    index.write('<html><h1>Python code in %s</h1>\n' % pydir)
    index.write('<ul>\n')

    py_files = glob.glob('*.py')
    for py_name in py_files:

        #if py_name != 'burgers.py': continue  # to test on one file

        f = open(py_name).read()

        # make a new version for a .md file that has the code in a
        # ```python ... ```  code block, and a title at the top:

        title = "---\ntitle: %s/%s\n" % (pydir,py_name)

        f2 = title + header + f + '\n```\n'

        md_name = os.path.join('pandoc', os.path.splitext(py_name)[0] + '.md')
        with open(md_name, 'w') as tfile:
            tfile.write(f2)

        # run pandoc to perform syntax highlighting and produce html:

        html_name = os.path.join('pandoc', os.path.splitext(py_name)[0] + '.html')
        cmd = 'pandoc %s -s --highlight-style %s -o %s' % (md_name,hstyle,html_name)
        print(cmd)
        os.system(cmd)

        os.system('rm %s' % md_name)  # remove markdown version

        index.write('<li> <a href="%s">%s</a>' \
            % (os.path.splitext(py_name)[0] + '.html', py_name))
    
    index.write('</ul>\n</html>\n')
    index.close()
    os.chdir(topdir)
