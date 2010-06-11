#!/usr/bin/env python
"""importList.py

Generate a list of files imported by the files given on the command line

"""


from modulefinder import ModuleFinder
import os.path
import os
import sys
import uripath



def main(argv):
    path = sys.path[:]
    path[0] = os.path.dirname(argv[0])
    mf = ModuleFinder(path)
    for f in argv:
        mf.run_script(f)
    paths = sorted(list(set([os.path.abspath(x.__file__) for x in mf.modules.values() if x.__file__])))
    cwd = os.getcwd()
    paths = [x for x in paths if x.startswith(cwd)]
    m = len(cwd) + 1
    paths = argv + [x[m:] for x in paths]
    print ' '.join(paths)
    
    




if __name__ == '__main__':
    main(sys.argv)
