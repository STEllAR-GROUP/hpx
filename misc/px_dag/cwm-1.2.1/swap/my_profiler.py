"""Profile.py

This will try to profile cwm.


"""

import hotshot, hotshot.stats
from cwm import doCommand
import os, sys


def runProfile(logfile):
    profiler = hotshot.Profile(logfile)
    saveout = sys.stdout
    fsock = open('/dev/null', 'w')
    sys.stdout = fsock
    profiler.runcall(doCommand)
    sys.stdout = saveout
    fsock.close()
    profiler.close()
    stats = hotshot.stats.load(logfile)
    stats.strip_dirs()
    stats.sort_stats('cumulative', 'time', 'calls')
    stats.print_stats(60)

if __name__ == '__main__':
    try:
        os.remove('/tmp/profile.log')
    except OSError:
        pass
    runProfile('/tmp/profile.log')

