"""
Use this with diagnostics so that it can be changed as necessary
For example, sometimes want on stdout maybe or in a scroll window....

Let's replace this with pythong logging stuff.
at least get it to the point where "make quick" in swap/test works.

There is some wierd stuff here because TimBL and I couldn't
figure out how to share the chatty global across modules.
Surely Daniel K and DanC should be able figure that out.

"""

import sys
import os, traceback
from codecs import utf_8_encode

##lineCount = [0]
def progress(*args):
    level = len(traceback.extract_stack())
    sys.stderr.write(" "*level)
    for a in args:
        i = 0
        a = unicode(a)
        while 1:
##    lineCount[0] += 1
            i = a.find("\n", i)
            if i < 0: break
            a = a[:i+1] + (" "*level) + a[i+1:]
            i = i+1
        q = utf_8_encode(u"%s " % (a,))[0]
        sys.stderr.write(q)
##        if lineCount[0] > 20:
##            lineCount[0] = 0
##            sys.stdin.readline()
    sys.stderr.write("\n")


global chatty_flag # verbosity debug flag
#chatty_flag  =0
chatty_flag = int(os.environ.get("CWM_VERBOSITY", 0))

global print_all_file_names
print_all_file_names = int(os.environ.get("CWM_LIST_FILES", 0))
global file_list
file_list = []


global tracking
tracking = 0  # Are we keeping reason information for proof generation?

def setTracking(x):
    global tracking
    chatty_flag = x

def setVerbosity(x):
    global chatty_flag
    chatty_flag = x

def verbosity():
    global chatty_flag
    return chatty_flag

    
def printState(prefix="#trace# "):
    """Output the (caller's) function name and local variables
    """
    import sys
    frame = sys._getframe(1)    # caller's state
    sys.stderr.write(prefix+
                     str(frame.f_code.co_name)+" "+
                     str(frame.f_locals['self'])+"\n")
    for varname in frame.f_locals.keys():
        if varname == "self": continue
        sys.stderr.write("%s    %-8s: %s\n" %
                         (prefix, varname, frame.f_locals[varname]))
    sys.stderr.write(prefix+"\n")
   

