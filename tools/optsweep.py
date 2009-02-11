#!/usr/bin/env python

#  Copyright (c) 2009 Maciek Brodowicz
# 
#  Distributed under the Boost Software License, Version 1.0. (See accompanying 
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

"""
Perform parameter sweep runs of an application.
"""

import sys, os, getopt, time, string
from types import *

# subprocess instantiation wrapper;
# unfortunately older Python still lurks on some machines
try:
    import subprocess
    
    class Process:
        _proc = None
        
        def __init__(self, cmd):
            self._proc = subprocess.Popen(cmd, stderr = subprocess.STDOUT, stdout = subprocess.PIPE)
            
        def wait(self):
            return self._proc.wait()

        def read(self):
            return self._proc.stdout.read()
            
except ImportError, err:
    # no "subprocess"; use older popen module
    import popen2

    class Process:
        _proc = None
        
        def __init__(self, cmd):
            self._proc = popen2.Popen4(cmd)

        def wait(self):
            return self._proc.wait()

        def read(self):
            return self._proc.fromchild.read()


# print usage info and exit with an error code
def usage(rc = 2):
    print '\nUsage:', sys.argv[0], '[options] application [const_options]'
    print 'options:'
    print '  -a name,range : specify range of values to be passed as option'
    print '                  "name" to the application; "range" is a python'
    print '                  expression producing list of values'
    print '  -n            : don\'t stream results to stdout'
    print '  -r number     : repeat each test "number" of times'
    print '  -o filename   : capture results to file "filename"'
    print '  -p number     : pad each test with "number" of seconds'
    print '  -t            : prefix application command line with "time"'
    print '  -T command    : as -t, but use explicit timing "command"'
    print '  -x list       : exclude cases with argument tuples matching the'
    print '                : items in the "list" (python expression)'
    print '  -k number     : insert generated options starting at index'
    print '                  "number" in the application command line'
    print '                  (default: append at the end of line);'
    print '                  application pathname is always at index 0'
    print '  -h            : prints this message'
    sys.exit(rc)


# write string to each open file descriptor in the list
def writeres(s, fdlist):
    for fd in fdlist:
        fd.write(s)


# select next option set to run
def next(ixd, opts, optv):
    if not ixd: return None
    for k in opts:
        ixd[k] += 1
        if ixd[k] >= len(optv[k]): ixd[k] = 0
        else: return ixd
    return None
        

# run the application and capture its output and error streams
def run(cmd, outfl):
    proc = Process(cmd)
    while True:
        s = proc.read()
        if s: writeres(s, outfl)
        else: break
    return proc.wait()


# wrapper for conversion of integer options
def intopt(opta, optn):
    try:
        return int(opta)
    except Exception, err:
        print 'Error: invalid argument to option "'+optn+'":', opta, '('+str(err)+')'
        usage()


# human-readable version of current timestamp
def timestr():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# quote option arguments to protect blanks
def quoteopts(olist, qchar = '"'):
    s = ''
    for o in olist:
        if type(o) is not StringType: o = str(o)
        for c in o:
            if c not in nonquot:
                s += ' '+qchar+o+qchar
                break
        else: s += ' '+o
    return s


# create separator with cenetred string
def sepstr(sepch = '-', s = ''):
    if s: s = ' '+s.strip()+' '
    nl = (seplen-len(s))/2
    nr = seplen-len(s)-nl
    # make sure it looks like separator
    if nl < 3: nl = 3
    if nr < 3: nr = 3
    return nl*sepch+s+nr*sepch


if __name__ == '__main__':
    # parse command line
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'a:hk:no:p:r:tT:x:')
    except getopt.GetoptError, err:
        print 'Error:', str(err)
        usage()

    # option value lists, option names, # test repetitions, temporal pad
    options, optnames, nrep, tpad = {}, [], 1, 0
    # external timing flag, external profiling command, option insertion index
    timef, profcmd, genoix = False, None, None
    # stdout usage flag, result file name, list of output file descriptors
    stdoutf, ofile, ofhs = True, None, []
    # exclusion list
    excl = []
    # execution counters: app. runs, unique configs, errors, excluded configs
    runs, configs, erruns, excnt = 0, 0, 0, 0
    # separator length for pretty printing
    seplen = 78
    # non-quotable characters
    nonquot = string.letters+string.digits+'-+='

    # process options
    for o, a in opts:
        if o == '-a':
            wl = a.split(',', 1)
            if len(wl) != 2:
                print 'Error: malformed argument to "-a" option:', a
                usage()
            try:
                options[wl[0]] = eval(wl[1])
            except Exception, err:
                print 'Error: failed to evaluate "'+wl[1]+'", check syntax'
                usage()
            if type(options[wl[0]]) not in (ListType, TupleType):
                options[wl[0]] = (options[wl[1]],)
            if not len(options[wl[0]]):
                print 'Error: empty value list for option "'+wl[0]+'"'
                usage()
            optnames.append(wl[0])
            if len(options[wl[0]]) == 1:
                print 'Warning: single value for option "'+wl[0]+'":', options[wl[0]]
        elif o == '-n': stdoutf = False
        elif o == '-p': tpad = intopt(a, o)
        elif o == '-r': nrep = intopt(a, o)
        elif o == '-o': ofile = a
        elif o == '-k': genoix = intopt(a, o)
        elif o == '-t': timef = True
        elif o == '-t': timef, profcmd = True, a
        elif o == '-x':
            try:
                excl = eval(a)
            except Exception, err:
                print 'Error: invalid exclusion list: ', str(a)
                usage()
        elif o == '-h': usage(0)

    if not args:
        print 'Error: no test application specified'
        usage()
    if ofile:
        try:
            of = open(ofile, 'w')
            ofhs.append(of)
        except Exception, err:
            print 'Error: failed to open "'+ofile+'"'
            sys.exit(1)
    if stdoutf: ofhs.append(sys.stdout)

    # form parts of application command line
    if genoix is None: genoix = len(args)
    prefix, appopts = args[:genoix], args[genoix:]
    if timef:
        prefix = [(profcmd, '/usr/bin/time')[not profcmd]]+prefix

    # initialize current option index dictionary
    optix = {}
    for k in options: optix[k] = 0

    # beginning banner
    writeres(sepstr('=')+'\n', ofhs)
    writeres('Start date: '+timestr()+'\n', ofhs)
    writeres('Command:'+quoteopts(sys.argv)+'\n', ofhs)
    # test loop
    while optix != None:
        # start building command line
        cmd = prefix[:]
        configs += 1
        # add generated options
        vallst = []
        for k in optnames:
            val = options[k][optix[k]]
            vallst += [val]
            if type(val) is not StringType: val = str(val)
            if k: cmd += [k, val]
            else: cmd.append(val)
        # check for exclusion
        if vallst in excl:
            writeres(sepstr('=')+'\nSkipping:'+quoteopts(cmd)+'\n', ofhs)
            optix = next(optix, optnames, options)
            excnt += 1
            continue
        # suffix with constant options and app arguments
        cmd += appopts
        writeres(sepstr('=')+'\nExecuting:'+quoteopts(cmd)+'\n', ofhs)
        # run test requested number of times
        for i in range(nrep):
            txt = 'BEGIN RUN '+str(i+1)+' @ '+timestr()
            writeres(sepstr('*', txt)+'\n', ofhs)
            runs += 1
            rc = run(cmd, ofhs)
            txt = 'END RUN '+str(i+1)+' @ '+timestr()
            if rc: erruns += 1
            outs = sepstr('-', txt)
            outs += '\nReturn code: '+str(rc)+'\n'+sepstr()+'\n'
            writeres(outs, ofhs)
            time.sleep(tpad)
        optix = next(optix, optnames, options)
    # final banner
    writeres('='*seplen+'\n', ofhs)
    writeres('End date: '+timestr()+'\n', ofhs)
    writeres('Configurations: '+str(configs)+'\n', ofhs)
    writeres('Excluded: '+str(excnt)+'\n', ofhs)
    writeres('Test runs: '+str(runs)+'\n', ofhs)
    writeres('Errors: '+str(erruns)+'\n', ofhs)
    writeres('='*seplen+'\n', ofhs)
    
    for f in ofhs:
        if f != sys.stdout: f.close()
