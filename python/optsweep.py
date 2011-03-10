#!/usr/bin/env python

#  Copyright (c) 2009 Maciej Brodowicz
# 
#  Distributed under the Boost Software License, Version 1.0. (See accompanying 
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

"""
Perform parameter sweep runs of an application.
"""

import sys, os, getopt, time, string
from re import compile
from types import *
from operator import *
from datetime import datetime
from signal import SIGKILL

# subprocess instantiation wrapper;
# unfortunately older Python still lurks on some machines
try:
    import subprocess
    
    class Process:
        _proc = None
        
        def __init__(self, cmd):
            self._proc = subprocess.Popen(cmd, stderr = subprocess.STDOUT, stdout = subprocess.PIPE, shell = (False, True)[type(cmd) == StringType])
            
        def wait(self):
            return self._proc.wait()

        def poll(self):
            return self._proc.poll()
        
        def pid(self):
            return self._proc.pid

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
        
        def poll(self):
            return self._proc.poll()

        def pid(self):
            return self._proc.pid

        def read(self):
            return self._proc.fromchild.read()


# print usage info and exit with an error code
def usage(rc = 2):
    print '\nUsage:', sys.argv[0], '[options] application [const_options]',
    print '''
Options:
 -a name,list : specify range of values, identified by "name", for a single
                option of the application;
                "list" is a python expression producing list of values
 -n           : don\'t stream results to stdout
 -r number    : repeat each test "number" of times
 -o filename  : capture results to file "filename"
 -d number    : delay test start by "number" of seconds
 -t command   : prefix application command line with profiling "command"
 -x list      : exclude cases with argument tuples matching any item in the
                "list" (python expression)
 -w seconds   : kill runs that take longer than "seconds" to complete (default
                360). 
 -i seconds   : interval to wait in between each poll of the run subprocesses
                (default 10).
 -b command   : run preprocessing "command" before starting test sequence for
                each configuration, applying option substitution
 -p command   : run postprocessing "command" after test sequence for each
                configuration, applying option substitution
 -h           : prints this message
'''
    sys.exit(rc)


# write string to each open file descriptor in the list
def writeres(s, fdlist): map(lambda x: x.write(s), fdlist)


# select next option set to run
def next(ixd, opts, optv):
    if not ixd: return None
    for k in opts:
        ixd[k] += 1
        if ixd[k] >= len(optv[k]): ixd[k] = 0
        else: return ixd
    return None
        

# run the application and optionally capture its output and error streams
def run(cmd, outfl = None, timeout = 360, tic = 10):
    start = datetime.now() 
    proc = Process(cmd)

    while proc.poll() is None:
        time.sleep(tic)
        now = datetime.now()
        if (now - start).seconds > timeout:
            print 'Error: "%s" timed out.' % cmd
            os.kill(proc.pid(), SIGKILL)
            os.waitpid(-1, os.WNOHANG)
            return -1

    while outfl:
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


# create separator with centered string
def sepstr(sepch = '-', s = ''):
    if s: s = ' '+s.strip()+' '
    nl = (seplen-len(s))/2
    nr = seplen-len(s)-nl
    # make sure it still looks like separator for oversized lines
    if nl < 3: nl = 3
    if nr < 3: nr = 3
    return nl*sepch+s+nr*sepch

# python has no conditional expression, so we need this to call if/else from
# eval
def if_else(pred, then, else_):
  if pred:
    return then
  return else_

# substitute all option ids in string with formatting keys
def optidsub(optids, s):
    # first pass - option subsitution
    for o in optids:
      s = s.replace(o, '%('+o+')s')
    return s


# run pre- or postprocessor
def runscript(cmdlst, options, ofhs, timeout, interval):
    for cmd in cmdlst:
        scr = cmd%options
        rc = run(scr, timeout, interval)
        if rc:
            writeres('Warning: command: "'+scr+'" returned '+str(rc)+'\n', ofhs)

    
if __name__ == '__main__':
    # parse command line
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'a:b:d:hno:p:r:t:x:w:i:')
    except getopt.GetoptError, err:
        print 'Error:', str(err)
        usage()

    # option value lists, option names, # test repetitions, temporal pad
    options, optnames, nrep, tpad = {}, [], 1, 0
    # external profiling command
    profcmd = None
    # stdout usage flag, result file name, list of output file descriptors
    stdoutf, ofile, ofhs = True, None, []
    # exclusion list, preprocessing command list, postprocessing command list
    excl, before, after = [], [], []
    # execution counters: app. runs, unique configs, errors, excluded configs
    runs, configs, erruns, excnt = 0, 0, 0, 0
    # separator length for pretty printing
    seplen = 78
    # timeout length/polling interval for process management
    timeout, interval = 360, 10
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
        elif o == '-d': tpad = intopt(a, o)
        elif o == '-r': nrep = intopt(a, o)
        elif o == '-o': ofile = a
        elif o == '-t': profcmd = a
        elif o == '-w': timeout = intopt(a, o)
        elif o == '-i': interval = intopt(a, o)
        elif o == '-x':
            try:
                excl = map(tuple, eval(a))
            except Exception, err:
                print 'Error: invalid exclusion list: ', str(a)
                usage()
        elif o == '-b': before += [a]
        elif o == '-p': after += [a]
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

    # form prototypes of application command line, pre- and postprocessor
    cmdproto = map(lambda o: optidsub(optnames, o), args)
    if profcmd: cmdproto = [profcmd]+cmdproto
    if before: before = map(lambda o: optidsub(optnames, o), before)
    if after: after = map(lambda o: optidsub(optnames, o), after)

    # initialize current option index dictionary
    optix = {}
    for k in options: optix[k] = 0

    # beginning banner
    writeres(sepstr('=')+'\n', ofhs)
    writeres('Start date: '+timestr()+'\n', ofhs)
    writeres('Command:'+quoteopts(sys.argv)+'\n', ofhs)
    # test loop
    while optix != None:
        configs += 1
        # create current instance of generated options
        vallst, optd = [], {}
        for k in optnames:
            val = options[k][optix[k]]
            if type(val) is not StringType: val = str(val)
            optd[k] = val
            vallst += [optd[k]]
        # check for exclusions
        if tuple(vallst) in excl:
            writeres(sepstr('=')+'\nSkipping:'+quoteopts(cmd)+'\n', ofhs)
            optix = next(optix, optnames, options)
            excnt += 1
            continue
        # run setup program
        if before: runscript(before, optd, ofhs, timeout, interval)
        # build command line
        cmd = map(lambda x: x%optd, cmdproto)
 
        # second pass - eval
        p = compile(r'eval\("([^"]*)"\)')

        for e in range(len(cmd)):
          while p.search(cmd[e]):
            ss = p.search(cmd[e]).expand(r'\1')
            cmd[e] = cmd[e].replace("eval(\"%s\")" % ss, str(eval(ss)))

        writeres(sepstr('=')+'\nExecuting:'+quoteopts(cmd)+'\n', ofhs)
        # run test requested number of times
        for i in range(nrep):
            txt = 'BEGIN RUN '+str(i+1)+' @ '+timestr()
            writeres(sepstr('*', txt)+'\n', ofhs)
            runs += 1
            rc = run(cmd, ofhs, timeout, interval)
            txt = 'END RUN '+str(i+1)+' @ '+timestr()
            if rc: erruns += 1
            outs = sepstr('-', txt)
            outs += '\nReturn code: '+str(rc)+'\n'+sepstr()+'\n'
            writeres(outs, ofhs)
            time.sleep(tpad)
        # run postprocessor
        if after: runscript(after, optd, ofhs, timeout, interval)

        optix = next(optix, optnames, options)
    # final banner
    writeres('='*seplen+'\n', ofhs)
    writeres('End date: '+timestr()+'\n', ofhs)
    writeres('Configurations: '+str(configs)+'\n', ofhs)
    writeres('Excluded: '+str(excnt)+'\n', ofhs)
    writeres('Test runs: '+str(runs)+'\n', ofhs)
    writeres('Errors: '+str(erruns)+'\n', ofhs)
    writeres('='*seplen+'\n', ofhs)
    # cleanup
    for f in ofhs:
        if f != sys.stdout: f.close()
