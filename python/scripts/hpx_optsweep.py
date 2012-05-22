#! /usr/bin/env python
#
# Copyright (c) 2009 Maciej Brodowicz
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

"""
Perform parameter sweep runs of an application.
"""

import sys, os, getopt, time, string
import os.path as osp
from re import compile
from types import *
from operator import *
from datetime import datetime
from pickle import dump

HPX_VERSION = "1.0.0"
 
if osp.exists(osp.join(sys.path[0], "../hpx")):
  sys.path.append(osp.join(sys.path[0], ".."))
if osp.exists(osp.join(sys.path[0], "../share/hpx-"+HPX_VERSION+"/python/hpx")):
  sys.path.append(osp.join(sys.path[0], "../share/hpx-"+HPX_VERSION+"/python"))

from hpx.process import process

OPTSWEEP_VERSION = 0x10 # version (mostly for version tracking in pickle output)

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
 -o filename  : capture stdout and stderr to file "filename"
 -t filename  : save results to file "filename" in the pickle format
 -d number    : delay test start by "number" of seconds
 -x list      : exclude cases with argument tuples matching any item in the
                "list" (python expression)
 -w seconds   : kill runs that take longer than "seconds" to complete (default
                360). 
 -b command   : run preprocessing "command" before starting test sequence for
                each configuration, applying option substitution
 -p command   : run postprocessing "command" after test sequence for each
                configuration, applying option substitution
 -h           : prints this message
'''
    sys.exit(rc)


# write string to each open file descriptor in the list
def writeres(s, fdlist):
  for x in fdlist:
    x.write(s)
    x.flush()
    if x.fileno() != 1:
      os.fsync(x.fileno())


# select next option set to run
def next(ixd, opts, optv):
    if not ixd: return None
    for k in opts:
        ixd[k] += 1
        if ixd[k] >= len(optv[k]): ixd[k] = 0
        else: return ixd
    return None
        

# run the application and optionally capture its output and error streams
def run(cmd, outfl = None, timeout = 360):
    start = datetime.now() 
    proc = process(cmd)
    (timed_out, returncode) = proc.wait(timeout)
    now = datetime.now()

    while outfl:
        s = proc.read()
        if s: writeres(s, outfl)
        else: break

    if timed_out: 
      writeres('Command timed out.\n', outfl)

    return (returncode, now - start)


# wrapper for conversion of integer options
def intopt(opta, optn):
    try:
        return int(opta)
    except Exception, err:
        print 'Error: invalid argument to option "'+optn+'":', opta, '('+str(err)+')'
        usage()


# human-readable version of current timestamp
def timestr(t):
    return t.strftime("%Y-%m-%d %H:%M:%S")


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

# substitute all option ids in string with formatting keys
def optidsub(optids, s):
    # first pass - option subsitution
    for o in optids:
      s = s.replace(o, '%('+o+')s')
    return s


# run pre- or postprocessor
def runscript(cmdlst, options, ofhs, timeout):
    for cmd in cmdlst:
        scr = cmd%options
        (rc, walltime) = run(scr, timeout)
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
    # stdout usage flag, result file name, list of output file descriptors
    stdoutf, ofile, rfile, rf, ofhs = True, None, None, None, []
    # exclusion list, preprocessing command list, postprocessing command list
    excl, before, after = [], [], []
    # execution counters: app. runs, unique configs, errors, excluded configs
    runs, configs, erruns, excnt = 0, 0, 0, 0
    # separator length for pretty printing
    seplen = 78
    # timeout 
    timeout = 360
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
        elif o == '-t': rfile = a
        elif o == '-w': timeout = intopt(a, o)
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
            print 'Error: failed to open output file "'+ofile+'"'
            sys.exit(1)
    if rfile:
        try:
            rf = open(rfile, 'w')
        except Exception, err:
            print 'Error: failed to open result file "'+rfile+'"'
            sys.exit(1)
    if stdoutf: ofhs.append(sys.stdout)

    # form prototypes of application command line, pre- and postprocessor
    cmdproto = map(lambda o: optidsub(optnames, o), args)
    if before: before = map(lambda o: optidsub(optnames, o), before)
    if after: after = map(lambda o: optidsub(optnames, o), after)

    # initialize current option index dictionary
    results = {}
    optix = {}
    for k in options: optix[k] = 0

    start_date = datetime.now()

    # beginning banner
    writeres(sepstr('=')+'\n', ofhs)
    writeres('Start date: '+timestr(start_date)+'\n', ofhs)
    writeres('Command:'+quoteopts(sys.argv)+'\n', ofhs)

    if rf:
      results['data'] = {}
      results['header'] = {}
      results['schema'] = {}

      results['schema']['keys'] = tuple(optnames)
      results['schema']['values'] = ('wall_time','return_code')

      results['header']['version'] = OPTSWEEP_VERSION
      results['header']['start_date'] = start_date
      results['header']['command'] = tuple(sys.argv)

    try: 
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
        # TODO: add timeout options
        if before: runscript(before, optd, ofhs)
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
            txt = 'BEGIN RUN '+str(i+1)+' @ '+timestr(datetime.now())
            writeres(sepstr('*', txt)+'\n', ofhs)
            (rc, walltime) = run(cmd, ofhs, timeout)
            txt = 'END RUN '+str(i+1)+' @ '+timestr(datetime.now())
            runs += 1
            if rc: erruns += 1
            outs = sepstr('-', txt)
            outs += '\nReturn code: '+str(rc)+'\n'+sepstr()+'\n'
            writeres(outs, ofhs)
            if rf:   
              if not results['data'].has_key(tuple(vallst)):
                results['data'][tuple(vallst)] = [(walltime, rc)]
              else:
                results['data'][tuple(vallst)].append((walltime, rc))
            time.sleep(tpad)
        # run postprocessor
        # TODO: add timeout options
        if after: runscript(after, optd, ofhs)
     
        optix = next(optix, optnames, options)
        
    except: 
        from traceback import print_exc
        print_exc()
     
    end_date = datetime.now()

    # final banner
    writeres('='*seplen+'\n', ofhs)
    writeres('End date: '+timestr(end_date)+'\n', ofhs)
    writeres('Configurations: '+str(configs)+'\n', ofhs)
    writeres('Exclusions: '+str(excnt)+'\n', ofhs)
    writeres('Total runs: '+str(runs)+'\n', ofhs)
    writeres('Failed runs: '+str(erruns)+'\n', ofhs)
    writeres('='*seplen+'\n', ofhs)

    if rf:
      results['header']['end_date'] = end_date
      results['header']['configurations'] = configs 
      results['header']['exclusions'] = excnt
      results['header']['total_runs'] = runs
      results['header']['failed_runs'] = erruns

      # dump the results dictionary to the result file, using
      # pickle protocol version 2 with binary output
      dump(results, rf, 2)

      rf.close()

    # cleanup
    for f in ofhs:
        if f != sys.stdout: f.close()

