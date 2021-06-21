#!/usr/bin/env python3
'''
Copyright (c) 2020 ETH Zurich

SPDX-License-Identifier: BSL-1.0
Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
'''

import json
import os

from pyutils import args, default_vars as var, env, log

script_dir = os.path.dirname(os.path.abspath(__file__))

def mkdirp(absolute_path_file):
    # Create the directories of the run_output files if does not exist
    import ntpath
    nested_dir, _ = ntpath.split(absolute_path_file)
    if nested_dir != '' and not os.path.exists(nested_dir):
            os.makedirs(nested_dir)

@args.command(description='main script for '+ var._project_name +' pyutils')
@args.arg('--verbose',
          '-v',
          action='count',
          default=0,
          help='increase verbosity (use -vv for all debug messages)')
@args.arg('--logfile', '-l', help='path to logfile')
def driver(verbose, logfile):
    log.set_verbosity(verbose)
    if logfile:
        log.log_to_file(logfile)


@driver.command(description='build '+ var._project_name)
@args.arg('--cmake-only',
          action='store_true',
          help='only execute CMake but do not build')
@args.arg('--build-dir', '-o', required=True, help='build directory')
@args.arg('--build-type', '-b', choices=['release', 'debug'], required=True)
@args.arg('--environment', '-e', help='path to environment file')
@args.arg('--install-dir', '-i', help='install directory')
@args.arg('--source-dir', help= var._project_name +' source directory')
@args.arg('--target', '-t', nargs='+', help='make targets to build')
def build(build_type, environment, target, source_dir, build_dir, install_dir,
          cmake_only):
    import build

    if source_dir is None:
        source_dir = os.path.abspath(os.path.join(script_dir, os.path.pardir))

    #env.set_cmake_arg('CMAKE_BUILD_TYPE', build_type.title())
    env.set_cmake_arg('PYUTILS_HPX_WITH_FETCH_ASIO', 'ON')
    env.set_cmake_arg('PYUTILS_HPX_WITH_MALLOC', 'jemalloc')
    env.set_cmake_arg('PYUTILS_HPX_WITH_TESTS_BENCHMARKS', 'ON')
    env.set_cmake_arg('PYUTILS_HPX_WITH_MAX_CPU_COUNT', '72')
    env.set_cmake_arg('PYUTILS_CMAKE_BUILD_TYPE', 'Release')
    env.set_cmake_arg('-GNinja', '')
    # For possibly more stable/focused results
    env.set_cmake_arg('PYUTILS_HPX_WITH_TIMER_POOL', 'OFF')
    env.set_cmake_arg('PYUTILS_HPX_WITH_IO_POOL', 'OFF')

    if environment:
        env.load(environment)

    build.cmake(source_dir, build_dir, install_dir)
    if not cmake_only:
        build.make(build_dir, target)


try:
    from pyutils import buildinfo
except ImportError:
    buildinfo = None

if buildinfo:

    @driver.command(description='run '+ var._project_name +' tests')
    @args.arg('--run-mpi-tests',
              '-m',
              action='store_true',
              help='enable execution of MPI tests')
    @args.arg('--perftests-only',
              action='store_true',
              help='only run perftests binaries')
    @args.arg('--verbose-ctest',
              action='store_true',
              help='run ctest in verbose mode')
    @args.arg('--examples-build-dir',
              help='build directory for examples',
              default=os.path.join(buildinfo.binary_dir, 'examples_build'))
    @args.arg('--build-examples',
              '-b',
              action='store_true',
              help='enable building of '+ var._project_name +' examples')
    def test(run_mpi_tests, perftests_only, verbose_ctest, examples_build_dir,
             build_examples):
        import test

        if perftests_only:
            test.run_perftests()
        else:
            test.run(run_mpi_tests, verbose_ctest)

        if build_examples:
            test.compile_and_run_examples(examples_build_dir, verbose_ctest)


@driver.command(description='performance test utilities')
def perftest():
    pass


if buildinfo:

    @perftest.command(description='run performance tests')
    @args.arg('--local',
              default=False,
              help='run without slurm')
    @args.arg('--scheduling-policy',
              '-s',
              default='local-priority-fifo',
              help='scheduling policy (default is local-priority-fifo)')
    @args.arg('--threads',
              default='4',
              type=int,
              help='number of runs to do for each test')
    @args.arg('--run_output',
              '-o',
              required=True,
              help='output file path, extension .json is added if not given')
    @args.arg('--extra-opts',
              nargs='+',
              type=str,
              help='extra arguments to pass to the test\nWarning prefer = to \
              space to assign values to hpx options')
    def run(local, scheduling_policy, threads, run_output, extra_opts):
        # options
        scheduling_policy='--hpx:queuing=' + scheduling_policy
        threads='--hpx:threads=' + str(threads)
        extra_opts = ' '.join(extra_opts).lstrip()

        if not run_output.lower().endswith('.json'):
            run_output += '.json'
        # Create directory of file if does not exists yet
        mkdirp(run_output)

        import perftest
        data = perftest.run(local, scheduling_policy, threads, extra_opts)
        with open(run_output, 'w') as outfile:
            json.dump(data, outfile, indent='  ')
            log.info(f'Successfully saved perftests output to {run_output}')


@perftest.command(description='plot performance results')
def plot():
    pass


def _load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)


@plot.command(description='plot performance comparison')
@args.arg('--output', '-o', required=True, help='output directory')
@args.arg('--input', '-i', required=True, nargs=2, help='two input files')
def compare(output, input):
    mkdirp(output)
    from perftest import plot
    exitcode = plot.compare(*(_load_json(i) for i in input), output)
    print("exit code in compare function " + str(exitcode))
    raise SystemExit(exitcode)

@plot.command(description='plot performance history')
@args.arg('--output', '-o', required=True, help='output directory')
@args.arg('--input',
          '-i',
          required=True,
          nargs='+',
          help='any number of input files')
@args.arg('--date',
          '-d',
          default='job',
          choices=['commit', 'job'],
          help='date to use, either the build/commit date or the date when '
          'the job was run')
@args.arg('--limit',
          '-l',
          type=int,
          help='limit the history size to the given number of results')
def history(h_output, h_input, date, limit):
    from perftest import plot

    plot.history([_load_json(i) for i in h_input], h_output, date, limit)


@plot.command(description='plot backends comparison')
@args.arg('--output', '-o', required=True, help='output directory')
@args.arg('--input',
          '-i',
          required=True,
          nargs='+',
          help='any number of input files')
def compare_backends(cb_output, cb_input):
    from perftest import plot

    plot.compare_backends([_load_json(i) for i in cb_input], cb_output)


# We disable this warning as the parameter has a default value (see argparse)
# pylint: disable=no-value-for-parameter
with log.exception_logging():
    driver()
