# -*- coding: utf-8 -*-
'''
Copyright (c) 2020 ETH Zurich

SPDX-License-Identifier: BSL-1.0
Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

perftest - A package to compile and run performance tests
'''

from datetime import datetime, timezone
import json
import os
import pprint

from pyutils import default_vars as var, env, log, runtools


def _git_commit():
    from pyutils import buildinfo
    return runtools.run(['git', 'rev-parse', 'HEAD'],
                        cwd=buildinfo.source_dir).strip()


def _git_datetime():
    from pyutils import buildinfo
    posixtime = runtools.run(
        ['git', 'show', '-s', '--format=%ct',
         _git_commit()],
        cwd=buildinfo.source_dir)
    return datetime.fromtimestamp(int(posixtime), timezone.utc).isoformat()


def _now():
    return datetime.now(timezone.utc).astimezone().isoformat()


def run(local, scheduling_policy, threads, extra_opts):
    from pyutils import buildinfo

    binary = os.path.join(buildinfo.binary_dir, 'bin', 'future_overhead_report_test')
    command = [binary] + [str(scheduling_policy)] + [str(threads)]
    if extra_opts:
        command += extra_opts.split()

    if local:
        output = runtools.run(command)
    else:
        output = runtools.srun(command)

    data = json.loads(output)

    data[var._project_name] = {'commit': _git_commit(), 'datetime': _git_datetime()}
    data['environment'] = {
        'hostname': env.hostname(),
        'clustername': env.clustername(local),
        'compiler': buildinfo.compiler,
        'datetime': _now(),
        'envfile': buildinfo.envfile
    }
    log.debug('Perftests data', pprint.pformat(data))

    return data
