# -*- coding: utf-8 -*-

from datetime import datetime, timezone
import json
import os
import pprint

from pyutils import env, log, runtools


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


def run(domain, runs):
    from pyutils import buildinfo

    binary = os.path.join(buildinfo.binary_dir, 'tests', 'regression',
                          'perftests')

    output = runtools.srun([binary] + [str(d)
                                       for d in domain] + [str(runs), '-d'])
    data = json.loads(output)

    data['gridtools'] = {'commit': _git_commit(), 'datetime': _git_datetime()}
    data['environment'] = {
        'hostname': env.hostname(),
        'clustername': env.clustername(),
        'compiler': buildinfo.compiler,
        'datetime': _now(),
        'envfile': buildinfo.envfile
    }
    data['domain'] = list(domain)
    log.debug('Perftests data', pprint.pformat(data))

    return data
