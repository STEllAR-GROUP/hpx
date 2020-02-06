#!/usr/bin/env python3
# Copyright (c) 2018 Parsa Amini
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# ## Synopsis
# ```
# usage: inspect_to_junit.py [-h] [source] [destination]
#
# Generate JUnit XML report from inspect output
#
# positional arguments:
#   source       File path to read inspect output from
#   destination  File path to write JUnit XML report to
#
# optional arguments:
#   -h, --help   show this help message and exit
# ```

import argparse
import re
import sys
from xml.dom import minidom
from collections import namedtuple

error_item = namedtuple('error_item', 'filename message')


def parse_inspect8_log(fh):
    line_pattern = re.compile('(.+):\ (\*.+\*.+)')
    split_pattern = re.compile(',\ (?=\*)')
    stipper_pattern = re.compile('<\/?\w+[^>]*>')

    errors = []

    for line in fh:
        m = line_pattern.match(line)
        if m:
            for message in split_pattern.split(m.group(2)):
                error = error_item(filename=m.group(1),
                                   message=stipper_pattern.sub('', message))
                errors.append(error)

    return errors


def convert(inspect8_log_fh):
    errors = parse_inspect8_log(inspect8_log_fh)

    doc = minidom.Document()
    suite = doc.createElement('testsuite')
    doc.appendChild(suite)
    suite.setAttribute('name', 'inspect')
    suite.setAttribute('errors', str(len(errors)))
    suite.setAttribute('failures', '0')
    suite.setAttribute('tests', str(len(errors)))

    if len(errors) == 0:
        case = doc.createElement('testcase')
        case.setAttribute('name', 'inspect')
        case.setAttribute('time', '')
        suite.appendChild(case)

    for error in errors:
        case = doc.createElement('testcase')
        case.setAttribute('name', error.filename)
        case.setAttribute('time', '')
        suite.appendChild(case)

        failure = doc.createElement('failure')
        case.appendChild(failure)

        failure.setAttribute('file', error.filename)
        failure.setAttribute('message', error.message)
        message = doc.createTextNode(error.message)
        failure.appendChild(message)

    return doc


def main():
    parser = argparse.ArgumentParser(
        description='Generate JUnit XML report from inspect html output')
    parser.add_argument('source',
                        type=argparse.FileType('r'),
                        nargs='?',
                        default=sys.stdin,
                        help='File path to read inspect html output from')
    parser.add_argument('destination',
                        type=argparse.FileType('w'),
                        nargs='?',
                        default=sys.stdout,
                        help='File path to write JUnit XML report to')

    args = parser.parse_args()
    report = convert(args.source)
    report.writexml(args.destination,
                    addindent='    ', newl='\n', encoding='utf-8')


if __name__ == '__main__':
    main()
