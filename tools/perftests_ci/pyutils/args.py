# -*- coding: utf-8 -*-
'''
Copyright (c) 2020 ETH Zurich

SPDX-License-Identifier: BSL-1.0
Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
'''

import argparse
import inspect


class arg:
    def __init__(self, *args, **kwargs):
        self.args = [(args, kwargs)]

    def __call__(self, func):
        if isinstance(func, arg):
            self.args += func.args
            self.func = func.func
        else:
            self.func = func
        return self

    @property
    def __name__(self):
        return self.func.__name__


class Command:
    def __init__(self, func, parser):
        if isinstance(func, arg):
            func, args = func.func, func.args
        else:
            args = []
        for func_args, func_kwargs in args:
            parser.add_argument(*func_args, **func_kwargs)
        self.func = func
        self.parser = parser
        self.subparsers = None

    def __call__(self, args=None):
        if args is None:
            args = vars(self.parser.parse_args())
        parameters = inspect.signature(self.func).parameters
        kwargs = {k: v for k, v in args.items() if k in parameters}
        self.func(**kwargs)
        if self.subparsers is not None:
            args[self._command_name](args)

    @property
    def _command_name(self):
        return '_command_' + self.func.__name__

    def command(self, **kwargs):
        def inner(func):
            if self.subparsers is None:
                self.subparsers = self.parser.add_subparsers(
                    dest=self.func.__name__ + ' subcommand')
                self.subparsers.required = True

            parser = self.subparsers.add_parser(
                func.__name__.replace('_', '-'), **kwargs)
            subcommand = Command(func, parser)
            parser.set_defaults(**{self._command_name: subcommand})
            return subcommand

        return inner


def command(**kwargs):
    def inner(func):
        return Command(func, argparse.ArgumentParser(**kwargs))

    return inner
