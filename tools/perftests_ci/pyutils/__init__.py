# -*- coding: utf-8 -*-
'''
Copyright (c) 2020 ETH Zurich

SPDX-License-Identifier: BSL-1.0
Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

pyutils - A module to run performance test CI for HPX

It includes argument parsing, logging and test packages
'''

import sys


if sys.version_info < (3, 6):
    raise Exception('Python 3.6 or newer is required')
