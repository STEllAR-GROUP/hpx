..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_async_cuda:

==========
async_cuda
==========

This library adds a simple API that enables the user to retrieve a future  from
a |cuda|_ stream. Typically, a user may launch one or more kernels and then get a
future from the stream that will become ready when those kernels have completed.
It is important to note that multiple kernels may be launched without fetching a
future, and multiple futures may be obtained from the helper. Please refer to
the unit tests and examples for further examples.

See the :ref:`API reference <modules_async_cuda_api>` of this module for more
details.
