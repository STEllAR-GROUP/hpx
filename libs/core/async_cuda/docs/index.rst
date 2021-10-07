..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_async_cuda:

============
async_cuda
============

This library adds a simple API that enables the user to retrieve a future 
from a cuda stream. Typically, a user may launch one or more kernels
and then get a future from the stream that will become ready when those
kernels have completed. The act of getting a future from the 
`cuda_stream_helper` object in this library hides the creation of a
cuda stream event and the attachment of this event to the promise
that is backing the future returned.

The usage is best illustrated by looking at an example

.. code-block:: C++

    // create a cuda target using device number 0,1,2...
    hpx::cuda::experimental::target target(device);
    // create a stream helper object
    hpx::cuda::experimental::cuda_future_helper helper(device);

    // launch a kernel and return a future
    auto fn = &cuda_trivial_kernel<double>;
    double d = 3.1415;
    auto f = helper.async(fn, d);

    // attach a continuation to the future
    f.then([](hpx::future<void>&& f) {
        std::cout << "trivial kernel completed \n";
    }).get();

Kernels and CPU work may be freely intermixed/overlapped
and synchronized with futures.

It is important to note that multiple kernels may be launched
without fetching a future, and multiple futures may be obtained
from the helper. Please refer to the unit tests and examples
for further examples.

See the :ref:`API reference <modules_async_cuda_api>` of this module for more
details.
