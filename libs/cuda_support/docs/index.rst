..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _libs_cuda_support:

============
cuda_support
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
    hpx::cuda::target target(device);
    // create a stream helper object
    hpx::cuda::cuda_future_helper helper(device);

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

CMake variables
---------------

``HPX_WITH_CUDA`` - this is a general option that will enable both ``HPX_WITH_CUDA_SUPPORT``
and ``HPX_WITH_CUDA_COMPUTE`` when turned ``ON``.

``HPX_WITH_CUDA_SUPPORT=ON`` enables the building of this module which requires
only the presence of CUDA on the system and only exposes cuda+fuures support
(``HPX_WITH_CUDA_SUPPORT`` may be used when ``HPX_WITH_CUDA_COMPUTE=OFF``).

``HPX_WITH_CUDA_COMPUTE=ON`` enables building HPX compute features that allow parallel
algorithms to be passed through to the GPU/CUDA backend by using algorithms
in the ``hpx::compute::cuda`` namespace.

See the :ref:`API reference <libs_cuda_support_api>` of this module for more
details.
