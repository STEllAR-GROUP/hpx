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

```
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
```
Kernels and CPU work may be freely intermixed.

It is important to note that multiple kernels may be launched
without fetching a future, and multiple futures may be obtained
from the helper. Please referr to the unit tests and examples
for more information.

============
CMake variables
============
HPX_WITH_CUDA_SUPPORT=ON will enable the building of this module which requires
only the presence of CUDA on the system and only exposes cuda+fuures support.

The CMake setting for HPX_WITH_CUDA=ON (or HPX_WITH_CUDA_COMPUTE=ON) enable more
sophisticated CUDA based integration using hpx::compute::cuda and allow the user
to execute parallel algorithms using hpx:: syntax on cuda enabled resources.


See the :ref:`API reference <libs_cuda_support_api>` of this module for more
details.

