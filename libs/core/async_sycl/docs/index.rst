..
    Copyright (c) 2022 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_async_sycl:

==========
async_sycl
==========

This module allows creating HPX futures using |sycl|_ events, effectively integrating asynchronous SYCL kernels and
memory transfers with HPX. Building on this integration, this module also contains a SYCL executor. This executor
encapsulates a SYCL queue. When SYCL queue member functions are launched with this executor, the user can automatically
obtain the HPX futures associated with them.

The creation of the HPX futures using SYCL events is based on the same event polling mechanism that the CUDA HPX
integration uses. Each registered event gets an associated callback and gets inserted into a callback vector to
be polled by the scheduler in between tasks. Once the polling reveals the event is complete, the callback will be
called, which in turn sets the future to ready (see sycl_event_callback.cpp).
There are multiple adaptions for HipSYCL for this: To keep the runtime alive (avoiding the repeated on-the-fly creation of
the runtime during the polling), we keep a default queue. Furthermore, we flush the internal SYCL DAG to ensure
that the launched SYCL function is actually being executed.

The SYCL executor offers the usual post and async_execute functions. Additionally, it contains two get_future functions.
One expects a pre-existing SYCL event to return a future, the other one does not but will launch an empty SYCL kernel
instead, to obtain an event (causing higher overhead for the sake of being more convenient).
The post and async_execute implementations here are actually different for HipSYCL and OneAPI, since the sycl::queue
in OneAPI uses a different interface (using a code_location parameter) which requires some adaptations here.

To make this module compile, we use the -fno-sycl and -fsycl compiler parameters for the OneAPI use-case (requiring
HPX to be compiled with dpcpp). For HipSYCL we use its cmake integration instead (requiring HPX to be compiled with
clang++ and including HipSYCL as a library).

To build with OneAPI, use the CMake Variable HPX_WITH_SYCL=ON.
To build with HipSYCL, use HPX_WITH_SYCL=ON and HPX_WITH_HIPSYCL=ON (and make sure find_package will find HipSYCL).

Lastly, the module contains three tests/examples. All three implement a simple vector add example. The first one
obtains a future using the free method get_future, the second one uses a single SYCL executor and the last one
is using multiple executors called from multiple host threads.

To build the tests, use " make tests.unit.modules.async_sycl "
To run the tests, use "ctest -R sycl".

NOTE: Theoretically, this module could work with other SYCL implementations, but was only tested using OneAPI and HipSYCL
so far.

See the :ref:`API reference <modules_async_sycl_api>` of this module for more
details.

