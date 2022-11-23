..
    Copyright (c) 2022 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_async_sycl:

==========
async_sycl
==========

This module allows creating HPX futures using SYCL events, effectively integrating asynchronous SYCL kernels and
memory transfers with HPX. Building on this integration, this module also contains a SYCL executor. This executor
encapsulates a SYCL queue. When SYCL queue member functions are launched with this executor, the user can automatically
obtain the HPX futures associated with them.

See the :ref:`API reference <modules_async_sycl_api>` of this module for more
details.

