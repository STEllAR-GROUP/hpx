..
    Copyright (c) 2020-2022 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_async_combinators:

=================
async_combinators
=================

This module contains combinators for futures. The ``when_*`` functions allow you
to turn multiple futures into a single future which is ready when all, any,
some, or each of the given futures are ready. The ``wait_*`` combinators are
equivalent to the ``when_*`` functions except that they do not return a future.
Those wait for all futures to become ready before returning to the user. Note
that the ``wait_*`` functions will rethrow one of the exceptions from
exceptional futures.
The ``wait_*_nothrow`` combinators are equivalent to the ``wait_*`` functions
exception that they do not throw if one of the futures has become exceptional.

The ``split_future`` combinator takes a single future of a container (e.g.
``tuple``) and turns it into a container of futures.

See :ref:`modules_lcos_local`, :ref:`modules_synchronization`, and
:ref:`modules_async_distributed` for other synchronization facilities.

See the :ref:`API reference <modules_async_combinators_api>` of this module for more
details.

