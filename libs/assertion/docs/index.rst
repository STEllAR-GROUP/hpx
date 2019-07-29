..
    Copyright (c) 2018 The STE||AR-Group

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _libs_assertion:

=========
assertion
=========

The assertion library implements the macros :c:macro:`HPX_ASSERT` and
:c:macro:`HPX_ASSERT_MSG`. Those two macros can be used to implement assertions
which are turned of during a release build.

By default, the location and function where the assert has been called from are
displayed when the assertion fires. This behavior can be modified by using
:cpp:func:`hpx::assertion::set_assertion_handler`. When HPX initializes, it uses
this function to specify a more elaborate assertion handler. If your application
needs to customize this, it needs to do so before calling
:cpp:func:`hpx::hpx_init`, :cpp:func:`hpx::hpx_main` or using the C-main
wrappers.

