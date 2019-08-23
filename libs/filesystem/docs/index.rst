..
   Copyright (c) 2019 The STE||AR-Group

   Distributed under the Boost Software License, Version 1.0. (See accompanying
   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _libs_filesystem:

==========
filesystem
==========

This module provides a compatibility layer for the C++17 filesystem library. If
the filesystem library is available this module will simply forward its contents
into the ``hpx::filesystem`` namespace. If the library is not available it will
fall back to Boost.Filesystem instead.
