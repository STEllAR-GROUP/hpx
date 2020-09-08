..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_serialization:

=============
serialization
=============

This module provides serialization primitives and support for all built-in
types as well as all C++ Standard Library collection and utility types. This
list is extended by |hpx| vocabulary types with proper support for global
reference counting. |hpx|'s mode of serialization is derived from `Boost's
serialization model
<https://www.boost.org/doc/libs/1_72_0/libs/serialization/doc/index.html>`_
and, as such, is mostly interface compatible with
its Boost counterpart.

The purest form of serializing data is to copy the content of the payload bit
by bit; however, this method is impractical for generic C++ types, which might
be composed of more than just regular built-in types. Instead, |hpx|'s approach
to serialization is derived from the Boost Serialization library, and is geared
towards allowing the programmer of a given class explicit control and syntax of
what to serialize. It is based on operator overloading of two special archive
types that hold a buffer or stream to store the serialized data and is
responsible for dispatching the serialization mechanism to the intrusive or
non-intrusive version. The serialization process is recursive. Each member that
needs to be serialized must be specified explicitly. The advantage of this
approach is that the serialization code is written in C++ and leverages all
necessary programming techniques. The generic, user-facing interface allows
for effective application of the serialization process without obstructing the
algorithms that need special code for packing and unpacking. It also allows for
optimizations in the implementation of the archives.

See the :ref:`API reference <modules_serialization_api>` of the module for more
details.
