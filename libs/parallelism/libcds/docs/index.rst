..
    Copyright (c) 2020 Weile Wei
    Copyright (c) 2020 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _libs_libcds:

======
libcds
======

LibCDS Overview
---------------

`LibCDS <https://github.com/khizmax/libcds>`_ implements a collection of
concurrent containers that don't require external (manual) synchronization
for shared access, and safe memory reclamation (SMR) algorithms like
Hazard Pointer and user-space RCU that is used as an epoch-based SMR.

Supporting features
-------------------
Currently, this module **only** supports *Hazard Pointer*
(a type of safe memory reclamation schemas/garbage collectors)
based containers in LibCDS
at |hpx| light-weight user-level thread. In the future, we might consider
adding support to other containers, including dynamic Hazard Pointer, RCU, etc,
at |hpx| user-level thread,
and might also explore suitable containers (i.e. flat-combining queue) in |hpx|
runtime scheduler.

To find out which container supports Hazard Pointer based garbage collector,
one might want to check
`LibCDS Documentation <http://libcds.sourceforge.net/doc/cds-api/index.html>`_.
For example, after clicking `cds > Modules > FeldmanHashMap <http://libcds.sourceforge.net/doc/cds-api/classcds_1_1container_1_1_feldman_hash_map.html>`_
, one can find Template parameters in FeldmanHashMap class
:cpp:class:`cds::container::FeldmanHashMap< GC, Key, T, Traits >`
suggests *GC - safe memory reclamation schema. Can be gc::HP, gc::DHP or one of RCU type`*
This means :cpp:type:`FeldmanHashMap` can be safely used with Hazard Pointer GC. However,
again, current |hpx| does not support *gc:DHP or RCU*, so we cannot use these two types of garbage collectors.

Build Hazard Pointer with |hpx| threads
-------------------------------------
To build and your own lock free container in LibCDS using
|hpx| light-weight user-level thread, one might first get familiar with
`LibCDS <https://github.com/khizmax/libcds>`_ itself. The simplest way to
launch Hazard Pointer in |hpx| threads is to do the following:

.. code-block:: c++

    #include <hpx/hpx_init.hpp>
    #include <cds/init.h>       // for cds::Initialize and cds::Terminate
    #include <cds/gc/hp.h>      // for cds::HP (Hazard Pointer) SMR

    int hpx_main(int, char**)
    {
        // Initialize libcds (hazard pointer type by default)
        hpx::cds::libcds_wrapper cds_init_wrapper;

        {
            // If main thread uses lock-free containers
            // the main thread should be attached to libcds infrastructure
            hpx::cds::hpxthread_manager_wrapper cdswrap;

            // Now you can use HP-based containers in the main thread
            //...
        }

        return hpx::finalize();
    }

    int main(int argc, char* argv[])
    {
        return hpx::init(argc, argv);
    }

Use Hazard Pointer supported Container w/ |hpx| threads or std::threads
---------------------------------------------------------------------

To note, to use Hazard Pointer in the context of |hpx| user-level threads,
one must use construct libcds object with
:cpp:type:`hpx::cds::libcds_wrapper cds_init_wrapper
(hpx::cds::smr_t::hazard_pointer_hpxthread)`. This ensures
hazard pointer is used as well as the thread data is bound
to user-level thread.
If one wants to use use default kernel thread and thus keep thread-private data
at kernel-level threads, the following should be used to create Hazard Pointer SMR
:cpp:type:`hpx::cds::libcds_wrapper cds_init_wrapper
(hpx::cds::smr_t::hazard_pointer_stdthread)`.

To use any Hazard Pointer supported container, one also needs to populate TLS type
to all levels of the container.
One simplest map is :cpp:type:`FeldmanHashMap`:

.. code-block:: c++

    using gc_type = cds::gc::custom_HP<cds::gc::hp::details::HPXDataHolder>;
    using key_type = std::size_t;
    using value_type = std::string;
    using map_type =
    cds::container::FeldmanHashMap<gc_type, key_type, value_type>;

A more complex map example can be found in `libcds_michael_map_hazard_pointer.cpp`,
where the map is built on top of a list. In this case, both map and list need to
use :cpp:type:`cds::gc::hp::details::HPXDataHolder` to template the Garbage Collector
type.

API
-----------------------------------------------------

The following API functions are exposed:

- :cpp:func:`hpx::cds::libcds_wrapper(smr_t smr_type = smr_t::hazard_pointer_hpxthread,
            std::size_t hazard_pointer_count = 1,
            std::size_t max_thread_count = std::stoul(hpx::get_config_entry(
                "hpx.cds.num_concurrent_hazard_pointer_threads", "128")),
            std::size_t max_retired_pointer_count = 16)`: This is a wrapper of
:cpp:func:`cds::Initialize()` and :cpp:func:`cds::Terminate()` as well as
supported SMR type. This allows initializing libcds infrastructure
(and destroying it after the object's lifetime).

To initialize different SMR for libcds,
one can pass different :cpp:func:`hpx::cds::smr_t::*` to libcds_wrapper constructor.
Current supported smr_t has :cpp:type:`hazard_pointer_hpxthread` (default),
:cpp:type:`hazard_pointer_stdthread`, :cpp:type:`rcu` (experimental).

To update max concurrent attached thread to Hazard Pointer SMR in HPX thread, one can
pass different command line values, for example:
:cpp:type:`--hpx:ini=hpx.cds.num_concurrent_hazard_pointer_threads=256`

To understand max_retired_pointer_count and the above mentioned variables
, more reference can be found in
`HP in LibCDS <https://github.com/khizmax/libcds/blob/master/cds/gc/hp.h>`_.

- :cpp:func:`hpx::cds::hpxthread_manager_wrapper`: This is a wrapper of
:cpp:func:`cds::gc::hp::custom_smr<cds::gc::hp::details::HPXDataHolder>::attach_thread()`
and :cpp:func:`cds::gc::hp::custom_smr<cds::gc::hp::details::HPXDataHolder>::detach_thread()`
This allows the calling |hpx| thread attach to Hazard Pointer threading infrastructure.


See the :ref:`API reference <libs_libcds_api>` of this module for more
details.

