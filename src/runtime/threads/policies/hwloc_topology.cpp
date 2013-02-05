//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2012-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_HWLOC)

#include <hpx/exception.hpp>
#include <hpx/util/spinlock.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/policies/hwloc_topology.hpp>

#include <hwloc.h>

#include <boost/format.hpp>

namespace hpx { namespace threads
{
    hwloc_topology::hwloc_topology()
      : topo(0), machine_affinity_mask_(0)
    { // {{{
        int err = hwloc_topology_init(&topo);
        if (err != 0)
        {
            HPX_THROW_EXCEPTION(no_success, "hwloc_topology::hwloc_topology",
                "Failed to init hwloc topology");
        }

        err = hwloc_topology_load(topo);
        if (err != 0)
        {
            HPX_THROW_EXCEPTION(no_success, "hwloc_topology::hwloc_topology",
                "Failed to load hwloc topology");
        }

        init_num_of_pus();

        socket_numbers_.reserve(num_of_pus_);
        numa_node_numbers_.reserve(num_of_pus_);
        core_numbers_.reserve(num_of_pus_);

        // Initialize each set of data entirely, as some of the initialization
        // routines rely on access to other pieces of topology data. The
        // compiler will optimize the loops where possible anyways.

        std::size_t num_of_sockets = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_SOCKET);
        if(num_of_sockets == 0) num_of_sockets = 1;
        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            std::size_t socket = init_socket_number(i);
            BOOST_ASSERT(socket < num_of_sockets);
            socket_numbers_.push_back(socket);
        }

        std::size_t num_of_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NODE);
        if(num_of_nodes == 0) num_of_nodes = 1;
        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            std::size_t numa_node = init_numa_node_number(i);
            BOOST_ASSERT(numa_node < num_of_nodes);
            numa_node_numbers_.push_back(numa_node);
        }

        std::size_t num_of_cores = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE);
        if(num_of_cores == 0) num_of_cores = 1;
        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            std::size_t core_number = init_core_number(i);
            BOOST_ASSERT(core_number < num_of_cores);
            core_numbers_.push_back(core_number);
        }

        machine_affinity_mask_ = init_machine_affinity_mask();
        socket_affinity_masks_.reserve(num_of_pus_);
        numa_node_affinity_masks_.reserve(num_of_pus_);
        core_affinity_masks_.reserve(num_of_pus_);
        thread_affinity_masks_.reserve(num_of_pus_);

        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            socket_affinity_masks_.push_back(init_socket_affinity_mask(i));
        }

        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            numa_node_affinity_masks_.push_back(init_numa_node_affinity_mask(i));
        }

        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            core_affinity_masks_.push_back(init_core_affinity_mask(i));
        }

        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            thread_affinity_masks_.push_back(init_thread_affinity_mask(i));
        }
    } // }}}

    hwloc_topology::~hwloc_topology()
    {
        if (topo)
            hwloc_topology_destroy(topo);
    }

    std::size_t hwloc_topology::get_socket_number(
        std::size_t num_thread
      , error_code& ec
        ) const
    { // {{{
        std::size_t num_pu = num_thread % num_of_pus_;

        if (num_pu < socket_numbers_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return socket_numbers_[num_pu];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_socket_number"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return std::size_t(-1);
    } // }}}

    std::size_t hwloc_topology::get_numa_node_number(
        std::size_t num_thread
      , error_code& ec
        ) const
    { // {{{
        std::size_t num_pu = num_thread % num_of_pus_;

        if (num_pu < numa_node_numbers_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return numa_node_numbers_[num_pu];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_numa_node_number"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return std::size_t(-1);
    } // }}}

    std::size_t hwloc_topology::get_core_number(
        std::size_t num_thread
      , error_code& ec
        ) const
    { // {{{
        std::size_t num_pu = num_thread % num_of_pus_;

        if (num_pu < core_numbers_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return core_numbers_[num_pu];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_core_number"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return std::size_t(-1);
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    mask_type hwloc_topology::get_machine_affinity_mask(
        error_code& ec
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return machine_affinity_mask_;
    }

    mask_type hwloc_topology::get_socket_affinity_mask(
        std::size_t num_thread
      , bool numa_sensitive
      , error_code& ec
        ) const
    { // {{{
        std::size_t num_pu = num_thread % num_of_pus_;

        if (num_pu < socket_affinity_masks_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return socket_affinity_masks_[num_pu];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_socket_affinity_mask"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return 0;
    } // }}}

    mask_type hwloc_topology::get_numa_node_affinity_mask(
        std::size_t num_thread
      , bool numa_sensitive
      , error_code& ec
        ) const
    { // {{{
        std::size_t num_pu = num_thread % num_of_pus_;

        if (num_pu < numa_node_affinity_masks_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return numa_node_affinity_masks_[num_pu];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_numa_node_affinity_mask"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return 0;
    } // }}}

    mask_type hwloc_topology::get_core_affinity_mask(
        std::size_t num_thread
      , bool numa_sensitive
      , error_code& ec
        ) const
    {
        std::size_t num_pu = num_thread % num_of_pus_;

        if (num_pu < core_affinity_masks_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return core_affinity_masks_[num_pu];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_core_affinity_mask"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return 0;
    }

    mask_type hwloc_topology::get_thread_affinity_mask(
        std::size_t num_thread
      , bool numa_sensitive
      , error_code& ec
        ) const
    { // {{{
        std::size_t num_pu = num_thread % num_of_pus_;

        if (num_pu < thread_affinity_masks_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return thread_affinity_masks_[num_pu];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_thread_affinity_mask"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return 0;
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    void hwloc_topology::set_thread_affinity_mask(
        boost::thread&
      , mask_type //mask
      , error_code& ec
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();
    }

    void hwloc_topology::set_thread_affinity_mask(
        mask_type mask
      , error_code& ec
        ) const
    { // {{{
        hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();

        for (std::size_t i = 0; i < sizeof(std::size_t) * CHAR_BIT; ++i)
        {
            if (mask & (static_cast<std::size_t>(1) << i))
            {
                hwloc_bitmap_set(cpuset, static_cast<unsigned int>(i));
            }
        }

        {
            scoped_lock lk(topo_mtx);
            if (hwloc_set_cpubind(topo, cpuset,
                  HWLOC_CPUBIND_STRICT | HWLOC_CPUBIND_THREAD))
            {
                // Strict binding not supported or failed, try weak binding.
                if (hwloc_set_cpubind(topo, cpuset, HWLOC_CPUBIND_THREAD))
                {
                    hwloc_bitmap_free(cpuset);

                    HPX_THROWS_IF(ec, kernel_error
                      , "hpx::threads::hwloc_topology::set_thread_affinity_mask"
                      , boost::str(boost::format(
                            "failed to set thread %x affinity mask")
                            % mask));

                    if (ec)
                        return;
                }
            }
        }
#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
        sleep(0);   // Allow the OS to pick up the change.
#endif

        hwloc_bitmap_free(cpuset);

        if (&ec != &throws)
            ec = make_success_code();
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    mask_type hwloc_topology::get_thread_affinity_mask_from_lva(
        naming::address::address_type
      , error_code& ec
        ) const
    { // {{{
        if (&ec != &throws)
            ec = make_success_code();

        return 0;
    } // }}}

    std::size_t hwloc_topology::init_node_number(
        std::size_t num_thread, hwloc_obj_type_t type
        )
    { // {{{
        if (std::size_t(-1) == num_thread)
            return std::size_t(-1);

        std::size_t num_pu = num_thread % num_of_pus_;

        {
            hwloc_obj_t obj;

            {
                scoped_lock lk(topo_mtx);
                obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU,
                    static_cast<unsigned>(num_pu));
            }

            while (obj)
            {
                if (hwloc_compare_types(obj->type, type) == 0)
                {
                    if (obj->os_index != ~0x0u)
                        return static_cast<std::size_t>(obj->os_index);

                    // on Windows os_index is always -1
                    return static_cast<std::size_t>(obj->logical_index);
                }
                obj = obj->parent;
            }
        }
        return 0;
    } // }}}

    void hwloc_topology::extract_node_mask(
        hwloc_obj_t parent
      , mask_type& mask
        ) const
    { // {{{
        hwloc_obj_t obj;

        {
            scoped_lock lk(topo_mtx);
            obj = hwloc_get_next_child(topo, parent, NULL);
        }

        while (obj)
        {
            if (hwloc_compare_types(HWLOC_OBJ_PU, obj->type) == 0)
            {
                do {
                    mask |= (static_cast<mask_type>(1) << obj->os_index);
                    {
                        scoped_lock lk(topo_mtx);
                        obj = hwloc_get_next_child(topo, parent, obj);
                    }
                } while (obj != NULL &&
                         hwloc_compare_types(HWLOC_OBJ_PU, obj->type) == 0);
                return;
            }

            extract_node_mask(obj, mask);

            scoped_lock lk(topo_mtx);
            obj = hwloc_get_next_child(topo, parent, obj);
        }
    } // }}}

    mask_type hwloc_topology::init_machine_affinity_mask() const
    { // {{{
        mask_type machine_affinity_mask = 0;

        hwloc_obj_t machine_obj;
        {
            scoped_lock lk(topo_mtx);
            machine_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_MACHINE, 0);
        }
        if (machine_obj)
        {
            extract_node_mask(machine_obj, machine_affinity_mask);
            return machine_affinity_mask;
        }

        HPX_THROW_EXCEPTION(kernel_error
          , "hpx::threads::hwloc_topology::init_machine_affinity_mask"
          , "failed to initialize machine affinity mask");
        return 0;
    } // }}}

    mask_type hwloc_topology::init_socket_affinity_mask_from_socket(
        std::size_t num_socket
        ) const
    { // {{{
        // If we have only one or no socket, the socket affinity mask
        // spans all processors
        if (std::size_t(-1) == num_socket)
            return machine_affinity_mask_;

        hwloc_obj_t socket_obj;

        {
            scoped_lock lk(topo_mtx);
            socket_obj = hwloc_get_obj_by_type(topo,
                HWLOC_OBJ_SOCKET, static_cast<unsigned>(num_socket));
        }

        if (socket_obj)
        {
            mask_type socket_affinity_mask = 0;
            extract_node_mask(socket_obj, socket_affinity_mask);
            return socket_affinity_mask;
        }

        return machine_affinity_mask_;
    } // }}}

    mask_type hwloc_topology::init_numa_node_affinity_mask_from_numa_node(
        std::size_t numa_node
        ) const
    { // {{{
        // If we have only one or no NUMA domain, the NUMA affinity mask
        // spans all processors
        if (std::size_t(-1) == numa_node)
        {
            return machine_affinity_mask_;
        }

        hwloc_obj_t numa_node_obj;

        {
            scoped_lock lk(topo_mtx);
            numa_node_obj = hwloc_get_obj_by_type(topo,
                HWLOC_OBJ_NODE, static_cast<unsigned>(numa_node));
        }

        if (numa_node_obj)
        {
            mask_type node_affinity_mask = 0;
            extract_node_mask(numa_node_obj, node_affinity_mask);
            return node_affinity_mask;
        }

        return machine_affinity_mask_;
    } // }}}

    mask_type hwloc_topology::init_core_affinity_mask_from_core(
        std::size_t core, mask_type default_mask
        ) const
    { // {{{
        if (std::size_t(-1) == core)
            return default_mask;

        hwloc_obj_t core_obj;

        {
            scoped_lock lk(topo_mtx);
            core_obj = hwloc_get_obj_by_type(topo,
                HWLOC_OBJ_CORE, static_cast<unsigned>(core));
        }

        if (core_obj)
        {
            mask_type core_affinity_mask = 0;
            extract_node_mask(core_obj, core_affinity_mask);
            return core_affinity_mask;
        }

        return default_mask;
    } // }}}

    mask_type hwloc_topology::init_thread_affinity_mask(
        std::size_t num_thread
        ) const
    { // {{{

        if (std::size_t(-1) == num_thread)
        {
            return get_core_affinity_mask(num_thread, false);
        }

        std::size_t num_pu = num_thread % num_of_pus_;

        hwloc_obj_t obj;

        {
            scoped_lock lk(topo_mtx);
            obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU,
                    static_cast<unsigned>(num_pu));
        }

        if (!obj)
        {
            return get_core_affinity_mask(num_thread, false);
        }

        mask_type mask = 0x0u;

        mask |= (static_cast<mask_type>(1) << obj->os_index);

        return mask;
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    void hwloc_topology::init_num_of_pus()
    {
        num_of_pus_ = 1;
        {
            scoped_lock lk(topo_mtx);
            int num_of_pus = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_PU);

            if (num_of_pus > 0)
                num_of_pus_ = static_cast<std::size_t>(num_of_pus);
        }
    }
}}

#endif
