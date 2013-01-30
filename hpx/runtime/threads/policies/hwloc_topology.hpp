////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c)      2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_50DFC0FC_EE99_43F5_A918_01EC45A58036)
#define HPX_50DFC0FC_EE99_43F5_A918_01EC45A58036

#include <hwloc.h>

#include <boost/format.hpp>

#include <hpx/runtime/threads/topology.hpp>
#include <hpx/exception.hpp>

#include <hpx/util/spinlock.hpp>

namespace hpx { namespace threads
{

struct hwloc_topology : topology
{
    hwloc_topology()
      : topo(0), machine_affinity_mask_(0)
    { // {{{
        std::size_t const num_of_cores = hardware_concurrency();

        int err = hwloc_topology_init(&topo);
        if (err != 0)
        {
            HPX_THROW_EXCEPTION(no_success, "hwloc_topology::hwloc_topology", 
                "Failed to init hwloc hwloc_topology");
        }

        err = hwloc_topology_load(topo);
        if (err != 0)
        {
            HPX_THROW_EXCEPTION(no_success, "hwloc_topology::hwloc_topology", 
                "Failed to load hwloc topology");
        }

        numa_node_numbers_.reserve(num_of_cores);
        core_numbers_.reserve(num_of_cores);
        numa_node_affinity_masks_.reserve(num_of_cores);
        core_affinity_masks_.reserve(num_of_cores);
        thread_affinity_masks_.reserve(num_of_cores);
        ns_thread_affinity_masks_.reserve(num_of_cores);

        // Initialize each set of data entirely, as some of the initialization
        // routines rely on access to other pieces of topology data. The
        // compiler will optimize the loops where possible anyways.

        machine_affinity_mask_ = init_machine_affinity_mask();

        for (std::size_t i = 0; i < num_of_cores; ++i)
            numa_node_numbers_.push_back(init_numa_node_number(i));

        for (std::size_t i = 0; i < num_of_cores; ++i)
            core_numbers_.push_back(init_core_number(i));

        for (std::size_t i = 0; i < num_of_cores; ++i)
        {
            numa_node_affinity_masks_.push_back(
                init_numa_node_affinity_mask(i));
        }

        for (std::size_t i = 0; i < num_of_cores; ++i)
            core_affinity_masks_.push_back(init_core_affinity_mask(i));

        for (std::size_t i = 0; i < num_of_cores; ++i)
        {
            thread_affinity_masks_.push_back(
                init_thread_affinity_mask(i, false));
            ns_thread_affinity_masks_.push_back(
                init_thread_affinity_mask(i, true));
        }
    } // }}}

    ~hwloc_topology()
    {
        if (topo)
            hwloc_topology_destroy(topo);
    }

    std::size_t get_numa_node_number(
        std::size_t num_thread
      , error_code& ec = throws
        ) const
    { // {{{
        if (num_thread < numa_node_numbers_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return numa_node_numbers_[num_thread];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_numa_node_number"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return std::size_t(-1);
    } // }}}

    std::size_t get_core_number(
        std::size_t num_thread
      , error_code& ec = throws
        ) const
    { // {{{
        if (num_thread < core_numbers_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return core_numbers_[num_thread];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_core_number"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return std::size_t(-1);
    } // }}}

    mask_type get_machine_affinity_mask(
        error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return machine_affinity_mask_;
    }

    mask_type get_numa_node_affinity_mask(
        std::size_t num_thread
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    { // {{{
        if (num_thread < numa_node_affinity_masks_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return numa_node_affinity_masks_[num_thread];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_numa_node_affinity_mask"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return 0;
    } // }}}

    mask_type get_core_affinity_mask(
        std::size_t num_thread
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    {
        if (num_thread < core_affinity_masks_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return core_affinity_masks_[num_thread];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_core_affinity_mask"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return 0;
    }

    mask_type get_thread_affinity_mask(
        std::size_t num_thread
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    { // {{{
        if (num_thread < thread_affinity_masks_.size())
        {
            if (&ec != &throws)
                ec = make_success_code();

            return numa_sensitive ? ns_thread_affinity_masks_[num_thread]
                                  : thread_affinity_masks_[num_thread];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::hwloc_topology::get_thread_affinity_mask"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return 0;
    } // }}}

    void set_thread_affinity_mask(
        boost::thread&
      , mask_type //mask
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();
    }

    void set_thread_affinity_mask(
        mask_type mask
      , error_code& ec = throws
        ) const
    { // {{{
        // Figure out how many cores are available.
        // Now set the affinity to the required PU.

        hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();

// #if HPX_DEBUG
//         hwloc_bitmap_singlify(cpuset);
//         hwloc_cpuset_t cpuset_cmp = hwloc_bitmap_alloc();
//         if (0 == hwloc_get_cpubind(topo, cpuset_cmp, HWLOC_CPUBIND_THREAD)) {
//             hwloc_bitmap_singlify(cpuset_cmp);
//             BOOST_ASSERT(hwloc_bitmap_compare(cpuset, cpuset_cmp) != 0);
//         }
//         hwloc_bitmap_free(cpuset_cmp);
// #endif

        hwloc_bitmap_from_ith_ulong(cpuset, 1, (mask >> 32) & 0xFFFFFFFF); //-V112
        hwloc_bitmap_from_ith_ulong(cpuset, 0, mask & 0xFFFFFFFF); //-V112

//         std::size_t idx = 0;
//         for (std::size_t i = 0; i < sizeof(std::size_t) * CHAR_BIT; ++i)
//         {
//             if (mask & (static_cast<std::size_t>(1) << i))
//             {
//                 idx = i;
//             }
//         }
//
//         hwloc_bitmap_only(cpuset, static_cast<unsigned int>(idx));

//         hwloc_bitmap_singlify(cpuset);
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
// #if HPX_DEBUG
//         cpuset_cmp = hwloc_bitmap_alloc();
//         if (0 == hwloc_get_cpubind(topo, cpuset_cmp, HWLOC_CPUBIND_THREAD)) {
//             hwloc_bitmap_singlify(cpuset_cmp);
//             BOOST_ASSERT(hwloc_bitmap_compare(cpuset, cpuset_cmp) == 0);
//         }
//         hwloc_bitmap_free(cpuset_cmp);
// #endif

        hwloc_bitmap_free(cpuset);

        if (&ec != &throws)
            ec = make_success_code();

    } // }}}

    mask_type get_thread_affinity_mask_from_lva(
        naming::address::address_type
      , error_code& ec = throws
        ) const
    { // {{{
        if (&ec != &throws)
            ec = make_success_code();

        return 0;
    } // }}}

  private:
    std::size_t init_numa_node_number(
        std::size_t num_thread
        )
    { // {{{
        if (std::size_t(-1) == num_thread)
            return std::size_t(-1);

        {
            hwloc_obj_t obj;
            {
                scoped_lock lk(topo_mtx);
                obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU,
                    static_cast<unsigned>(num_thread));
            }
            while (obj)
            {
                if (hwloc_compare_types(obj->type, HWLOC_OBJ_NODE) == 0)
                    return static_cast<std::size_t>(obj->os_index);
                obj = obj->parent;
            }
        }

        return std::size_t(-1);
    } // }}}

    std::size_t init_core_number(
        std::size_t num_thread
        )
    { // {{{
        if (std::size_t(-1) == num_thread)
            return std::size_t(-1);

        {
            hwloc_obj_t obj;
            {
                scoped_lock lk(topo_mtx);
                obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU,
                    static_cast<unsigned>(num_thread));
            }
            while (obj)
            {
                if (hwloc_compare_types(obj->type, HWLOC_OBJ_CORE) == 0)
                    return static_cast<std::size_t>(obj->os_index);
                obj = obj->parent;
            }
        }

        return std::size_t(-1);
    } // }}}

    void extract_node_mask(
        hwloc_obj_t parent
      , mask_type& mask
        )
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

    mask_type init_machine_affinity_mask()
    { // {{{
        mask_type node_affinity_mask = 0;

        hwloc_obj_t numa_node_obj;
        {
            scoped_lock lk(topo_mtx);
            numa_node_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_MACHINE, 0);
        }
        if (numa_node_obj)
        {
            extract_node_mask(numa_node_obj, node_affinity_mask);
            return node_affinity_mask;
        }

        HPX_THROW_EXCEPTION(kernel_error
          , "hpx::threads::hwloc_topology::init_machine_affinity_mask"
          , "failed to initialize machine affinity mask");
        return 0;
    } // }}}

    mask_type init_numa_node_affinity_mask(
        std::size_t num_thread
        )
    { // {{{
        mask_type node_affinity_mask = 0;
        std::size_t numa_node = get_numa_node_number(num_thread);

        // If we have only one or no numa domain, the numa affinity mask spans
        // all processors
        if (std::size_t(-1) == numa_node)
        {
            for(std::size_t i = 0; i < core_numbers_.size(); ++i)
            {
                node_affinity_mask |= 1 << i;
            }
            return node_affinity_mask;
        }
        hwloc_obj_t numa_node_obj;
        {
            scoped_lock lk(topo_mtx);
            numa_node_obj = hwloc_get_obj_by_type(topo,
                HWLOC_OBJ_NODE, static_cast<unsigned>(numa_node));
        }
        if (numa_node_obj)
        {
            extract_node_mask(numa_node_obj, node_affinity_mask);
            return node_affinity_mask;
        }

        HPX_THROW_EXCEPTION(kernel_error
          , "hpx::threads::hwloc_topology::init_numa_node_affinity_mask"
          , boost::str(boost::format(
                "failed to initialize NUMA node affinity mask for thread %1%")
                % num_thread));
        return 0;
    } // }}}

    mask_type init_core_affinity_mask(
        std::size_t num_thread
        )
    { // {{{
        mask_type node_affinity_mask = 0;
        std::size_t core = get_core_number(num_thread);

        if (std::size_t(-1) == core)
            return 0;
        hwloc_obj_t core_obj;
        {
            scoped_lock lk(topo_mtx);
            core_obj = hwloc_get_obj_by_type(topo,
                HWLOC_OBJ_CORE, static_cast<unsigned>(core));
        }
        if (core_obj)
        {
            extract_node_mask(core_obj, node_affinity_mask);
            return node_affinity_mask;

        }

        HPX_THROW_EXCEPTION(kernel_error
          , "hpx::threads::hwloc_topology::init_core_affinity_mask"
          , boost::str(boost::format(
                "failed to initialize core affinity mask for thread %1%")
                % num_thread));
        return 0;
    } // }}}

    mask_type init_thread_affinity_mask(
        std::size_t num_thread
      , bool numa_sensitive
        )
    { // {{{
        std::size_t num_of_cores = hardware_concurrency();
        mask_type affinity = num_thread % num_of_cores;

        {
            std::size_t numa_nodes = 0;
            int numa_nodes_int;
            {
                scoped_lock lk(topo_mtx);
                numa_nodes_int = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NODE);
            }
            if(numa_nodes_int == -1) return 0;
            numa_nodes = static_cast<std::size_t>(numa_nodes_int);

            if (numa_nodes == 0)
            {
                numa_nodes = 1;
            }

            std::size_t num_of_cores_per_numa_node = num_of_cores / numa_nodes;
            mask_type mask = 0;

            mask_type node_affinity_mask = 
                get_numa_node_affinity_mask(num_thread, numa_sensitive);

            std::size_t node_index = affinity % num_of_cores_per_numa_node;

            // We need to detect the node_index-th bit which is set in
            // node_affinity_mask, this bit must be set in the result mask.
            std::size_t count = 0;
            for (std::size_t i = 0; i < sizeof(std::size_t) * CHAR_BIT; ++i)
            {
                // Is the i-th bit set?
                if (node_affinity_mask & (static_cast<mask_type>(1) << i))
                {
                    if (count == node_index)
                    {
                        mask = (static_cast<mask_type>(1) << i);
                        break;
                    }
                    ++count;
                }
            }

            return mask;
        }

        return 0;
    } // }}}

    hwloc_topology_t topo;

    mutable hpx::util::spinlock topo_mtx;
    typedef hpx::util::spinlock::scoped_lock scoped_lock;

    std::vector<std::size_t> numa_node_numbers_;
    std::vector<std::size_t> core_numbers_;

    mask_type machine_affinity_mask_;
    std::vector<mask_type> numa_node_affinity_masks_;
    std::vector<mask_type> core_affinity_masks_;
    std::vector<mask_type> thread_affinity_masks_;
    std::vector<mask_type> ns_thread_affinity_masks_;
};

///////////////////////////////////////////////////////////////////////////////
inline topology* create_topology()
{
    return new hwloc_topology;
}

}}

#endif // HPX_50DFC0FC_EE99_43F5_A918_01EC45A58036

