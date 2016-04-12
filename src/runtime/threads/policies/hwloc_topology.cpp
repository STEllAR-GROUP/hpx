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
#include <boost/scoped_ptr.hpp>
#include <boost/io/ios_state.hpp>

#include <vector>

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        void write_to_log(char const* valuename, std::size_t value)
        {
            LTM_(debug) << "hwloc_topology: " << valuename << ": " << value; //-V128
        }

        void write_to_log_mask(char const* valuename, mask_cref_type value)
        {
            LTM_(debug) << "hwloc_topology: " << valuename
                        << ": " HPX_CPU_MASK_PREFIX
                        << std::hex << value;
        }

        void write_to_log(char const* valuename,
            std::vector<std::size_t> const& values)
        {
            LTM_(debug) << "hwloc_topology: " << valuename << "s, size: " //-V128
                        << values.size();

            std::size_t i = 0;
            for (std::size_t value : values)
            {
                LTM_(debug) << "hwloc_topology: " << valuename //-V128
                            << "(" << i++ << "): " << value;
            }
        }

        void write_to_log_mask(char const* valuename,
            std::vector<mask_type> const& values)
        {
            LTM_(debug) << "hwloc_topology: " << valuename << "s, size: " //-V128
                        << values.size();

            std::size_t i = 0;
            for (mask_cref_type value : values)
            {
                LTM_(debug) << "hwloc_topology: " << valuename //-V128
                            << "(" << i++ << "): " HPX_CPU_MASK_PREFIX
                            << std::hex << value;
            }
        }

        std::size_t get_index(hwloc_obj_t obj)
        {
            // on Windows logical_index is always -1
            if (obj->logical_index == ~0x0u)
                return static_cast<std::size_t>(obj->os_index);

            return static_cast<std::size_t>(obj->logical_index);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    mask_type hwloc_topology::empty_mask = mask_type();

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

        std::size_t num_of_sockets = get_number_of_sockets();
        if (num_of_sockets == 0) num_of_sockets = 1;

        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            std::size_t socket = init_socket_number(i);
            HPX_ASSERT(socket < num_of_sockets);
            socket_numbers_.push_back(socket);
        }

        std::size_t num_of_nodes = get_number_of_numa_nodes();
        if (num_of_nodes == 0) num_of_nodes = 1;

        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            std::size_t numa_node = init_numa_node_number(i);
            HPX_ASSERT(numa_node < num_of_nodes);
            numa_node_numbers_.push_back(numa_node);
        }

        std::size_t num_of_cores = get_number_of_cores();
        if (num_of_cores == 0) num_of_cores = 1;

        for (std::size_t i = 0; i < num_of_pus_; ++i)
        {
            std::size_t core_number = init_core_number(i);
            HPX_ASSERT(core_number < num_of_cores);
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

    void hwloc_topology::write_to_log() const
    {
        std::size_t num_of_sockets = get_number_of_sockets();
        if (num_of_sockets == 0) num_of_sockets = 1;
        detail::write_to_log("num_sockets", num_of_sockets);


        std::size_t num_of_nodes = get_number_of_numa_nodes();
        if (num_of_nodes == 0) num_of_nodes = 1;
        detail::write_to_log("num_of_nodes", num_of_nodes);

        std::size_t num_of_cores = get_number_of_cores();
        if (num_of_cores == 0) num_of_cores = 1;
        detail::write_to_log("num_of_cores", num_of_cores);

        detail::write_to_log("num_of_pus", num_of_pus_);

        detail::write_to_log("socket_number", socket_numbers_);
        detail::write_to_log("numa_node_number", numa_node_numbers_);
        detail::write_to_log("core_number", core_numbers_);

        detail::write_to_log_mask("machine_affinity_mask", machine_affinity_mask_);

        detail::write_to_log_mask("socket_affinity_mask", socket_affinity_masks_);
        detail::write_to_log_mask("numa_node_affinity_mask", numa_node_affinity_masks_);
        detail::write_to_log_mask("core_affinity_mask", core_affinity_masks_);
        detail::write_to_log_mask("thread_affinity_mask", thread_affinity_masks_);
    }

    hwloc_topology::~hwloc_topology()
    {
        if (topo)
            hwloc_topology_destroy(topo);
    }

    std::size_t hwloc_topology::get_pu_number(
        std::size_t num_core
      , std::size_t num_pu
      , error_code& ec
        ) const
    { // {{{
        scoped_lock lk(topo_mtx);

        int num_cores = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE);

        // If num_cores is smaller 0, we have an error, it should never be zero
        // either to avoid division by zero, we should always have at least one
        // core
        if(num_cores <= 0)
        {
            HPX_THROWS_IF(ec, no_success, "hwloc_topology::hwloc_get_nobjs_by_type",
                "Failed to get number of cores");
            return std::size_t(-1);
        }
        num_core %= num_cores; //-V101 //-V104 //-V107

        hwloc_obj_t core_obj;

        core_obj = hwloc_get_obj_by_type(topo,
            HWLOC_OBJ_CORE, static_cast<unsigned>(num_core));

        num_pu %= core_obj->arity; //-V101 //-V104

        return std::size_t(core_obj->children[num_pu]->logical_index);
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    mask_cref_type hwloc_topology::get_machine_affinity_mask(
        error_code& ec
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return machine_affinity_mask_;
    }

    mask_cref_type hwloc_topology::get_socket_affinity_mask(
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
        return empty_mask;
    } // }}}

    mask_cref_type hwloc_topology::get_numa_node_affinity_mask(
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
        return empty_mask;
    } // }}}

    mask_cref_type hwloc_topology::get_core_affinity_mask(
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
        return empty_mask;
    }

    mask_cref_type hwloc_topology::get_thread_affinity_mask(
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
        return empty_mask;
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    void hwloc_topology::set_thread_affinity_mask(
        boost::thread&
      , mask_cref_type //mask
      , error_code& ec
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();
    }

    void hwloc_topology::set_thread_affinity_mask(
        mask_cref_type mask
      , error_code& ec
        ) const
    { // {{{

#if !defined(__APPLE__)
        // setting thread affinities is not supported by OSX
        hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();

        for (std::size_t i = 0; i < mask_size(mask); ++i)
        {
            if (test(mask, i))
            {
                int const pu_depth =
                    hwloc_get_type_or_below_depth(topo, HWLOC_OBJ_PU);
                for (unsigned int j = 0; j != num_of_pus_; ++j)
                {
                    hwloc_obj_t const pu_obj =
                        hwloc_get_obj_by_depth(topo, pu_depth, j);
                    unsigned idx =
                        static_cast<unsigned>(detail::get_index(pu_obj));

                    if(idx == i)
                    {
                        hwloc_bitmap_set(cpuset,
                            static_cast<unsigned int>(pu_obj->os_index));
                        break;
                    }
                }
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
                    boost::scoped_ptr<char> buffer(new char [1024]);

                    hwloc_bitmap_snprintf(buffer.get(), 1024, cpuset);
                    hwloc_bitmap_free(cpuset);

                    HPX_THROWS_IF(ec, kernel_error
                      , "hpx::threads::hwloc_topology::set_thread_affinity_mask"
                      , boost::str(boost::format(
                            "failed to set thread affinity mask ("
                            HPX_CPU_MASK_PREFIX "%x) for cpuset %s")
                            % mask % buffer.get()));

                    if (ec)
                        return;
                }
            }
        }
#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
        sleep(0);   // Allow the OS to pick up the change.
#endif
        hwloc_bitmap_free(cpuset);
#endif  // __APPLE__

        if (&ec != &throws)
            ec = make_success_code();
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    mask_type hwloc_topology::get_thread_affinity_mask_from_lva(
        naming::address::address_type lva
      , error_code& ec
        ) const
    { // {{{
        if (&ec != &throws)
            ec = make_success_code();

        hwloc_membind_policy_t policy = HWLOC_MEMBIND_DEFAULT;
        hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();

        {
            scoped_lock lk(topo_mtx);
            int ret = hwloc_get_area_membind_nodeset(topo,
                reinterpret_cast<void const*>(lva), 1, nodeset, &policy, 0);

            if (-1 != ret)
            {
                hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
                hwloc_cpuset_from_nodeset(topo, cpuset, nodeset);
                lk.unlock();

                hwloc_bitmap_free(nodeset);

                mask_type mask = mask_type();
                resize(mask, get_number_of_pus());

                int const pu_depth =
                    hwloc_get_type_or_below_depth(topo, HWLOC_OBJ_PU);
                for (unsigned int i = 0; i != num_of_pus_; ++i)
                {
                    hwloc_obj_t const pu_obj =
                        hwloc_get_obj_by_depth(topo, pu_depth, i);
                    unsigned idx = static_cast<unsigned>(pu_obj->os_index);
                    if (hwloc_bitmap_isset(cpuset, idx) != 0)
                        set(mask, detail::get_index(pu_obj));
                }

                hwloc_bitmap_free(cpuset);
                return mask;
            }
        }

        hwloc_bitmap_free(nodeset);
        return empty_mask;
    } // }}}

    std::size_t hwloc_topology::init_node_number(
        std::size_t num_thread, hwloc_obj_type_t type
        )
    { // {{{
        if (std::size_t(-1) == num_thread)
            return std::size_t(-1);

        std::size_t num_pu = (num_thread + pu_offset) % num_of_pus_;

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
                    return detail::get_index(obj);
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
                    set(mask, detail::get_index(obj)); //-V106
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

    std::size_t hwloc_topology::extract_node_count(
        hwloc_obj_t parent
      , hwloc_obj_type_t type
      , std::size_t count
        ) const
    { // {{{
        hwloc_obj_t obj;

        if(parent == NULL) return count;

        {
            scoped_lock lk(topo_mtx);
            obj = hwloc_get_next_child(topo, parent, NULL);
        }

        while (obj)
        {
            if (hwloc_compare_types(type, obj->type) == 0)
            {
                /*
                do {
                    ++count;
                    {
                        scoped_lock lk(topo_mtx);
                        obj = hwloc_get_next_child(topo, parent, obj);
                    }
                } while (obj != NULL && hwloc_compare_types(type, obj->type) == 0);
                return count;
                */
                ++count;
            }

            count = extract_node_count(obj, type, count);

            scoped_lock lk(topo_mtx);
            obj = hwloc_get_next_child(topo, parent, obj);
        }

        return count;
    } // }}}

    std::size_t hwloc_topology::get_number_of_sockets() const
    {
        int nobjs = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_SOCKET);
        if(0 > nobjs)
        {
            HPX_THROW_EXCEPTION(kernel_error
              , "hpx::threads::hwloc_topology::get_number_of_sockets"
              , "hwloc_get_nbobjs_by_type failed");
            return std::size_t(nobjs);
        }
        return std::size_t(nobjs);
    }

    std::size_t hwloc_topology::get_number_of_numa_nodes() const
    {
        int nobjs =  hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NODE);
        if(0 > nobjs)
        {
            HPX_THROW_EXCEPTION(kernel_error
              , "hpx::threads::hwloc_topology::get_number_of_numa_nodes"
              , "hwloc_get_nbobjs_by_type failed");
            return std::size_t(nobjs);
        }
        return std::size_t(nobjs);
    }

    std::size_t hwloc_topology::get_number_of_cores() const
    {
        int nobjs =  hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE);
        // If num_cores is smaller 0, we have an error, it should never be zero
        // either to avoid division by zero, we should always have at least one
        // core
        if(0 >= nobjs)
        {
            HPX_THROW_EXCEPTION(kernel_error
              , "hpx::threads::hwloc_topology::get_number_of_cores"
              , "hwloc_get_nbobjs_by_type failed");
            return std::size_t(nobjs);
        }
        return std::size_t(nobjs);
    }

    std::size_t hwloc_topology::get_number_of_socket_pus(
        std::size_t num_socket
        ) const
    {
        hwloc_obj_t socket_obj;

        {
            scoped_lock lk(topo_mtx);
            socket_obj = hwloc_get_obj_by_type(topo,
                HWLOC_OBJ_SOCKET, static_cast<unsigned>(num_socket));
        }

        if (socket_obj)
        {
            std::size_t pu_count = 0;
            return extract_node_count(socket_obj, HWLOC_OBJ_PU, pu_count);
        }

        return num_of_pus_;
    }

    std::size_t hwloc_topology::get_number_of_numa_node_pus(
        std::size_t numa_node
        ) const
    {
        hwloc_obj_t node_obj;

        {
            scoped_lock lk(topo_mtx);
            node_obj = hwloc_get_obj_by_type(topo,
                HWLOC_OBJ_NODE, static_cast<unsigned>(numa_node));
        }

        if (node_obj)
        {
            std::size_t pu_count = 0;
            return extract_node_count(node_obj, HWLOC_OBJ_PU, pu_count);
        }

        return num_of_pus_;
    }

    std::size_t hwloc_topology::get_number_of_core_pus(
        std::size_t core
        ) const
    {
        hwloc_obj_t core_obj;

        {
            scoped_lock lk(topo_mtx);
            core_obj = hwloc_get_obj_by_type(topo,
                HWLOC_OBJ_CORE, static_cast<unsigned>(core));
        }

        if (core_obj)
        {
            std::size_t pu_count = 0;
            return extract_node_count(core_obj, HWLOC_OBJ_PU, pu_count);
        }

        return num_of_pus_;
    }

    std::size_t hwloc_topology::get_number_of_socket_cores(
        std::size_t num_socket
        ) const
    {
        hwloc_obj_t socket_obj;

        {
            scoped_lock lk(topo_mtx);
            socket_obj = hwloc_get_obj_by_type(topo,
                HWLOC_OBJ_SOCKET, static_cast<unsigned>(num_socket));
        }

        if (socket_obj)
        {
            std::size_t pu_count = 0;
            return extract_node_count(socket_obj, HWLOC_OBJ_CORE, pu_count);
        }

        return get_number_of_cores();
    }

    std::size_t hwloc_topology::get_number_of_numa_node_cores(
        std::size_t numa_node
        ) const
    {
        hwloc_obj_t node_obj;

        {
            scoped_lock lk(topo_mtx);
            node_obj = hwloc_get_obj_by_type(topo,
                HWLOC_OBJ_NODE, static_cast<unsigned>(numa_node));
        }

        if (node_obj)
        {
            std::size_t pu_count = 0;
            return extract_node_count(node_obj, HWLOC_OBJ_CORE, pu_count);
        }

        return get_number_of_cores();
    }

    namespace detail
    {
        void print_info(std::ostream& os, hwloc_obj_t obj, char const* name, bool comma)
        {
            if (comma)
                os << ", ";
            os << name;

            if (obj->logical_index != ~0x0u)
                os << "L#" << obj->logical_index;
            if (obj->os_index != ~0x0u)
                os << "(P#" << obj->os_index << ")";
        }

        void print_info(std::ostream& os, hwloc_obj_t obj, bool comma = false)
        {
            switch (obj->type) {
            case HWLOC_OBJ_PU:
                print_info(os, obj, "PU ", comma);
                break;

            case HWLOC_OBJ_CORE:
                print_info(os, obj, "Core ", comma);
                break;

            case HWLOC_OBJ_SOCKET:
                print_info(os, obj, "Socket ", comma);
                break;

            case HWLOC_OBJ_NODE:
                print_info(os, obj, "Node ", comma);
                break;

            default:
                break;
            }
        }
    }

    void hwloc_topology::print_affinity_mask(std::ostream& os,
        std::size_t num_thread, mask_type const& m) const
    {
        boost::io::ios_flags_saver ifs(os);
        bool first = true;

        for(std::size_t i = 0; i != num_of_pus_; ++i)
        {

            hwloc_obj_t obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, unsigned(i));
            if (!obj)
            {
                HPX_THROW_EXCEPTION(kernel_error
                  , "hpx::threads::hwloc_topology::print_affinity_mask"
                  , "object not found");
                return;
            }

            if(!test(m, detail::get_index(obj))) //-V106
                continue;

            if (first) {
                first = false;
                os << std::setw(4) << num_thread << ": "; //-V112 //-V128
            }
            else {
                os << "      ";
            }

            detail::print_info(os, obj);

            while(obj->parent)
            {
                detail::print_info(os, obj->parent, true);
                obj = obj->parent;
            }

            os << std::endl;
        }
    }

    mask_type hwloc_topology::init_machine_affinity_mask() const
    { // {{{
        mask_type machine_affinity_mask = mask_type();
        resize(machine_affinity_mask, get_number_of_pus());

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
        return empty_mask;
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
            mask_type socket_affinity_mask = mask_type();
            resize(socket_affinity_mask, get_number_of_pus());

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
            mask_type node_affinity_mask = mask_type();
            resize(node_affinity_mask, get_number_of_pus());

            extract_node_mask(numa_node_obj, node_affinity_mask);
            return node_affinity_mask;
        }

        return machine_affinity_mask_;
    } // }}}

    mask_type hwloc_topology::init_core_affinity_mask_from_core(
        std::size_t core, mask_cref_type default_mask
        ) const
    { // {{{
        if (std::size_t(-1) == core)
            return default_mask;

        hwloc_obj_t core_obj;

        std::size_t num_core = (core + core_offset) % get_number_of_cores();

        {
            scoped_lock lk(topo_mtx);
            core_obj = hwloc_get_obj_by_type(topo,
                HWLOC_OBJ_CORE, static_cast<unsigned>(num_core));
        }

        if (core_obj)
        {
            mask_type core_affinity_mask = mask_type();
            resize(core_affinity_mask, get_number_of_pus());

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

        std::size_t num_pu = (num_thread + pu_offset) % num_of_pus_;

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

        mask_type mask = mask_type();
        resize(mask, get_number_of_pus());

        set(mask, detail::get_index(obj)); //-V106

        return mask;
    } // }}}

    mask_type hwloc_topology::init_thread_affinity_mask(
        std::size_t num_core,
        std::size_t num_pu
        ) const
    { // {{{
        hwloc_obj_t obj;

        {
            scoped_lock lk(topo_mtx);
            int num_cores = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE);
            // If num_cores is smaller 0, we have an error, it should never be zero
            // either to avoid division by zero, we should always have at least one
            // core
            if (num_cores <= 0) {
                HPX_THROW_EXCEPTION(kernel_error
                  , "hpx::threads::hwloc_topology::init_thread_affinity_mask"
                  , "hwloc_get_nbobjs_by_type failed");
                return empty_mask;
            }

            num_core = (num_core + core_offset) % std::size_t(num_cores);
            obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_CORE,
                    static_cast<unsigned>(num_core));
        }

        if (!obj)
            return empty_mask;//get_core_affinity_mask(num_thread, false);

        num_pu %= obj->arity; //-V101 //-V104

        mask_type mask = mask_type();
        resize(mask, get_number_of_pus());

        set(mask, detail::get_index(obj->children[num_pu])); //-V106

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
            {
                num_of_pus_ = static_cast<std::size_t>(num_of_pus);
                pu_numbers_.resize(num_of_pus_);
                for(std::size_t i = 0; i < num_of_pus_; ++i)
                {
                    hwloc_obj_t obj;
                    obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU,
                        static_cast<unsigned>(i));
                    if(!obj) pu_numbers_[i] = i;
                    else     pu_numbers_[i] = std::size_t(detail::get_index(obj));
                }
            }
        }
    }

    std::size_t hwloc_topology::get_number_of_pus() const
    {
        return num_of_pus_;
    }

    ///////////////////////////////////////////////////////////////////////////
    mask_type hwloc_topology::get_cpubind_mask(error_code& ec) const
    {
        hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();

        mask_type mask = mask_type();
        resize(mask, get_number_of_pus());

        {
            scoped_lock lk(topo_mtx);
            if (hwloc_get_cpubind(topo, cpuset, HWLOC_CPUBIND_THREAD))
            {
                hwloc_bitmap_free(cpuset);
                HPX_THROWS_IF(ec, kernel_error
                  , "hpx::threads::hwloc_topology::get_cpubind_mask"
                  , "hwloc_get_cpubind failed");
                return empty_mask;
            }

            int const pu_depth = hwloc_get_type_or_below_depth(topo, HWLOC_OBJ_PU);
            for (unsigned int i = 0; i != num_of_pus_; ++i) //-V104
            {
                hwloc_obj_t const pu_obj = hwloc_get_obj_by_depth(topo, pu_depth, i);
                unsigned idx = static_cast<unsigned>(pu_obj->os_index);
                if (hwloc_bitmap_isset(cpuset, idx) != 0)
                    set(mask, detail::get_index(pu_obj));
            }
        }

        hwloc_bitmap_free(cpuset);

        if (&ec != &throws)
            ec = make_success_code();

        return mask;
    }

    mask_type hwloc_topology::get_cpubind_mask(boost::thread & handle,
        error_code& ec) const
    {
        hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();

        mask_type mask = mask_type();
        resize(mask, get_number_of_pus());

        {
            scoped_lock lk(topo_mtx);
            if (hwloc_get_thread_cpubind(topo, handle.native_handle(), cpuset,
                    HWLOC_CPUBIND_THREAD))
            {
                hwloc_bitmap_free(cpuset);
                HPX_THROWS_IF(ec, kernel_error
                  , "hpx::threads::hwloc_topology::get_cpubind_mask"
                  , "hwloc_get_cpubind failed");
                return empty_mask;
            }

            int const pu_depth = hwloc_get_type_or_below_depth(topo, HWLOC_OBJ_PU);
            for (unsigned int i = 0; i != num_of_pus_; ++i) //-V104
            {
                hwloc_obj_t const pu_obj = hwloc_get_obj_by_depth(topo, pu_depth, i);
                unsigned idx = static_cast<unsigned>(pu_obj->os_index);
                if (hwloc_bitmap_isset(cpuset, idx) != 0)
                    set(mask, detail::get_index(pu_obj));
            }
        }

        hwloc_bitmap_free(cpuset);

        if (&ec != &throws)
            ec = make_success_code();

        return mask;
    }

    /// This is equivalent to malloc(), except that it tries to allocate
    /// page-aligned memory from the OS.
    void* hwloc_topology::allocate(std::size_t len)
    {
        return hwloc_alloc(topo, len);
    }

    /// Free memory that was previously allocated by allocate
    void hwloc_topology::deallocate(void* addr, std::size_t len)
    {
        hwloc_free(topo, addr, len);
    }
}}

#endif
