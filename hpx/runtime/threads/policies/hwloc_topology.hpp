////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2012-2013 Thomas Heller
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
        hwloc_topology();
        ~hwloc_topology();

        std::size_t get_socket_number(
            std::size_t num_thread
          , error_code& ec = throws
            ) const
        {
            return socket_numbers_[num_thread % num_of_pus_];
        }

        std::size_t get_numa_node_number(
            std::size_t num_thread
          , error_code& ec = throws
            ) const
        {
            return numa_node_numbers_[num_thread % num_of_pus_];
        }

        std::size_t get_core_number(
            std::size_t num_thread
          , error_code& ec = throws
            ) const
        {
            return core_numbers_[num_thread % num_of_pus_];
        }

        std::size_t get_pu_number(
            std::size_t num_thread
          , error_code& ec = throws
            ) const
        {
            return pu_numbers_[num_thread % num_of_pus_];
        }

        std::size_t get_pu_number(
            std::size_t num_core
          , std::size_t num_pu
          , error_code& ec = throws
            ) const;

        mask_cref_type get_machine_affinity_mask(
            error_code& ec = throws
            ) const;

        mask_cref_type get_socket_affinity_mask(
            std::size_t num_thread
          , bool numa_sensitive
          , error_code& ec = throws
            ) const;

        mask_cref_type get_numa_node_affinity_mask(
            std::size_t num_thread
          , bool numa_sensitive
          , error_code& ec = throws
            ) const;

        mask_cref_type get_core_affinity_mask(
            std::size_t num_thread
          , bool numa_sensitive
          , error_code& ec = throws
            ) const;

        mask_cref_type get_thread_affinity_mask(
            std::size_t num_thread
          , bool numa_sensitive
          , error_code& ec = throws
            ) const;

        void set_thread_affinity_mask(
            boost::thread&
          , mask_cref_type //mask
          , error_code& ec = throws
            ) const;

        void set_thread_affinity_mask(
            mask_cref_type mask
          , error_code& ec = throws
            ) const;

        mask_cref_type get_thread_affinity_mask_from_lva(
            naming::address::address_type
          , error_code& ec = throws
            ) const;

        ///////////////////////////////////////////////////////////////////////
        mask_type init_socket_affinity_mask_from_socket(
            std::size_t num_socket
            ) const;
        mask_type init_numa_node_affinity_mask_from_numa_node(
            std::size_t num_numa_node
            ) const;
        mask_type init_core_affinity_mask_from_core(
            std::size_t num_core, mask_cref_type default_mask
            ) const;
        mask_type init_thread_affinity_mask(std::size_t num_thread) const;
        mask_type init_thread_affinity_mask(
            std::size_t num_core
          , std::size_t num_pu
            ) const;

        mask_type get_cpubind_mask(error_code& ec = throws) const;

        ///////////////////////////////////////////////////////////////////////
        std::size_t get_number_of_sockets() const;
        std::size_t get_number_of_numa_nodes() const;
        std::size_t get_number_of_cores() const;
        std::size_t get_number_of_pus() const;

        ///////////////////////////////////////////////////////////////////////
        std::size_t get_number_of_socket_pus(
            std::size_t socket
            ) const;
        std::size_t get_number_of_numa_node_pus(
            std::size_t numa_node
            ) const;
        std::size_t get_number_of_core_pus(
            std::size_t core
            ) const;

        std::size_t get_number_of_socket_cores(
            std::size_t socket
            ) const;
        std::size_t get_number_of_numa_node_cores(
            std::size_t numa_node
            ) const;

        void print_affinity_mask(std::ostream& os, std::size_t num_thread,
            mask_type const& m) const;

        struct hwloc_topology_tag {};

    private:
        static mask_type empty_mask;

        std::size_t init_node_number(
            std::size_t num_thread, hwloc_obj_type_t type
            );

        std::size_t init_socket_number(std::size_t num_thread)
        {
            return init_node_number(num_thread, HWLOC_OBJ_SOCKET);
        }

        std::size_t init_numa_node_number(std::size_t num_thread)
        {
            return init_node_number(num_thread, HWLOC_OBJ_NODE);
        }

        std::size_t init_core_number(std::size_t num_thread)
        {
            return init_node_number(num_thread, HWLOC_OBJ_CORE);
        }

        void extract_node_mask(
            hwloc_obj_t parent
          , mask_type& mask
            ) const;

        std::size_t extract_node_count(
            hwloc_obj_t parent
          , hwloc_obj_type_t type
          , std::size_t count
            ) const;

        mask_type init_machine_affinity_mask() const;
        mask_type init_socket_affinity_mask(std::size_t num_thread) const
        {
            return init_socket_affinity_mask_from_socket(
                get_socket_number(num_thread));
        }
        mask_type init_numa_node_affinity_mask(std::size_t num_thread) const
        {
            return init_numa_node_affinity_mask_from_numa_node(
                get_numa_node_number(num_thread));
        }
        mask_type init_core_affinity_mask(std::size_t num_thread) const
        {
            mask_type default_mask =
                get_numa_node_affinity_mask(num_thread, false);
            return init_core_affinity_mask_from_core(
                get_core_number(num_thread), default_mask);
        }

        void init_num_of_pus();

        hwloc_topology_t topo;

        // We need to define a constant pu offset.
        // This is mainly to skip the first Core on the Xeon Phi
        // which is reserved for OS related tasks
#if !defined(HPX_NATIVE_MIC)
        static const std::size_t pu_offset = 0;
        static const std::size_t core_offset = 0;
#else
        static const std::size_t pu_offset = 4;
        static const std::size_t core_offset = 1;
#endif

        std::size_t num_of_pus_;

        mutable hpx::util::spinlock topo_mtx;
        typedef hpx::util::spinlock::scoped_lock scoped_lock;

        std::vector<std::size_t> socket_numbers_;
        std::vector<std::size_t> numa_node_numbers_;
        std::vector<std::size_t> core_numbers_;
        std::vector<std::size_t> pu_numbers_;

        mask_type machine_affinity_mask_;
        std::vector<mask_type> socket_affinity_masks_;
        std::vector<mask_type> numa_node_affinity_masks_;
        std::vector<mask_type> core_affinity_masks_;
        std::vector<mask_type> thread_affinity_masks_;
    };

    ///////////////////////////////////////////////////////////////////////////
    inline hwloc_topology& create_topology()
    {
        util::static_<hwloc_topology, hwloc_topology::hwloc_topology_tag> topo;
        return topo.get();
    }
}}

#endif // HPX_50DFC0FC_EE99_43F5_A918_01EC45A58036

