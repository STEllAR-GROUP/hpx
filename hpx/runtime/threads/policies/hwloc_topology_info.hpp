////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2012-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef HPX_RUNTIME_THREADS_POLICIES_HWLOC_TOPOLOGY_INFO_HPP
#define HPX_RUNTIME_THREADS_POLICIES_HWLOC_TOPOLOGY_INFO_HPP

#include <hpx/config.hpp>

#include <hwloc.h>

#include <hpx/compat/thread.hpp>
#include <hpx/error_code.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/resource/partitioner_fwd.hpp>

#include <hpx/util/spinlock.hpp>
#include <hpx/util/static.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#if defined(HPX_NATIVE_MIC) && HWLOC_API_VERSION < 0x00010600
#error On Intel Xeon/Phi coprocessors HPX cannot be use with a HWLOC version earlier than V1.6.
#endif

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace resource
{
    class resource_partitioner;
}}

namespace hpx { namespace threads
{
    struct hpx_hwloc_bitmap_wrapper
    {
        HPX_NON_COPYABLE(hpx_hwloc_bitmap_wrapper);

        // take ownership of the hwloc allocated bitmap
        hpx_hwloc_bitmap_wrapper(void *bmp) {
            bmp_ = reinterpret_cast<hwloc_bitmap_t >(bmp);
        }
        // frees the hwloc allocated bitmap
        ~hpx_hwloc_bitmap_wrapper() {
            hwloc_bitmap_free(bmp_);
        }

        hwloc_bitmap_t get_bmp() const {
            return bmp_;
        }

        // stringify the bitmp using hwloc
        friend HPX_EXPORT std::ostream& operator<<(std::ostream& os,
            hpx_hwloc_bitmap_wrapper const* bmp);

    private:
        // the raw bitmap object
        hwloc_bitmap_t bmp_;
    };

    /// \brief Please see hwloc documentation for the corresponding
    /// enums HWLOC_MEMBIND_XXX
    enum hpx_hwloc_membind_policy : int {
        membind_default    = HWLOC_MEMBIND_DEFAULT,
        membind_firsttouch = HWLOC_MEMBIND_FIRSTTOUCH,
        membind_bind       = HWLOC_MEMBIND_BIND,
        membind_interleave = HWLOC_MEMBIND_INTERLEAVE,
        membind_replicate  = HWLOC_MEMBIND_REPLICATE,
        membind_nexttouch  = HWLOC_MEMBIND_NEXTTOUCH,
        membind_mixed      = HWLOC_MEMBIND_MIXED
    };

    struct HPX_EXPORT hwloc_topology_info : topology
    {
        friend resource::resource_partitioner;

        hwloc_topology_info();
        ~hwloc_topology_info();

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
            std::size_t num_core
          , std::size_t num_pu
          , error_code& ec = throws
            ) const;

        mask_cref_type get_machine_affinity_mask(
            error_code& ec = throws
            ) const;

        mask_cref_type get_socket_affinity_mask(
            std::size_t num_thread
          , error_code& ec = throws
            ) const;

        mask_cref_type get_numa_node_affinity_mask(
            std::size_t num_thread
          , error_code& ec = throws
            ) const;

        mask_type get_numa_node_affinity_mask_from_numa_node(
            std::size_t numa_node
            ) const
        {
            return init_numa_node_affinity_mask_from_numa_node(numa_node);
        }

        mask_cref_type get_core_affinity_mask(
            std::size_t num_thread
          , error_code& ec = throws
            ) const;

        mask_cref_type get_thread_affinity_mask(
            std::size_t num_thread
          , error_code& ec = throws
            ) const;

        void set_thread_affinity_mask(
            mask_cref_type mask
          , error_code& ec = throws
            ) const;

        mask_type get_thread_affinity_mask_from_lva(
            naming::address_type
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
            std::size_t num_core, mask_cref_type default_mask = mask_type()
            ) const;
        mask_type init_thread_affinity_mask(std::size_t num_thread) const;
        mask_type init_thread_affinity_mask(
            std::size_t num_core
          , std::size_t num_pu
            ) const;

        mask_type get_cpubind_mask(error_code& ec = throws) const;
        mask_type get_cpubind_mask(compat::thread & handle,
            error_code& ec = throws) const;

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

        /// convert a cpu mask into a numa node mask in hwloc bitmap form
        hwloc_bitmap_ptr cpuset_to_nodeset(mask_cref_type cpuset) const;

        void print_affinity_mask(std::ostream& os, std::size_t num_thread,
            mask_cref_type m, const std::string &pool_name) const;

        struct hwloc_topology_tag {};

        void write_to_log() const;

        /// This is equivalent to malloc(), except that it tries to allocate
        /// page-aligned memory from the OS.
        void* allocate(std::size_t len) const;

        void* allocate_membind(std::size_t len,
            hwloc_bitmap_ptr bitmap,
            hpx_hwloc_membind_policy policy, int flags) const;

        /// Free memory that was previously allocated by allocate
        void deallocate(void* addr, std::size_t len) const;

        threads::mask_type get_area_membind_nodeset(
            const void *addr, std::size_t len, void *nodeset) const;

        bool set_area_membind_nodeset(
            const void *addr, std::size_t len, void *nodeset) const;

        int get_numa_domain(const void *addr, void *nodeset) const;

        void print_vector(
            std::ostream& os, std::vector<std::size_t> const& v) const;
        void print_mask_vector(
            std::ostream& os, std::vector<mask_type> const& v) const;
        void print_hwloc(std::ostream&) const;

        hwloc_bitmap_t mask_to_bitmap(mask_cref_type mask, hwloc_obj_type_t htype) const;
        mask_type bitmap_to_mask(hwloc_bitmap_t bitmap, hwloc_obj_type_t htype) const;

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
            mask_type default_mask = numa_node_affinity_masks_[num_thread];
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

        // Number masks:
        // Vectors of non-negative integers
        // Indicating which architecture object each PU belongs to.
        // For example, numa_node_numbers[0] indicates which numa node
        // number PU #0 (zero-based index) belongs to
        std::vector<std::size_t> socket_numbers_;
        std::vector<std::size_t> numa_node_numbers_;
        std::vector<std::size_t> core_numbers_;
//        std::vector<std::size_t> pu_numbers_; // (empty vector)

        // Affinity masks: vectors of bitmasks
        // - Length of the vector: number of PUs of the machine
        // - Elements of the vector:
        // Bitmasks of length equal to the number of PUs of the machine.
        // The bitmasks indicate which PUs belong to which resource.
        // For example, core_affinity_masks[0] is a bitmask, where the
        // elements = 1 indicate the PUs that belong to the core on which
        // PU #0 (zero-based index) lies.
        mask_type machine_affinity_mask_;
        std::vector<mask_type> socket_affinity_masks_;
        std::vector<mask_type> numa_node_affinity_masks_;
        std::vector<mask_type> core_affinity_masks_;
        std::vector<mask_type> thread_affinity_masks_;
    };

    ///////////////////////////////////////////////////////////////////////////
    inline hwloc_topology_info& create_topology()
    {
        util::static_<
                hwloc_topology_info, hwloc_topology_info::hwloc_topology_tag
            > topo;
        return topo.get();
    }
}}

#include <hpx/config/warnings_suffix.hpp>

#endif /*HPX_RUNTIME_THREADS_POLICIES_HWLOC_TOPOLOGY_HPP*/
