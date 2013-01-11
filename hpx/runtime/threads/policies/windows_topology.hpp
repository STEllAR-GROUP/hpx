////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_E8DE9DDC_9463_40AC_BC50_4672C3920270)
#define HPX_E8DE9DDC_9463_40AC_BC50_4672C3920270

#include <Psapi.h>

#include <boost/format.hpp>

#include <hpx/runtime/threads/topology.hpp>
#include <hpx/exception.hpp>

namespace hpx { namespace threads
{

struct windows_topology : topology
{
    windows_topology()
    { // {{{
        std::size_t const num_of_cores = hardware_concurrency();

        numa_node_numbers_.reserve(num_of_cores);
        numa_node_affinity_masks_.reserve(num_of_cores);
        ns_numa_node_affinity_masks_.reserve(num_of_cores);
        thread_affinity_masks_.reserve(num_of_cores);
        ns_thread_affinity_masks_.reserve(num_of_cores);

        // Initialize each set of data entirely, as some of the initialization
        // routines rely on access to other pieces of topology data. The
        // compiler will optimize the loops where possible anyways.

        for (std::size_t i = 0; i < num_of_cores; ++i)
            numa_node_numbers_.push_back(init_numa_node_number(i));

        for (std::size_t i = 0; i < num_of_cores; ++i)
        {
            numa_node_affinity_masks_.push_back(
                init_numa_node_affinity_mask(i, false));
            ns_numa_node_affinity_masks_.push_back(
                init_numa_node_affinity_mask(i, true));
        }

        for (std::size_t i = 0; i < num_of_cores; ++i)
        {
            thread_affinity_masks_.push_back(
                init_thread_affinity_mask(i, false));
            ns_thread_affinity_masks_.push_back(
                init_thread_affinity_mask(i, true));
        }
    } // }}}

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
          , "hpx::threads::windows_topology::get_numa_node_number"
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

        std::size_t const num_of_cores = hardware_concurrency();
        // the machine mask is the bitor of all masks in the system
        mask_type mask = 0;
        for (std::size_t i = 0; i < num_of_cores; ++i)
            mask |= numa_node_affinity_masks_[i];

        return mask;
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

            return numa_sensitive ? ns_numa_node_affinity_masks_[num_thread]
                                  : numa_node_affinity_masks_[num_thread];
        }

        HPX_THROWS_IF(ec, bad_parameter
          , "hpx::threads::windows_topology::get_numa_node_affinity_mask"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return 0;
    } // }}}

    mask_type get_core_affinity_mask(
        std::size_t thread_num
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

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
          , "hpx::threads::windows_topology::get_thread_affinity_mask"
          , boost::str(boost::format(
                "thread number %1% is out of range")
                % num_thread));
        return 0;
    } // }}}

    void set_thread_affinity_mask(
        boost::thread& thrd
      , mask_type mask
      , error_code& ec = throws
        ) const
    { // {{{
        if (!SetThreadAffinityMask(thrd.native_handle(), DWORD_PTR(mask)))
        {
            HPX_THROWS_IF(ec, kernel_error
              , "hpx::threads::windows_topology::set_thread_affinity_mask"
              , boost::str(boost::format(
                    "failed to set thread %1% affinity mask")
                    % mask));
        }

        else if (&ec != &throws)
            ec = make_success_code();
    } // }}}

    void set_thread_affinity_mask(
        mask_type mask
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();
    }

    mask_type get_thread_affinity_mask_from_lva(
        naming::address::address_type lva
      , error_code& ec = throws
        ) const
    { // {{{
        PSAPI_WORKING_SET_EX_INFORMATION info;
        info.VirtualAddress = reinterpret_cast<void*>(lva);

        if (!QueryWorkingSetEx(GetCurrentProcess(), &info, sizeof(info)))
        {
            HPX_THROWS_IF(ec, kernel_error
              , "hpx::threads::windows_topology::get_thread_affinity_mask_from_lva"
              , "failed to get thread affinity mask");
        }
        if (ec)
            return 0;

        ULONGLONG mask = 0;
        if (!GetNumaNodeProcessorMask(info.VirtualAttributes.Node, &mask))
        {
            HPX_THROWS_IF(ec, kernel_error
              , "hpx::threads::windows_topology::get_thread_affinity_mask_from_lva"
              , "failed to get thread affinity mask");
        }
        if (ec)
            return 0;
        else if (&ec != &throws)
            ec = make_success_code();

        return static_cast<std::size_t>(mask);
    } // }}}

  private:
    std::size_t init_numa_node_number(
        std::size_t num_thread
        )
    { // {{{
        if (std::size_t(-1) == num_thread)
             return std::size_t(-1);

        UCHAR node_number = 0;
        if (GetNumaProcessorNode(UCHAR(num_thread), &node_number))
            return node_number;

        std::size_t num_of_cores = hardware_concurrency();
        if (0 == num_of_cores)
            num_of_cores = 1;     // assume one core

        std::size_t num_of_numa_cores = num_of_cores;
        ULONG numa_nodes = 0;
        if (GetNumaHighestNodeNumber(&numa_nodes) && 0 != numa_nodes)
            num_of_numa_cores = num_of_cores / (numa_nodes + 1);

        return num_thread / num_of_numa_cores;
    } // }}}

    mask_type init_numa_node_affinity_mask(
        std::size_t num_thread
      , bool numa_sensitive
        )
    { // {{{
        std::size_t num_of_cores = hardware_concurrency();
        UCHAR affinity = UCHAR(num_thread % num_of_cores);

        ULONG numa_nodes = 1;
        if (GetNumaHighestNodeNumber(&numa_nodes))
            ++numa_nodes;

        ULONGLONG mask = 0;
        if (numa_sensitive) {
            UCHAR numa_node = affinity % numa_nodes;
            if (!GetNumaNodeProcessorMask(numa_node, &mask))
            {
                HPX_THROW_EXCEPTION(kernel_error
                  , "hpx::threads::windows_topology::init_numa_node_affinity_mask"
                  , boost::str(boost::format(
                        "failed to initialize NUMA node affinity mask for "
                        "thread %1%")
                        % num_thread));
            }
            return static_cast<mask_type>(mask);
        }

        UCHAR numa_node = UCHAR(get_numa_node_number(num_thread));
        if (!GetNumaNodeProcessorMask(numa_node, &mask))
        {
            HPX_THROW_EXCEPTION(kernel_error
              , "hpx::threads::windows_topology::init_numa_node_affinity_mask"
              , boost::str(boost::format(
                    "failed to initialize NUMA node affinity mask for "
                    "thread %1%")
                    % num_thread));
        }

        return static_cast<mask_type>(mask);
    } // }}}

    mask_type init_thread_affinity_mask(
        std::size_t num_thread
      , bool numa_sensitive
        )
    { // {{{
        std::size_t num_of_cores = hardware_concurrency();
        std::size_t affinity = num_thread % num_of_cores;

        ULONG numa_nodes = 1;
        if (GetNumaHighestNodeNumber(&numa_nodes))
            ++numa_nodes;

        std::size_t num_of_cores_per_numa_node = num_of_cores / numa_nodes;
        ULONGLONG node_affinity_mask = 0;
        ULONGLONG mask = 0x01LL;

        if (numa_sensitive) {
            UCHAR numa_node = UCHAR(affinity % numa_nodes);

            if (!GetNumaNodeProcessorMask(numa_node, &node_affinity_mask))
            {
                HPX_THROW_EXCEPTION(kernel_error
                  , "hpx::threads::windows_topology::init_thread_affinity_mask"
                  , boost::str(boost::format(
                        "failed to initialize thread %1% affinity mask")
                        % num_thread));
            }
            mask = least_significant_bit(node_affinity_mask) <<
                (affinity / numa_nodes);
        }
        else {
            UCHAR numa_node = UCHAR(get_numa_node_number(num_thread));

            if (!GetNumaNodeProcessorMask(numa_node, &node_affinity_mask))
            {
                HPX_THROW_EXCEPTION(kernel_error
                  , "hpx::threads::windows_topology::init_thread_affinity_mask"
                  , boost::str(boost::format(
                        "failed to initialize thread %1% affinity mask")
                        % num_thread));
            }
            mask = least_significant_bit(node_affinity_mask) <<
                (affinity % num_of_cores_per_numa_node);
        }

        while (!(mask & node_affinity_mask)) {
            mask <<= 1LL;
            if (0 == mask)
                mask = 0x01LL;
        }

        return static_cast<mask_type>(mask);
    } // }}}

    std::vector<std::size_t> numa_node_numbers_;

    std::vector<mask_type> numa_node_affinity_masks_;
    std::vector<mask_type> ns_numa_node_affinity_masks_;

    std::vector<mask_type> thread_affinity_masks_;
    std::vector<mask_type> ns_thread_affinity_masks_;
};

///////////////////////////////////////////////////////////////////////////////
inline topology* create_topology()
{
    return new windows_topology;
}

}}

#endif // HPX_E8DE9DDC_9463_40AC_BC50_4672C3920270

