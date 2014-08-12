////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c)      2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_079E367D_741C_4FA1_913F_EA33A192BDAD)
#define HPX_079E367D_741C_4FA1_913F_EA33A192BDAD

#include <hpx/runtime/threads/topology.hpp>
#include <hpx/exception.hpp>

#if defined(__ANDROID__) && defined(ANDROID)
#include <cpu-features.h>
#endif

namespace hpx { namespace threads
{

struct noop_topology : topology
{
private:
    static mask_type empty_mask;

public:
    std::size_t get_pu_number(std::size_t num_thread, error_code& ec = throws) const
    {
        return 0;
    }

    std::size_t get_numa_node_number(
        std::size_t thread_num
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return std::size_t(-1);
    }

    mask_cref_type get_machine_affinity_mask(
        error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return empty_mask;
    }

    mask_cref_type get_socket_affinity_mask(
        std::size_t thread_num
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return empty_mask;
    }

    mask_cref_type get_numa_node_affinity_mask(
        std::size_t thread_num
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return empty_mask;
    }

    mask_cref_type get_core_affinity_mask(
        std::size_t thread_num
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return empty_mask;
    }

    mask_cref_type get_thread_affinity_mask(
        std::size_t thread_num
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return empty_mask;
    }

    void set_thread_affinity_mask(
        boost::thread& thrd
      , mask_cref_type mask
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();
    }

    void set_thread_affinity_mask(
        mask_cref_type mask
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();
    }

    mask_cref_type get_thread_affinity_mask_from_lva(
        naming::address::address_type lva
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return empty_mask;
    }

    mask_type get_cpubind_mask(
        error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return empty_mask;
    }

    static std::size_t hardware_concurrency()
    {
#if defined(__ANDROID__) && defined(ANDROID)
        return std::size_t(::android_getCpuCount());
#else
        return std::size_t(boost::thread::hardware_concurrency());
#endif
    }

    std::size_t get_number_of_cores() const
    {
        return ~std::size_t(0);
    }

    std::size_t get_number_of_pus() const
    {
        return noop_topology::hardware_concurrency();
    }

    std::size_t get_number_of_core_pus(std::size_t core) const
    {
        return ~std::size_t(0);
    }

    std::size_t get_core_number(std::size_t num_thread, error_code& ec = throws) const
    {
        return 0;
    }

    void print_affinity_mask(std::ostream& os, std::size_t num_thread, mask_type const& m) const
    {
    }

    struct noop_topology_tag {};
};

///////////////////////////////////////////////////////////////////////////////
inline topology& create_topology()
{
    util::static_<noop_topology, noop_topology::noop_topology_tag> topo;
    return topo.get();
}

}}

#endif // HPX_079E367D_741C_4FA1_913F_EA33A192BDAD

