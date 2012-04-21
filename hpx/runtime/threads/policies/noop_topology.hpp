////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c)      2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_079E367D_741C_4FA1_913F_EA33A192BDAD)
#define HPX_079E367D_741C_4FA1_913F_EA33A192BDAD

namespace hpx { namespace threads 
{

struct topology
{
    std::size_t get_numa_node_number(
        std::size_t thread_num
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return std::size_t(-1);
    }

    std::size_t get_numa_node_affinity_mask(
        std::size_t thread_num
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return 0;
    }

    std::size_t get_thread_affinity_mask(
        std::size_t thread_num
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return 0;
    }

    void set_thread_affinity(
        boost::thread& thrd
      , std::size_t num_thread
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();
    }

    void set_thread_affinity(
        std::size_t num_thread
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();
    }

    std::size_t get_thread_affinity_mask_from_lva(
        naming::address::address_type lva
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return 0;
    }
};

}}

#endif // HPX_079E367D_741C_4FA1_913F_EA33A192BDAD

