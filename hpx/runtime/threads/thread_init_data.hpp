//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREAD_INIT_DATA_SEP_22_2009_1034AM)
#define HPX_THREAD_INIT_DATA_SEP_22_2009_1034AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <boost/move/move.hpp>

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    struct thread_init_data
    {
        thread_init_data()
          : description(0), lva(0), parent_locality_id(0), parent_id(0),
            parent_phase(0), priority(thread_priority_normal),
            num_os_thread(std::size_t(-1))
        {}

        thread_init_data(BOOST_RV_REF(thread_init_data) rhs)
          : func(boost::move(rhs.func)),
            description(rhs.description), lva(rhs.lva),
            parent_locality_id(rhs.parent_locality_id), parent_id(rhs.parent_id),
            parent_phase(rhs.parent_phase), priority(rhs.priority),
            num_os_thread(rhs.num_os_thread)
        {}

        template <typename F>
        thread_init_data(BOOST_FWD_REF(F) f, char const* desc,
                naming::address::address_type lva = 0,
                thread_priority priority = thread_priority_normal,
                std::size_t os_thread = std::size_t(-1))
          : func(boost::forward<F>(f)), description(desc),
            lva(lva), parent_locality_id(0), parent_id(0), parent_phase(0),
            priority(priority), num_os_thread(os_thread)
        {}

        HPX_STD_FUNCTION<threads::thread_function_type> func;
        char const* description;
        naming::address::address_type lva;
        boost::uint32_t parent_locality_id;
        threads::thread_id_type parent_id;
        std::size_t parent_phase;
        thread_priority priority;
        std::size_t num_os_thread;

    private:
        // we don't use the assignment operator
        thread_init_data(thread_init_data const& rhs);
        thread_init_data& operator=(thread_init_data const& rhs);
    };
}}

#endif

