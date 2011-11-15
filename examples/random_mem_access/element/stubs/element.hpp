//  Copyright (c) 2011 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_5678AE3E_BDD2_4B46_9A6E_196038D8261D)
#define HPX_5678AE3E_BDD2_4B46_9A6E_196038D8261D

#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/element.hpp"

namespace random_mem_access { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct element : hpx::components::stub_base<server::element>
    {
        /// Initialize the element. Fire-and-forget semantics.
        static void init(hpx::naming::id_type const& gid, std::size_t i)
        {
            typedef server::element::init_action action_type;
            hpx::lcos::eager_future<action_type>(gid, i).get();
        }

        /// Increment the element's value. 
        static void add(hpx::naming::id_type const& gid)
        {
            add_async(gid).get();
        }

        /// Asynchronously increment the element's value.
        static hpx::lcos::promise<void>
        add_async(hpx::naming::id_type const& gid)
        {
            typedef server::element::add_action action_type;
            return hpx::lcos::eager_future<action_type>(gid);
        }

        /// Print the current value of the element.
        static void print(hpx::naming::id_type const& gid)
        {
            print_async(gid).get();
        }

        /// Asynchronously print the current value of the element.
        static hpx::lcos::promise<void>
        print_async(hpx::naming::id_type const& gid)
        {
            typedef server::element::print_action action_type;
            return hpx::lcos::eager_future<action_type>(gid);
        }
    };
}}

#endif

