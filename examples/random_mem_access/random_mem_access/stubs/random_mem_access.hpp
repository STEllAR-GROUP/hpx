//  Copyright (c) 2011 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_RANDOM_JUN_06_2011_1125AM)
#define HPX_COMPONENTS_STUBS_RANDOM_JUN_06_2011_1125AM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/include/async.hpp>

#include <examples/random_mem_access/random_mem_access/server/random_mem_access.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#simple_accumulator class is the client side representation
    /// of all \a server#simple_accumulator components
    struct random_mem_access : stub_base<server::random_mem_access>
    {
        /// Query the current value of the server#simple_accumulator instance
        /// with the given \a gid. This is a non-blocking call. The caller
        /// needs to call \a future#get on the return value of
        /// this function to obtain the result as returned by the simple_accumulator.
        static lcos::future<int> query_async(naming::id_type gid)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::random_mem_access::query_action action_type;
            return hpx::async<action_type>(gid);
        }

        static lcos::future<void> add_async(naming::id_type gid)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::random_mem_access::add_action action_type;
            return hpx::async<action_type>(gid);
        }

        static lcos::future<void> print_async(naming::id_type gid)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::random_mem_access::print_action action_type;
            return hpx::async<action_type>(gid);
        }

        /// Query the current value of the server#simple_accumulator instance
        /// with the given \a gid. Block for the current simple_accumulator value to
        /// be returned.
        static double query(naming::id_type gid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return query_async(gid).get();
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the simple_accumulator
        /// value of the server#simple_accumulator instance
        /// with the given \a gid
        static void init(naming::id_type gid,int i)
        {
            hpx::apply<server::random_mem_access::init_action>(gid,i);
        }

        static void add(naming::id_type gid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            add_async(gid).get();
        }

        /// Print the current value of the server#simple_accumulator instance
        /// with the given \a gid
        static void print(naming::id_type gid)
        {
            print_async(gid).get();
          //  hpx::apply<server::random_mem_access::print_action>(gid);
        }
    };

}}}

#endif
