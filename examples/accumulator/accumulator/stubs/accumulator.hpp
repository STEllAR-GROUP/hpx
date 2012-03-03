//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_ACCUMULATOR_JUN_09_2008_0458PM)
#define HPX_COMPONENTS_STUBS_ACCUMULATOR_JUN_09_2008_0458PM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <examples/accumulator/accumulator/server/accumulator.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#accumulator class is the client side representation of all
    /// \a server#accumulator components
    struct accumulator : stub_base<server::accumulator>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Query the current value of the server#accumulator instance
        /// with the given \a gid. This is a non-blocking call. The caller
        /// needs to call \a promise#get on the return value of
        /// this function to obtain the result as returned by the accumulator.
        static lcos::promise<unsigned long> query_async(naming::id_type const& gid)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized eager_future, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::accumulator::query_action action_type;
            return lcos::eager_future<action_type>(gid);
        }

        /// Query the current value of the server#accumulator instance
        /// with the given \a gid. Block for the current accumulator value to
        /// be returned.
        static unsigned long query(naming::id_type const& gid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            return query_async(gid).get();
        }

        /// Initialize the accumulator value of the server#accumulator instance
        /// with the given \a gid
        static void init(naming::id_type gid)
        {
            applier::apply<server::accumulator::init_action>(gid);
        }

        /// Add the given number to the server#accumulator instance
        /// with the given \a gid
        static void add (naming::id_type gid, unsigned long arg)
        {
            applier::apply<server::accumulator::add_action>(gid, arg);
        }

        /// Print the current value of the server#accumulator instance
        /// with the given \a gid
        static void print(naming::id_type gid)
        {
            applier::apply<server::accumulator::print_action>(gid);
        }
    };

}}}

#endif
