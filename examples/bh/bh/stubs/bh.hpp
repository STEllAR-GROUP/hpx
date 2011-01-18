//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_bh_JUN_09_2008_0458PM)
#define HPX_COMPONENTS_STUBS_bh_JUN_09_2008_0458PM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <examples/bh/bh/server/bh.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#bh class is the client side representation of all
    /// \a server#bh components
    struct bh : stub_base<server::bh>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Query the current value of the server#bh instance 
        /// with the given \a gid. This is a non-blocking call. The caller 
        /// needs to call \a future_value#get on the return value of 
        /// this function to obtain the result as returned by the bh.
        static lcos::future_value<unsigned long> query_async(naming::id_type const& gid) 
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized eager_future, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::bh::query_action action_type;
            return lcos::eager_future<action_type>(gid);
        }

        /// Query the current value of the server#bh instance 
        /// with the given \a gid. Block for the current bh value to 
        /// be returned.
        static unsigned long query(naming::id_type const& gid) 
        {
            // The following get yields control while the action above 
            // is executed and the result is returned to the future_value
            return query_async(gid).get();
        }

        /// Initialize the bh value of the server#bh instance 
        /// with the given \a gid
        static void init(naming::id_type gid) 
        {
            applier::apply<server::bh::init_action>(gid);
        }

        /// Add the given number to the server#bh instance 
        /// with the given \a gid
        static void add (naming::id_type gid, unsigned long arg) 
        {
            applier::apply<server::bh::add_action>(gid, arg);
        }

        /// Print the current value of the server#bh instance 
        /// with the given \a gid
        static void print(naming::id_type gid) 
        {
            applier::apply<server::bh::print_action>(gid);
        }
    };

}}}

#endif
