//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_ACCUMULATOR_JUN_09_2008_0458PM)
#define HPX_COMPONENTS_STUBS_ACCUMULATOR_JUN_09_2008_0458PM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/components/server/accumulator.hpp>
#include <hpx/lcos/simple_future.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#accumulator class is the client side representation of all
    /// \a server#accumulator components
    class accumulator
    {
    public:
        /// Create a client side representation for any existing
        /// \a server#accumulator instance.
        accumulator(applier::applier& appl) 
          : app_(appl)
        {}

        ~accumulator() 
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator value of the server#accumulator instance 
        /// with the given \a gid
        void init(naming::id_type gid) 
        {
            app_.apply<server::detail::accumulator::init_action>(gid);
        }

        /// Add the given number to the server#accumulator instance 
        /// with the given \a gid
        void add (naming::id_type gid, double arg) 
        {
            app_.apply<server::detail::accumulator::add_action>(gid, arg);
        }

        /// Print the current value of the server#accumulator instance 
        /// with the given \a gid
        void print(naming::id_type gid) 
        {
            app_.apply<server::detail::accumulator::print_action>(gid);
        }

        /// Query the current value of the server#accumulator instance 
        /// with the given \a gid. This is a non-blocking call. The caller 
        /// needs to call \a simple_future#get_result on the result of this 
        /// function to obtain the result as returned by the accumulator.
        static lcos::simple_future<double> query_async(
            applier::applier& appl, naming::id_type gid) 
        {
            // Create a simple_future, execute the required action and wait 
            // for the result to be returned to the future.
            lcos::simple_future<double> lco;

            // The simple_future instance is associated with the following 
            // apply action by sending it along as its continuation
            appl.apply<server::detail::accumulator::query_action>(
                new components::continuation(lco.get_gid(appl)), gid);

            // we simply return the initialized simple_future, the caller needs
            // to call get_result() on the return value to obtain the result
            return lco;
        }

        /// Query the current value of the server#accumulator instance 
        /// with the given \a gid. Block for the current accumulator value to 
        /// be returned.
        static double query(threadmanager::px_thread_self& self, 
            applier::applier& appl, naming::id_type gid) 
        {
            // The following get_result yields control while the action above 
            // is executed and the result is returned to the simple_future
            return query_async(appl, gid).get_result(self);
        }

        ///
        lcos::simple_future<double> query_async(naming::id_type targetgid) 
        {
            return query_async(app_, targetgid);
        }

        ///
        double query(threadmanager::px_thread_self& self,
            naming::id_type targetgid) 
        {
            return query(self, app_, targetgid);
        }

    private:
        applier::applier& app_;
    };
    
}}}

#endif
