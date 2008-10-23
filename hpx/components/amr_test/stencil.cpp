//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/components/amr_test/stencil.hpp>

#include <boost/tuple/tuple.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    template <typename T1, typename T2, typename T3>
    inline boost::tuple<T1, T2, T3>
    wait (threads::thread_self& self, lcos::future_value<T1>& f1, 
        lcos::future_value<T2>& f2, lcos::future_value<T3>& f3)
    {
        return boost::make_tuple(f1.get(self), f2.get(self), f3.get(self));
    }

    template <typename T1, typename T2, typename T3, typename T4>
    inline boost::tuple<T1, T2, T3, T4>
    wait (threads::thread_self& self, lcos::future_value<T1>& f1, 
        lcos::future_value<T2>& f2, lcos::future_value<T3>& f3, 
        lcos::future_value<T4>& f4)
    {
        return boost::make_tuple(f1.get(self), f2.get(self), f3.get(self), f4.get(self));
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil
    // Compute the result value for the current time step
    threads::thread_state stencil::eval(threads::thread_self& self, 
        applier::applier& appl, naming::id_type* result, 
        naming::id_type const& gid1, naming::id_type const& gid2, 
        naming::id_type const& gid3)
    {
        // count timesteps
        ++timestep_;

        // start asynchronous get operations
        stubs::memory_block stub(appl);

        // get all input memory_block_data instances, create new dataset
        memory_data<double> val1, val2, val3;
        naming::id_type retval_gid;

        boost::tie(val1, val2, val3, retval_gid) = wait(self, 
            stub.get_async(gid1), stub.get_async(gid2), stub.get_async(gid3),
            stub.create_async(appl.get_prefix(), sizeof(double)));

        // create new dataset
        memory_block retval(appl, retval_gid);

        // do the actual computation
        memory_data<double> r(retval.get(self));
        *r = (*val1 + *val2 + *val3) / 3;

        // store result value
        *result = retval.get_gid();
        return threads::terminated;
    }

    // Return, whether the current time step is the final one
    threads::thread_state stencil::is_last_timestep(threads::thread_self&, 
            applier::applier&, bool* result)
    {
        *result = (timestep_ == 2) ? true : false;
        return threads::terminated;
    }

}}}

