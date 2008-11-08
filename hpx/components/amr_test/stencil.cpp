//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/components/amr_test/stencil.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////
    struct timestep_data
    {
        int index_;       // sequential number of this datapoint
        int timestep_;    // current time step
        double value_;    // current value
    };

    ///////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil
    // Compute the result value for the current time step
    threads::thread_state stencil::eval(threads::thread_self& self, 
        applier::applier& appl, bool* is_last, naming::id_type const& result, 
        std::vector<naming::id_type> const& gids)
    {
        BOOST_ASSERT(gids.size() == 3);

        // start asynchronous get operations
        stubs::memory_block stub(appl);

        // get all input memory_block_data instances
        access_memory_block<timestep_data> val1, val2, val3, resultval;
        boost::tie(val1, val2, val3, resultval) = 
            wait(self, stub.get_async(gids[0]), 
                stub.get_async(gids[1]), stub.get_async(gids[2]),
                stub.get_async(result));

        // make sure all input data items agree on the time step number
        BOOST_ASSERT(val1->timestep_ == val2->timestep_);
        BOOST_ASSERT(val1->timestep_ == val3->timestep_);

        // the middle point is our direct predecessor
        resultval->index_ = val2->index_;
        if (val2->timestep_ < 2) {
            // this is the actual calculation
            resultval->timestep_ = val2->timestep_ + 1;
            resultval->value_ = (val1->value_ + val2->value_ + val3->value_) / 3;
        }
        else {
            // the last time step has been reached, just copy over the data
            resultval->timestep_ = val2->timestep_;
            resultval->value_ = val2->value_;
        }

        // set return value to true if this is the last time step
        *is_last = resultval->timestep_ >= 2;

        // store result value
        return threads::terminated;
    }

    threads::thread_state stencil::alloc_data(threads::thread_self& self, 
        applier::applier& appl, naming::id_type* result, int item)
    {
        *result = components::stubs::memory_block::create(self, appl, 
            appl.get_runtime_support_gid(), sizeof(timestep_data));

        if (-1 != item) {
            // provide initial data for the given data value 
            access_memory_block<timestep_data> val(
                components::stubs::memory_block::checkout(self, appl, *result));

            val->index_ = item;
            val->timestep_ = 0;
            val->value_ = item;
        }
        return threads::terminated;
    }

    /// The free function releases the memory allocated by init
    threads::thread_state stencil::free_data(threads::thread_self& self, 
        applier::applier& appl, naming::id_type const& gid)
    {
        components::stubs::memory_block::free(appl, gid);
        return threads::terminated;
    }

}}}

