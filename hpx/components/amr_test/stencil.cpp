//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/future_wait.hpp>

#include <hpx/components/amr_test/stencil.hpp>
#include <hpx/components/amr_test/logging.hpp>
#include <hpx/components/amr_test/stencil_data.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    stencil::stencil()
      : numsteps_(0)
    {}

    ///////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil
    // Compute the result value for the current time step
    threads::thread_state stencil::eval(bool* is_last, 
        naming::id_type const& result, std::vector<naming::id_type> const& gids)
    {
        BOOST_ASSERT(gids.size() == 3);

        // start asynchronous get operations
        components::stubs::memory_block stub;

        // get all input memory_block_data instances
        access_memory_block<timestep_data> val1, val2, val3, resultval;
        boost::tie(val1, val2, val3, resultval) = 
            wait(stub.get_async(gids[0]), 
                stub.get_async(gids[1]), stub.get_async(gids[2]),
                stub.get_async(result));

        // make sure all input data items agree on the time step number
        BOOST_ASSERT(val1->timestep_ == val2->timestep_);
        BOOST_ASSERT(val1->timestep_ == val3->timestep_);

        // the middle point is our direct predecessor
        resultval->max_index_ = val2->max_index_;
        resultval->index_ = val2->index_;
        if (val2->timestep_ < numsteps_) {
            // this is the actual calculation
            resultval->timestep_ = val2->timestep_ + 1;
            resultval->value_ = 0.25 * val1->value_ + 0.75 * val3->value_;

            if (log_)     // send result to logging instance
                stubs::logging::logentry(log_, resultval.get());
        }
        else {
            // the last time step has been reached, just copy over the data
            resultval->timestep_ = val2->timestep_;
            resultval->value_ = val2->value_;
        }

        // set return value to true if this is the last time step
        *is_last = resultval->timestep_ >= numsteps_;

        return threads::terminated;
    }

    threads::thread_state stencil::alloc_data(naming::id_type* result, 
        int item, int maxitems)
    {
        *result = components::stubs::memory_block::create(
            applier::get_applier().get_runtime_support_gid(), sizeof(timestep_data));

        if (-1 != item) {
            // provide initial data for the given data value 
            access_memory_block<timestep_data> val(
                components::stubs::memory_block::checkout(*result));

            val->max_index_ = maxitems;
            val->index_ = item;
            val->timestep_ = 0;
            if (item < int(maxitems / 3.) || item >= int(2. * maxitems / 3.))
                val->value_ = 0;
            else
                val->value_ = std::pow(item - 1/3, 4.) * std::pow(item - 2/3, 4.);
        }
        return threads::terminated;
    }

    /// The free function releases the memory allocated by init
    threads::thread_state stencil::free_data(
        naming::id_type const& gid)
    {
        components::stubs::memory_block::free(gid);
        return threads::terminated;
    }

    /// The free function releases the memory allocated by init
    threads::thread_state stencil::init(
        std::size_t numsteps, naming::id_type const& logging)
    {
        numsteps_ = numsteps;
        log_ = logging;
        return threads::terminated;
    }

}}}

