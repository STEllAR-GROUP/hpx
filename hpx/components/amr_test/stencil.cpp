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
    ///////////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil
    // Compute the result value for the current time step
    threads::thread_state stencil::eval(threads::thread_self& self, 
        applier::applier& appl, bool* is_last, naming::id_type const& result, 
        std::vector<naming::id_type> const& gids)
    {
//         // start asynchronous get operations
//         stubs::memory_block stub(appl);
// 
//         // get all input memory_block_data instances, create new dataset
//         access_memory_block<double> val1, val2, val3;
//         boost::tie(val1, val2, val3) = wait(self, 
//             stub.get_async(gid1), stub.get_async(gid2), stub.get_async(gid3));
// 
//         // we reuse the spot of the middle point
//         *val2 = (*val1 + *val2 + *val3) / 3;
// 
//         // store result value
//         *result = gid2;
        return threads::terminated;
    }

    threads::thread_state stencil::init(threads::thread_self&, 
        applier::applier&, naming::id_type* result)
    {
        return threads::terminated;
    }

}}}

