//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
//
#include <hpx/runtime/parcelset/rma/detail/memory_region_impl.hpp>
#include <hpx/runtime/parcelset/rma/detail/memory_region_allocator.hpp>
#include <hpx/runtime/parcelset/rma/allocator.hpp>
#include <hpx/runtime/parcelset/rma/memory_pool.hpp>

namespace hpx {
namespace parcelset {
namespace rma
{
    namespace detail
    {
        memory_pool_base *get_parcelport_allocator_pool()
        {
            // we must find the default parcelport
            parcelset::parcelhandler &ph =
                hpx::get_runtime().get_parcel_handler();
            auto pp = ph.get_default_parcelport();

            rma::allocator<char> *allocator = pp->get_allocator();
            return allocator->get_memory_pool();
        }
    }

}}}

