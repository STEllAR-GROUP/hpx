//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_RMA_DETAIL_ALLOCATOR
#define HPX_PARCELSET_POLICIES_RMA_DETAIL_ALLOCATOR

#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/atomic_count.hpp>
//
#include <hpx/config/parcelport_defines.hpp>
//
#include <hpx/runtime/parcelset/rma/detail/memory_region_impl.hpp>
#include <hpx/runtime/parcelset/rma/detail/memory_region_allocator.hpp>
#include <hpx/runtime/parcelset/rma/memory_pool.hpp>
//
#include <plugins/parcelport/parcelport_logging.hpp>
#include <plugins/parcelport/performance_counter.hpp>
//
#include <boost/core/null_deleter.hpp>
//
#include <atomic>
#include <stack>
#include <unordered_map>
#include <iostream>
#include <cstddef>
#include <memory>
#include <array>
#include <sstream>
#include <string>

namespace hpx {
namespace parcelset {
namespace rma
{

}}}

#endif
