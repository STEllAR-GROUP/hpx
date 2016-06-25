//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/lcos/detail/promise_lco.hpp>

#include <hpx/runtime/components/component_type.hpp>
#include <hpx/util/atomic_count.hpp>

namespace hpx { namespace traits { namespace detail
{
    util::atomic_count unique_type(
        static_cast<long>(components::component_last));
}}}
