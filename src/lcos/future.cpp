//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>

HPX_REGISTER_TYPED_CONTINUATION(
    hpx::lcos::future<void>,
    hpx_lcos_future_void_typed_continuation)

HPX_REGISTER_TYPED_CONTINUATION(
    hpx::lcos::shared_future<void>,
    hpx_lcos_shared_future_void_typed_continuation)
