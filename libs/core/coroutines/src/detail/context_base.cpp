//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2012 Hartmut Kaiser
//  Copyright (c) 2009 Oliver Kowalke
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/coroutines/detail/context_base.hpp>
#include <hpx/coroutines/detail/coroutine_impl.hpp>

namespace hpx { namespace threads { namespace coroutines { namespace detail {

    template class context_base<coroutine_impl>;

}}}}    // namespace hpx::threads::coroutines::detail
