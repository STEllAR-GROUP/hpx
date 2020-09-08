//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/functional/bind.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/mem_fn.hpp>
#include <hpx/functional/traits/is_bind_expression.hpp>
#include <hpx/functional/traits/is_placeholder.hpp>
#include <hpx/functional/unique_function.hpp>

namespace hpx {
    using hpx::traits::is_bind_expression;
    using hpx::traits::is_placeholder;
    using hpx::util::bind_back;
    using hpx::util::bind_front;
    using hpx::util::function;
    using hpx::util::function_nonser;
    using hpx::util::invoke;
    using hpx::util::invoke_fused;
    using hpx::util::mem_fn;
    using hpx::util::unique_function;
    using hpx::util::unique_function_nonser;

    namespace placeholders {
        using namespace hpx::util::placeholders;
    }
}    // namespace hpx
