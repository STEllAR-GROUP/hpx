//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
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
#include <hpx/functional/move_only_function.hpp>
#include <hpx/functional/traits/is_bind_expression.hpp>
#include <hpx/functional/traits/is_placeholder.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/threading_base/scoped_annotation.hpp>

namespace hpx {

    using hpx::invoke_fused;
    using hpx::mem_fn;
}    // namespace hpx
