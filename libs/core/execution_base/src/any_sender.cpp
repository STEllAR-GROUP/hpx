//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/errors/error.hpp>
#include <hpx/errors/throw_exception.hpp>
#include <hpx/execution_base/any_sender.hpp>
#include <hpx/modules/format.hpp>

#include <atomic>
#include <exception>
#include <string>
#include <utility>

namespace hpx::execution::experimental::detail {
    void empty_any_operation_state::start() & noexcept
    {
        HPX_THROW_EXCEPTION(hpx::bad_function_call,
            "attempted to call start on empty any_operation_state",
            "any_operation_state::start");
    }

    bool empty_any_operation_state::empty() const noexcept
    {
        return true;
    }

    void tag_invoke(
        hpx::execution::experimental::start_t, any_operation_state& os) noexcept
    {
        os.storage.get().start();
    }

    void throw_bad_any_call(char const* class_name, char const* function_name)
    {
        HPX_THROW_EXCEPTION(hpx::bad_function_call,
            hpx::util::format(
                "attempted to call {} on empty {}", function_name, class_name),
            hpx::util::format("{}::{}", class_name, function_name));
    }
}    // namespace hpx::execution::experimental::detail
