//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#include <hpx/threading_base/annotated_function.hpp>

#include <string>
#include <unordered_set>
#include <utility>

namespace hpx::detail {
    char const* store_function_annotation(std::string name)
    {
        static thread_local std::unordered_set<std::string> names;
        auto r = names.emplace(HPX_MOVE(name));
        return (*std::get<0>(r)).c_str();
    }
}    // namespace hpx::detail

#else

#include <string>

namespace hpx::detail {
    char const* store_function_annotation(std::string)
    {
        return "<unknown>";
    }
}    // namespace hpx::detail

#endif
