//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/config.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/runtime_local/get_locality_name.hpp>
#include <hpx/runtime_local/runtime_local.hpp>

#include <string>

namespace hpx { namespace detail {

    std::string get_locality_base_name()
    {
        runtime* rt = get_runtime_ptr();
        if (rt == nullptr)
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::detail::get_locality_name",
                "the runtime system is not operational at this point");
            return "";
        }
        return rt->get_locality_name();
    }

    std::string get_locality_name()
    {
        std::string basename = get_locality_base_name();
        return basename + '#' + std::to_string(get_locality_id());
    }
}}    // namespace hpx::detail

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    std::string get_locality_name()
    {
        return detail::get_locality_name();
    }
}    // namespace hpx
