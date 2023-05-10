//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/program_options/variables_map.hpp>

#include <functional>

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm);

///////////////////////////////////////////////////////////////////////////////
namespace hpx_startup {

    std::function<int(hpx::program_options::variables_map&)> const&
    get_main_func()
    {
        static std::function main_f(
            static_cast<hpx::hpx_main_type>(::hpx_main));

        return main_f;
    }
}    // namespace hpx_startup
