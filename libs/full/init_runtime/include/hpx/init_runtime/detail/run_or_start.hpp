//  Copyright (c)      2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/functional/function.hpp>
#include <hpx/hpx_init_params.hpp>
#include <hpx/modules/program_options.hpp>

namespace hpx {
    /// \cond NOINTERNAL
    namespace detail {
        HPX_EXPORT int run_or_start(
            hpx::function<int(hpx::program_options::variables_map& vm)> const&
                f,
            int argc, char** argv, init_params const& params, bool blocking);
    }    // namespace detail
    /// \endcond
}    // namespace hpx
