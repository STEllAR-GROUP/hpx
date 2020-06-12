//  Copyright (c)      2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/prefix/find_prefix.hpp>
#include <hpx/resource_partitioner/partitioner_fwd.hpp>
#include <hpx/runtime_configuration/runtime_mode.hpp>

#include <hpx/modules/program_options.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if !defined(HPX_EXPORTS)
// This function must be implemented by the application.
int hpx_main(hpx::program_options::variables_map& vm);
typedef int (*hpx_main_type)(hpx::program_options::variables_map&);
#endif

namespace hpx { namespace detail {
    HPX_EXPORT int init_helper(hpx::program_options::variables_map&,
        util::function_nonser<int(int, char**)> const&);
}}    // namespace hpx::detail

namespace hpx { namespace resource { namespace detail {
    HPX_EXPORT partitioner& create_partitioner(
        util::function_nonser<int(
            hpx::program_options::variables_map& vm)> const& f,
        hpx::program_options::options_description const& desc_cmdline, int argc,
        char** argv, std::vector<std::string> ini_config,
        resource::partitioner_mode rpmode, runtime_mode mode, bool check,
        std::vector<std::shared_ptr<components::component_registry_base>>&
            component_registries,
        int* result);

}}}    // namespace hpx::resource::detail
