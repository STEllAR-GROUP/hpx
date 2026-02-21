//  Copyright (c) 2007-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/serialization.hpp>

#include <exception>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::runtime_local::detail {

    HPX_CXX_CORE_EXPORT HPX_CORE_EXPORT void save_custom_exception(
        hpx::serialization::output_archive&, std::exception_ptr const&,
        unsigned int);
    HPX_CXX_CORE_EXPORT HPX_CORE_EXPORT void load_custom_exception(
        hpx::serialization::input_archive&, std::exception_ptr&, unsigned int);
}    // namespace hpx::runtime_local::detail
