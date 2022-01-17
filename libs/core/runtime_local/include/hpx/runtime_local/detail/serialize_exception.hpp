//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/serialization/exception_ptr.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <exception>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace runtime_local { namespace detail {
    HPX_CORE_EXPORT void save_custom_exception(
        hpx::serialization::output_archive&, std::exception_ptr const&,
        unsigned int);
    HPX_CORE_EXPORT void load_custom_exception(
        hpx::serialization::input_archive&, std::exception_ptr&, unsigned int);
}}}    // namespace hpx::runtime_local::detail
