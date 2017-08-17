//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/report_error.hpp

#if !defined(HPX_RUNTIME_REPORT_ERROR_HPP)
#define HPX_RUNTIME_REPORT_ERROR_HPP

#include <hpx/config.hpp>

#include <cstddef>
#include <exception>

namespace hpx
{
    /// The function report_error reports the given exception to the console
    HPX_API_EXPORT void report_error(std::size_t num_thread,
        std::exception_ptr const& e);

    /// The function report_error reports the given exception to the console
    HPX_API_EXPORT void report_error(std::exception_ptr const& e);
}

#endif

