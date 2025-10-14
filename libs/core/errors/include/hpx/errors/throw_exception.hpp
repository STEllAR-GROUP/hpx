//  Copyright (c) 2007-2024 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file throw_exception.hpp
/// \page HPX_THROW_EXCEPTION, HPX_THROW_BAD_ALLOC, HPX_THROWS_IF
/// \headerfile hpx/exception.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/errors/error.hpp>
#include <hpx/errors/exception_fwd.hpp>
#include <hpx/modules/assertion.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/preprocessor.hpp>

#include <exception>
#include <string>
#include <system_error>

#include <hpx/config/warnings_prefix.hpp>

/// \cond NODETAIL
namespace hpx::detail {

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename Exception>
    [[noreturn]] HPX_CORE_EXPORT void throw_exception(Exception const& e,
        std::string const& func, std::string const& file, long line);

    HPX_CORE_MODULE_EXPORT_EXTERN [[noreturn]] HPX_CORE_EXPORT void
    throw_exception(error errcode, std::string const& msg,
        std::string const& func, std::string const& file, long line);

    HPX_CORE_MODULE_EXPORT_EXTERN [[noreturn]] HPX_CORE_EXPORT void
    throw_bad_alloc_exception(char const* func, char const* file, long line);

    HPX_CORE_MODULE_EXPORT_EXTERN [[noreturn]] HPX_CORE_EXPORT void
    rethrow_exception(exception const& e, std::string const& func);

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename Exception>
    [[nodiscard]] HPX_CORE_EXPORT std::exception_ptr get_exception(
        Exception const& e, std::string const& func = "<unknown>",
        std::string const& file = "<unknown>", long line = -1,
        std::string const& auxinfo = "");

    HPX_CORE_MODULE_EXPORT_NODISCARD std::exception_ptr get_exception(
        error errcode, std::string const& msg, throwmode mode,
        std::string const& func = "<unknown>",
        std::string const& file = "<unknown>", long line = -1,
        std::string const& auxinfo = "");

    HPX_CORE_MODULE_EXPORT_NODISCARD std::exception_ptr get_exception(
        std::error_code const& ec, std::string const& msg, throwmode mode,
        std::string const& func = "<unknown>",
        std::string const& file = "<unknown>", long line = -1,
        std::string const& auxinfo = "");

    HPX_CORE_MODULE_EXPORT void throws_if(hpx::error_code& ec, error errcode,
        std::string const& msg, std::string const& func,
        std::string const& file, long line);

    HPX_CORE_MODULE_EXPORT void throws_bad_alloc_if(
        hpx::error_code& ec, char const* func, char const* file, long line);

    HPX_CORE_MODULE_EXPORT void rethrows_if(
        hpx::error_code& ec, exception const& e, std::string const& func);

    HPX_CORE_MODULE_EXPORT_EXTERN [[noreturn]] HPX_CORE_EXPORT void
    throw_thread_interrupted_exception();
}    // namespace hpx::detail
/// \endcond

namespace hpx {
    /// \cond NOINTERNAL

    /// \brief throw a hpx::exception initialized from the given arguments
    HPX_CORE_MODULE_EXPORT_EXTERN [[noreturn]] inline void throw_exception(
        error e, std::string const& msg, std::string const& func,
        std::string const& file = "", long line = -1)
    {
        detail::throw_exception(e, msg, func, file, line);
    }
    /// \endcond
}    // namespace hpx

#include <hpx/config/warnings_suffix.hpp>
