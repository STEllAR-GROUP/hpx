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

    HPX_CXX_EXPORT struct std_exception;
    HPX_CXX_EXPORT struct bad_alloc;
    HPX_CXX_EXPORT struct bad_exception;
    HPX_CXX_EXPORT struct bad_cast;
    HPX_CXX_EXPORT struct bad_typeid;

    ////////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT HPX_CXX_EXTERN template <typename Exception>
    [[noreturn]] HPX_CORE_EXPORT void throw_exception(Exception const& e,
        std::string const& func, std::string const& file, long line);

    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        hpx::exception const&, std::string const&, std::string const&, long);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        std::system_error const&, std::string const&, std::string const&, long);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        std::exception const&, std::string const&, std::string const&, long);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        hpx::detail::std_exception const&, std::string const&,
        std::string const&, long);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        std::bad_exception const&, std::string const&, std::string const&,
        long);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        hpx::detail::bad_exception const&, std::string const&,
        std::string const&, long);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        std::bad_typeid const&, std::string const&, std::string const&, long);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        hpx::detail::bad_typeid const&, std::string const&, std::string const&,
        long);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        std::bad_cast const&, std::string const&, std::string const&, long);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        hpx::detail::bad_cast const&, std::string const&, std::string const&,
        long);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        std::bad_alloc const&, std::string const&, std::string const&, long);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        hpx::detail::bad_alloc const&, std::string const&, std::string const&,
        long);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        std::logic_error const&, std::string const&, std::string const&, long);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        std::runtime_error const&, std::string const&, std::string const&,
        long);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        std::out_of_range const&, std::string const&, std::string const&, long);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT void throw_exception(
        std::invalid_argument const&, std::string const&, std::string const&,
        long);

    HPX_CXX_EXPORT HPX_CXX_EXTERN [[noreturn]] HPX_CORE_EXPORT void
    throw_exception(hpx::error errcode, std::string const& msg,
        std::string const& func, std::string const& file, long line);

    HPX_CXX_EXPORT HPX_CXX_EXTERN [[noreturn]] HPX_CORE_EXPORT void
    throw_bad_alloc_exception(char const* func, char const* file, long line);

    HPX_CXX_EXPORT HPX_CXX_EXTERN [[noreturn]] HPX_CORE_EXPORT void
    rethrow_exception(exception const& e, std::string const& func);

    ////////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT HPX_CXX_EXTERN template <typename Exception>
    [[nodiscard]] HPX_CORE_EXPORT std::exception_ptr get_exception(
        Exception const& e, std::string const& func = "<unknown>",
        std::string const& file = "<unknown>", long line = -1,
        std::string const& auxinfo = "");

    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(hpx::exception const&, std::string const&, std::string const&,
        long, std::string const&);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(std::system_error const&, std::string const&,
        std::string const&, long, std::string const&);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(std::exception const&, std::string const&, std::string const&,
        long, std::string const&);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(hpx::detail::std_exception const&, std::string const&,
        std::string const&, long, std::string const&);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(std::bad_exception const&, std::string const&,
        std::string const&, long, std::string const&);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(hpx::detail::bad_exception const&, std::string const&,
        std::string const&, long, std::string const&);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(std::bad_typeid const&, std::string const&,
        std::string const&, long, std::string const&);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(hpx::detail::bad_typeid const&, std::string const&,
        std::string const&, long, std::string const&);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(std::bad_cast const&, std::string const&, std::string const&,
        long, std::string const&);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(hpx::detail::bad_cast const&, std::string const&,
        std::string const&, long, std::string const&);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(std::bad_alloc const&, std::string const&, std::string const&,
        long, std::string const&);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(hpx::detail::bad_alloc const&, std::string const&,
        std::string const&, long, std::string const&);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(std::logic_error const&, std::string const&,
        std::string const&, long, std::string const&);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(std::runtime_error const&, std::string const&,
        std::string const&, long, std::string const&);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(std::out_of_range const&, std::string const&,
        std::string const&, long, std::string const&);
    HPX_CXX_EXPORT extern template HPX_CORE_EXPORT std::exception_ptr
    get_exception(std::invalid_argument const&, std::string const&,
        std::string const&, long, std::string const&);

    HPX_CORE_MODULE_EXPORT_NODISCARD std::exception_ptr get_exception(
        hpx::error errcode, std::string const& msg, throwmode mode,
        std::string const& func = "<unknown>",
        std::string const& file = "<unknown>", long line = -1,
        std::string const& auxinfo = "");

    HPX_CORE_MODULE_EXPORT_NODISCARD std::exception_ptr get_exception(
        std::error_code const& ec, std::string const& msg, throwmode mode,
        std::string const& func = "<unknown>",
        std::string const& file = "<unknown>", long line = -1,
        std::string const& auxinfo = "");

    ////////////////////////////////////////////////////////////////////////////
    HPX_CORE_MODULE_EXPORT void throws_if(hpx::error_code& ec,
        hpx::error errcode, std::string const& msg, std::string const& func,
        std::string const& file, long line);

    HPX_CORE_MODULE_EXPORT void throws_bad_alloc_if(
        hpx::error_code& ec, char const* func, char const* file, long line);

    HPX_CORE_MODULE_EXPORT void rethrows_if(
        hpx::error_code& ec, exception const& e, std::string const& func);

    HPX_CXX_EXPORT HPX_CXX_EXTERN [[noreturn]] HPX_CORE_EXPORT void
    throw_thread_interrupted_exception();
}    // namespace hpx::detail
/// \endcond

namespace hpx {
    /// \cond NOINTERNAL

    /// \brief throw a hpx::exception initialized from the given arguments
    HPX_CXX_EXPORT [[noreturn]] inline void throw_exception(error e,
        std::string const& msg, std::string const& func,
        std::string const& file = "", long line = -1)
    {
        detail::throw_exception(e, msg, func, file, line);
    }
    /// \endcond
}    // namespace hpx

#include <hpx/config/warnings_suffix.hpp>
