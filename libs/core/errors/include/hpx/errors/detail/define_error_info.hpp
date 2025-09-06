//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/errors/exception.hpp>
#include <hpx/errors/exception_info.hpp>

#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>

#define HPX_DEFINE_ERROR_INFO(NAME, TYPE)                                      \
    HPX_CORE_MODULE_EXPORT_EXTERN struct NAME : ::hpx::error_info<NAME, TYPE>  \
    {                                                                          \
        explicit NAME(TYPE const& value) noexcept(                             \
            std::is_nothrow_copy_constructible_v<TYPE>)                        \
          : error_info(value)                                                  \
        {                                                                      \
        }                                                                      \
                                                                               \
        explicit NAME(TYPE&& value) noexcept                                   \
          : error_info(HPX_MOVE(value))                                        \
        {                                                                      \
        }                                                                      \
    } /**/

#include <hpx/config/warnings_prefix.hpp>

/// \cond NODETAIL
namespace hpx::detail {

    // Stores the information about the function name the exception has been
    // raised in. This information will show up in error messages under the
    // [function] tag.
    HPX_DEFINE_ERROR_INFO(throw_function, std::string);

    // Stores the information about the source file name the exception has
    // been raised in. This information will show up in error messages under
    // the [file] tag.
    HPX_DEFINE_ERROR_INFO(throw_file, std::string);

    // Stores the information about the source file line number the exception
    // has been raised at. This information will show up in error messages
    // under the [line] tag.
    HPX_DEFINE_ERROR_INFO(throw_line, long);    //-V835

    struct std_exception : std::exception
    {
    private:
        std::string what_;

    public:
        explicit std_exception(std::string w)
          : what_(HPX_MOVE(w))
        {
        }

        [[nodiscard]] char const* what() const noexcept override
        {
            return what_.c_str();
        }
    };

    struct bad_alloc : std::bad_alloc
    {
    private:
        std::string what_;

    public:
        explicit bad_alloc(std::string w)
          : what_(HPX_MOVE(w))
        {
        }

        [[nodiscard]] char const* what() const noexcept override
        {
            return what_.c_str();
        }
    };

    struct bad_exception : std::bad_exception
    {
    private:
        std::string what_;

    public:
        explicit bad_exception(std::string w)
          : what_(HPX_MOVE(w))
        {
        }

        [[nodiscard]] char const* what() const noexcept override
        {
            return what_.c_str();
        }
    };

    struct bad_cast : std::bad_cast
    {
    private:
        std::string what_;

    public:
        explicit bad_cast(std::string w)
          : what_(HPX_MOVE(w))
        {
        }

        [[nodiscard]] char const* what() const noexcept override
        {
            return what_.c_str();
        }
    };

    struct bad_typeid : std::bad_typeid
    {
    private:
        std::string what_;

    public:
        explicit bad_typeid(std::string w)
          : what_(HPX_MOVE(w))
        {
        }

        [[nodiscard]] char const* what() const noexcept override
        {
            return what_.c_str();
        }
    };
}    // namespace hpx::detail
/// \endcond

#include <hpx/config/warnings_suffix.hpp>
