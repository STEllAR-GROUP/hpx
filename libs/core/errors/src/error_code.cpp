//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/errors/error_code.hpp>
#include <hpx/errors/exception.hpp>

#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
    namespace detail {
        class hpx_category : public std::error_category
        {
        public:
            const char* name() const noexcept
            {
                return "HPX";
            }

            std::string message(int value) const
            {
                if (value >= hpx::error::success &&
                    value < hpx::error::last_error)
                {
                    return std::string("HPX(") + error_names[value] +
                        ")";    //-V108
                }
                if (value & hpx::error::system_error_flag)
                {
                    return std::string("HPX(system_error)");
                }
                return "HPX(unknown_error)";
            }
        };

        struct lightweight_hpx_category : hpx_category
        {
        };

        // this doesn't add any text to the exception what() message
        class hpx_category_rethrow : public std::error_category
        {
        public:
            const char* name() const noexcept
            {
                return "";
            }

            std::string message(int) const noexcept
            {
                return "";
            }
        };

        struct lightweight_hpx_category_rethrow : hpx_category_rethrow
        {
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    std::error_category const& get_hpx_category() noexcept
    {
        static detail::hpx_category hpx_category;
        return hpx_category;
    }

    std::error_category const& get_hpx_rethrow_category() noexcept
    {
        static detail::hpx_category_rethrow hpx_category_rethrow;
        return hpx_category_rethrow;
    }

    std::error_category const& get_lightweight_hpx_category() noexcept
    {
        static detail::lightweight_hpx_category lightweight_hpx_category;
        return lightweight_hpx_category;
    }

    std::error_category const& get_hpx_category(throwmode mode) noexcept
    {
        switch (mode)
        {
        case throwmode::rethrow:
            return get_hpx_rethrow_category();

        case throwmode::lightweight:
        case throwmode::lightweight_rethrow:
            return get_lightweight_hpx_category();

        case throwmode::plain:
        default:
            break;
        }
        return get_hpx_category();
    }

    ///////////////////////////////////////////////////////////////////////////
    error_code::error_code(error e, throwmode mode)
      : std::error_code(make_system_error_code(e, mode))
    {
        if (e != hpx::error::success && e != hpx::error::no_success &&
            !(mode & throwmode::lightweight))
        {
            exception_ = detail::get_exception(e, "", mode);
        }
    }

    error_code::error_code(
        error e, char const* func, char const* file, long line, throwmode mode)
      : std::error_code(make_system_error_code(e, mode))
    {
        if (e != hpx::error::success && e != hpx::error::no_success &&
            !(mode & throwmode::lightweight))
        {
            exception_ = detail::get_exception(e, "", mode, func, file, line);
        }
    }

    error_code::error_code(error e, char const* msg, throwmode mode)
      : std::error_code(make_system_error_code(e, mode))
    {
        if (e != hpx::error::success && e != hpx::error::no_success &&
            !(mode & throwmode::lightweight))
        {
            exception_ = detail::get_exception(e, msg, mode);
        }
    }

    error_code::error_code(error e, char const* msg, char const* func,
        char const* file, long line, throwmode mode)
      : std::error_code(make_system_error_code(e, mode))
    {
        if (e != hpx::error::success && e != hpx::error::no_success &&
            !(mode & throwmode::lightweight))
        {
            exception_ = detail::get_exception(e, msg, mode, func, file, line);
        }
    }

    error_code::error_code(error e, std::string const& msg, throwmode mode)
      : std::error_code(make_system_error_code(e, mode))
    {
        if (e != hpx::error::success && e != hpx::error::no_success &&
            !(mode & throwmode::lightweight))
        {
            exception_ = detail::get_exception(e, msg, mode);
        }
    }

    error_code::error_code(error e, std::string const& msg, char const* func,
        char const* file, long line, throwmode mode)
      : std::error_code(make_system_error_code(e, mode))
    {
        if (e != hpx::error::success && e != hpx::error::no_success &&
            !(mode & throwmode::lightweight))
        {
            exception_ = detail::get_exception(e, msg, mode, func, file, line);
        }
    }

    error_code::error_code(int err, hpx::exception const& e)
    {
        this->std::error_code::assign(err, get_hpx_category());
        exception_ = std::make_exception_ptr(e);
    }

    error_code::error_code(std::exception_ptr const& e)
      : std::error_code(
            make_system_error_code(get_error(e), throwmode::rethrow))
      , exception_(e)
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    std::string error_code::get_message() const
    {
        if (exception_)
        {
            try
            {
                std::rethrow_exception(exception_);
            }
            catch (std::exception const& be)
            {
                return be.what();
            }
        }
        return get_error_what(*this);    // provide at least minimal error text
    }

    ///////////////////////////////////////////////////////////////////////////
    error_code::error_code(error_code const& rhs)
      : std::error_code(rhs.value() == hpx::error::success ?
                make_success_code(
                    (category() == get_lightweight_hpx_category()) ?
                        hpx::throwmode::lightweight :
                        hpx::throwmode::plain) :
                rhs)
      , exception_(rhs.exception_)
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    error_code& error_code::operator=(error_code const& rhs)
    {
        if (this != &rhs)
        {
            if (rhs.value() == hpx::error::success)
            {
                // if the rhs is a success code, we maintain our throw mode
                this->std::error_code::operator=(make_success_code(
                    (category() == get_lightweight_hpx_category()) ?
                        hpx::throwmode::lightweight :
                        hpx::throwmode::plain));
            }
            else
            {
                this->std::error_code::operator=(rhs);
            }
            exception_ = rhs.exception_;
        }
        return *this;
    }
    /// \endcond
}    // namespace hpx
