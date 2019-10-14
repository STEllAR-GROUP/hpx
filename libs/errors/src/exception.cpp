//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/errors/error.hpp>
#include <hpx/errors/error_code.hpp>
#include <hpx/errors/exception.hpp>
#include <hpx/errors/exception_info.hpp>
#include <hpx/format.hpp>
#include <hpx/logging.hpp>

#if defined(HPX_WINDOWS)
#include <process.h>
#elif defined(HPX_HAVE_UNISTD_H)
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <crt_externs.h>
#define environ (*_NSGetEnviron())
#elif defined(__FreeBSD__)
HPX_EXPORT char** freebsd_environ = nullptr;
#elif !defined(HPX_WINDOWS)
extern char** environ;
#endif

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    /// Construct a hpx::exception from a \a hpx::error.
    ///
    /// \param e    The parameter \p e holds the hpx::error code the new
    ///             exception should encapsulate.
    exception::exception(error e)
      : boost::system::system_error(make_error_code(e, plain))
    {
        HPX_ASSERT((e >= success && e < last_error) || (e & system_error_flag));
        if (e != success)
        {
            LERR_(error) << "created exception: " << this->what();
        }
    }

    /// Construct a hpx::exception from a boost#system_error.
    exception::exception(boost::system::system_error const& e)
      : boost::system::system_error(e)
    {
        LERR_(error) << "created exception: " << this->what();
    }

    /// Construct a hpx::exception from a boost#system#error_code (this is
    /// new for Boost V1.69).
    exception::exception(boost::system::error_code const& e)
      : boost::system::system_error(e)
    {
        LERR_(error) << "created exception: " << this->what();
    }

    /// Construct a hpx::exception from a \a hpx::error and an error message.
    ///
    /// \param e      The parameter \p e holds the hpx::error code the new
    ///               exception should encapsulate.
    /// \param msg    The parameter \p msg holds the error message the new
    ///               exception should encapsulate.
    /// \param mode   The parameter \p mode specifies whether the returned
    ///               hpx::error_code belongs to the error category
    ///               \a hpx_category (if mode is \a plain, this is the
    ///               default) or to the category \a hpx_category_rethrow
    ///               (if mode is \a rethrow).
    exception::exception(error e, char const* msg, throwmode mode)
      : boost::system::system_error(make_system_error_code(e, mode), msg)
    {
        HPX_ASSERT((e >= success && e < last_error) || (e & system_error_flag));
        if (e != success)
        {
            LERR_(error) << "created exception: " << this->what();
        }
    }

    /// Construct a hpx::exception from a \a hpx::error and an error message.
    ///
    /// \param e      The parameter \p e holds the hpx::error code the new
    ///               exception should encapsulate.
    /// \param msg    The parameter \p msg holds the error message the new
    ///               exception should encapsulate.
    /// \param mode   The parameter \p mode specifies whether the returned
    ///               hpx::error_code belongs to the error category
    ///               \a hpx_category (if mode is \a plain, this is the
    ///               default) or to the category \a hpx_category_rethrow
    ///               (if mode is \a rethrow).
    exception::exception(error e, std::string const& msg, throwmode mode)
      : boost::system::system_error(make_system_error_code(e, mode), msg)
    {
        HPX_ASSERT((e >= success && e < last_error) || (e & system_error_flag));
        if (e != success)
        {
            LERR_(error) << "created exception: " << this->what();
        }
    }

    /// Destruct a hpx::exception
    ///
    /// \throws nothing
    exception::~exception() noexcept {}

    /// The function \a get_error() returns the hpx::error code stored
    /// in the referenced instance of a hpx::exception. It returns
    /// the hpx::error code this exception instance was constructed
    /// from.
    ///
    /// \throws nothing
    error exception::get_error() const noexcept
    {
        return static_cast<error>(
            this->boost::system::system_error::code().value());
    }

    /// The function \a get_error_code() returns a hpx::error_code which
    /// represents the same error condition as this hpx::exception instance.
    ///
    /// \param mode   The parameter \p mode specifies whether the returned
    ///               hpx::error_code belongs to the error category
    ///               \a hpx_category (if mode is \a plain, this is the
    ///               default) or to the category \a hpx_category_rethrow
    ///               (if mode is \a rethrow).
    error_code exception::get_error_code(throwmode mode) const noexcept
    {
        (void) mode;
        return error_code(
            this->boost::system::system_error::code().value(), *this);
    }

    static custom_exception_info_handler_type custom_exception_info_handler;

    HPX_EXPORT void set_custom_exception_info_handler(
        custom_exception_info_handler_type f)
    {
        custom_exception_info_handler = f;
    }

    static pre_exception_handler_type pre_exception_handler;

    HPX_EXPORT void set_pre_exception_handler(pre_exception_handler_type f)
    {
        pre_exception_handler = f;
    }
}    // namespace hpx

namespace hpx { namespace detail {
    template <typename Exception>
    HPX_EXPORT std::exception_ptr construct_lightweight_exception(
        Exception const& e, std::string const& func, std::string const& file,
        long line)
    {
        // create a std::exception_ptr object encapsulating the Exception to
        // be thrown and annotate it with all the local information we have
        try
        {
            throw_with_info(e,
                std::move(
                    hpx::exception_info().set(hpx::detail::throw_function(func),
                        hpx::detail::throw_file(file),
                        hpx::detail::throw_line(line))));
        }
        catch (...)
        {
            return std::current_exception();
        }

        // need this return to silence a warning with icc
        HPX_ASSERT(false);
        return std::exception_ptr();
    }

    template <typename Exception>
    HPX_EXPORT std::exception_ptr construct_lightweight_exception(
        Exception const& e)
    {
        // create a std::exception_ptr object encapsulating the Exception to
        // be thrown and annotate it with all the local information we have
        try
        {
            hpx::throw_with_info(e);
        }
        catch (...)
        {
            return std::current_exception();
        }

        // need this return to silence a warning with icc
        HPX_ASSERT(false);
        return std::exception_ptr();
    }

    template HPX_EXPORT std::exception_ptr construct_lightweight_exception(
        hpx::thread_interrupted const&);

    template <typename Exception>
    HPX_EXPORT std::exception_ptr construct_custom_exception(Exception const& e,
        std::string const& func, std::string const& file, long line,
        std::string const& auxinfo)
    {
        if (!custom_exception_info_handler)
        {
            return construct_lightweight_exception(e, func, file, line);
        }

        // create a std::exception_ptr object encapsulating the Exception to
        // be thrown and annotate it with information provided by the hook
        try
        {
            throw_with_info(
                e, custom_exception_info_handler(func, file, line, auxinfo));
        }
        catch (...)
        {
            return std::current_exception();
        }

        // need this return to silence a warning with icc
        HPX_ASSERT(false);
        return std::exception_ptr();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Exception>
    inline bool is_of_lightweight_hpx_category(Exception const& e)
    {
        return false;
    }

    inline bool is_of_lightweight_hpx_category(hpx::exception const& e)
    {
        return e.get_error_code().category() == get_lightweight_hpx_category();
    }

    ///////////////////////////////////////////////////////////////////////////
    std::exception_ptr access_exception(error_code const& e)
    {
        return e.exception_;
    }

    template <typename Exception>
    HPX_EXPORT std::exception_ptr get_exception(Exception const& e,
        std::string const& func, std::string const& file, long line,
        std::string const& auxinfo)
    {
        if (is_of_lightweight_hpx_category(e))
        {
            return construct_lightweight_exception(e, func, file, line);
        }

        return construct_custom_exception(e, func, file, line, auxinfo);
    }

    template <typename Exception>
    HPX_EXPORT void throw_exception(Exception const& e, std::string const& func,
        std::string const& file, long line)
    {
        if (pre_exception_handler)
        {
            pre_exception_handler();
        }

        std::rethrow_exception(get_exception(e, func, file, line));
    }

    ///////////////////////////////////////////////////////////////////////////
    template HPX_EXPORT std::exception_ptr get_exception(hpx::exception const&,
        std::string const&, std::string const&, long, std::string const&);

    template HPX_EXPORT void throw_exception(
        hpx::exception const&, std::string const&, std::string const&, long);

    template HPX_EXPORT void throw_exception(boost::system::system_error const&,
        std::string const&, std::string const&, long);

    template HPX_EXPORT void throw_exception(
        std::exception const&, std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(hpx::detail::std_exception const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(std::bad_exception const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(hpx::detail::bad_exception const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(
        std::bad_typeid const&, std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(hpx::detail::bad_typeid const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(
        std::bad_cast const&, std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(hpx::detail::bad_cast const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(
        std::bad_alloc const&, std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(hpx::detail::bad_alloc const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(
        std::logic_error const&, std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(std::runtime_error const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(
        std::out_of_range const&, std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(std::invalid_argument const&,
        std::string const&, std::string const&, long);
}}    // namespace hpx::detail

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    /// Return the error message.
    std::string get_error_what(hpx::exception_info const& xi)
    {
        // Try a cast to std::exception - this should handle boost.system
        // error codes in addition to the standard library exceptions.
        std::exception const* se = dynamic_cast<std::exception const*>(&xi);
        return se ? se->what() : std::string("<unknown>");
    }

    ///////////////////////////////////////////////////////////////////////////
    error get_error(hpx::exception const& e)
    {
        return static_cast<hpx::error>(e.get_error());
    }

    error get_error(hpx::error_code const& e)
    {
        return static_cast<hpx::error>(e.value());
    }

    error get_error(std::exception_ptr const& e)
    {
        try
        {
            std::rethrow_exception(e);
        }
        catch (hpx::thread_interrupted const&)
        {
            return hpx::thread_cancelled;
        }
        catch (hpx::exception const& he)
        {
            return he.get_error();
        }
        catch (boost::system::system_error const& e)
        {
            int code = e.code().value();
            if (code < success || code >= last_error)
                code |= system_error_flag;
            return static_cast<hpx::error>(code);
        }
        catch (...)
        {
            return unknown_error;
        }
    }

    /// Return the function name from which the exception was thrown.
    std::string get_error_function_name(hpx::exception_info const& xi)
    {
        std::string const* function = xi.get<hpx::detail::throw_function>();
        if (function)
            return *function;

        return std::string();
    }

    /// Return the (source code) file name of the function from which the
    /// exception was thrown.
    std::string get_error_file_name(hpx::exception_info const& xi)
    {
        std::string const* file = xi.get<hpx::detail::throw_file>();
        if (file)
            return *file;

        return "<unknown>";
    }

    /// Return the line number in the (source code) file of the function from
    /// which the exception was thrown.
    long get_error_line_number(hpx::exception_info const& xi)
    {
        long const* line = xi.get<hpx::detail::throw_line>();
        if (line)
            return *line;
        return -1;
    }

}    // namespace hpx
