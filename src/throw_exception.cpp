//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/compat/exception.hpp>
#include <hpx/error.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/filesystem_compatibility.hpp>

#include <boost/system/error_code.hpp>
#include <boost/throw_exception.hpp>

#include <string>

namespace hpx { namespace detail
{
    HPX_ATTRIBUTE_NORETURN void throw_exception(
        error errcode, std::string const& msg,
        std::string const& func, std::string const& file, long line)
    {
        boost::filesystem::path p(hpx::util::create_path(file));
        hpx::detail::throw_exception(
            hpx::exception(errcode, msg, hpx::plain),
            func, p.string(), line);
    }

    HPX_ATTRIBUTE_NORETURN void rethrow_exception(
        exception const& e, std::string const& func)
    {
        hpx::detail::throw_exception(
            hpx::exception(e.get_error(), e.what(), hpx::rethrow),
            func, hpx::get_error_file_name(e), hpx::get_error_line_number(e));
    }

    compat::exception_ptr get_exception(
        error errcode, std::string const& msg, throwmode mode,
        std::string const& func, std::string const& file, long line,
        std::string const& auxinfo)
    {
        boost::filesystem::path p(hpx::util::create_path(file));
        return hpx::detail::get_exception(
            hpx::exception(errcode, msg, mode),
            p.string(), file, line, auxinfo);
    }

    compat::exception_ptr get_exception(
        boost::system::error_code ec, std::string const& msg, throwmode mode,
        std::string const& func, std::string const& file, long line,
        std::string const& auxinfo)
    {
        return hpx::detail::get_exception(
            hpx::exception(ec),
            func, file, line, auxinfo);
    }

    void throws_if(
        hpx::error_code& ec, error errcode, std::string const& msg,
        std::string const& func, std::string const& file, long line)
    {
        if (&ec == &hpx::throws) {
            hpx::detail::throw_exception(errcode, msg, func, file, line);
        } else {
            ec = make_error_code(static_cast<hpx::error>(errcode), msg,
                func.c_str(), file.c_str(), line,
                (ec.category() == hpx::get_lightweight_hpx_category()) ?
                    hpx::lightweight : hpx::plain);
        }
    }

    void rethrows_if(
        hpx::error_code& ec, exception const& e, std::string const& func)
    {
        if (&ec == &hpx::throws) {
            hpx::detail::rethrow_exception(e, func);
        } else {
            ec = make_error_code(e.get_error(), e.what(),
                func.c_str(), hpx::get_error_file_name(e).c_str(),
                hpx::get_error_line_number(e),
                (ec.category() == hpx::get_lightweight_hpx_category()) ?
                    hpx::lightweight_rethrow : hpx::rethrow);
        }
    }

    HPX_ATTRIBUTE_NORETURN void throw_thread_interrupted_exception()
    {
        boost::throw_exception(hpx::thread_interrupted());
    }
}}
