//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/components/console_error_sink.hpp>
#if HPX_STACKTRACES != 0
  #include <boost/backtrace.hpp>
#endif
#include <stdexcept>

namespace hpx { namespace detail
{
#if HPX_STACKTRACES != 0

    HPX_EXPORT std::string backtrace()
    {
        return boost::trace();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Exception>
    HPX_EXPORT void throw_exception(Exception const& e, std::string const& func, 
        std::string const& file, int line, std::string const& back_trace)
    {
        threads::thread_self* self = threads::get_self_ptr();
        if (NULL != self) {
            threads::thread_id_type id = self->get_thread_id();
            throw boost::enable_current_exception(
                boost::enable_error_info(e) 
                    << throw_stacktrace(back_trace)
                    << throw_thread_name(threads::get_thread_description(id))
                    << ::boost::throw_function(func.c_str()) 
                    << ::boost::throw_file(file.c_str())
                    << ::boost::throw_line(line));
        }
        else {
            throw boost::enable_current_exception(
                boost::enable_error_info(e) 
                    << throw_stacktrace(back_trace)
                    << ::boost::throw_function(func.c_str()) 
                    << ::boost::throw_file(file.c_str())
                    << ::boost::throw_line(line));
        }
    }

    template HPX_EXPORT void throw_exception(hpx::exception const&, 
        std::string const&, std::string const&, int, std::string const&);

    template HPX_EXPORT void throw_exception(boost::system::system_error const&, 
        std::string const&, std::string const&, int, std::string const&);

    template HPX_EXPORT void throw_exception(std::exception const&, 
        std::string const&, std::string const&, int, std::string const&);
    template HPX_EXPORT void throw_exception(std::bad_exception const&,
        std::string const&, std::string const&, int, std::string const&);
#ifndef BOOST_NO_TYPEID
    template HPX_EXPORT void throw_exception(std::bad_typeid const&, 
        std::string const&, std::string const&, int, std::string const&);
    template HPX_EXPORT void throw_exception(std::bad_cast const&, 
        std::string const&, std::string const&, int, std::string const&);
#endif
    template HPX_EXPORT void throw_exception(std::bad_alloc const&, 
        std::string const&, std::string const&, int, std::string const&);
    template HPX_EXPORT void throw_exception(std::logic_error const&, 
        std::string const&, std::string const&, int, std::string const&);
    template HPX_EXPORT void throw_exception(std::runtime_error const&,
        std::string const&, std::string const&, int, std::string const&);
    template HPX_EXPORT void throw_exception(std::out_of_range const&, 
        std::string const&, std::string const&, int, std::string const&);
    template HPX_EXPORT void throw_exception(std::invalid_argument const&, 
        std::string const&, std::string const&, int, std::string const&);

#else // HPX_STACKTRACES != 0

    ///////////////////////////////////////////////////////////////////////////
    template <typename Exception>
    HPX_EXPORT void throw_exception(Exception const& e, std::string const& func, 
        std::string const& file, int line)
    {
        threads::thread_self* self = threads::get_self_ptr();
        if (NULL != self) {
            threads::thread_id_type id = self->get_thread_id();
            throw boost::enable_current_exception(
                boost::enable_error_info(e) 
                    << throw_thread_name(threads::get_thread_description(id))
                    << ::boost::throw_function(func.c_str()) 
                    << ::boost::throw_file(file.c_str())
                    << ::boost::throw_line(line));
        }
        else {
            throw boost::enable_current_exception(
                boost::enable_error_info(e) 
                    << ::boost::throw_function(func.c_str()) 
                    << ::boost::throw_file(file.c_str())
                    << ::boost::throw_line(line));
        }
    }

    template HPX_EXPORT void throw_exception(hpx::exception const&, 
        std::string const&, std::string const&, int);

    template HPX_EXPORT void throw_exception(boost::system::system_error const&, 
        std::string const&, std::string const&, int);

    template HPX_EXPORT void throw_exception(std::exception const&, 
        std::string const&, std::string const&, int);
    template HPX_EXPORT void throw_exception(std::bad_exception const&, 
        std::string const&, std::string const&, int);
#ifndef BOOST_NO_TYPEID
    template HPX_EXPORT void throw_exception(std::bad_typeid const&, 
        std::string const&, std::string const&, int);
    template HPX_EXPORT void throw_exception(std::bad_cast const&, 
        std::string const&, std::string const&, int);
#endif
    template HPX_EXPORT void throw_exception(std::bad_alloc const&, 
        std::string const&, std::string const&, int);
    template HPX_EXPORT void throw_exception(std::logic_error const&, 
        std::string const&, std::string const&, int);
    template HPX_EXPORT void throw_exception(std::runtime_error const&, 
        std::string const&, std::string const&, int);
    template HPX_EXPORT void throw_exception(std::out_of_range const&, 
        std::string const&, std::string const&, int);
    template HPX_EXPORT void throw_exception(std::invalid_argument const&, 
        std::string const&, std::string const&, int);
#endif

    ///////////////////////////////////////////////////////////////////////////
    void assertion_failed(char const* expr, char const* function,
        char const* file, long line)
    {
        assertion_failed_msg(expr, expr, function, file, line);
    }

    void assertion_failed_msg(char const* msg, char const* expr,
        char const* function, char const* file, long line)
    {
        try {
            boost::filesystem::path p(hpx::util::create_path(file));
            std::string str("assertion '" + std::string(msg) + "' failed (");
            str += std::string(expr) + ")";
            hpx::detail::throw_exception(
                hpx::exception(hpx::assertion_failure, str), 
                function, p.string(), line);
        }

        catch (...) {
            // If the runtime pointer is available, we can safely get the prefix
            // of this locality. If it's not available, then just terminate.
            if (NULL != get_runtime_ptr())  {
                get_runtime().report_error(boost::current_exception());
            }
            else {
                std::cerr << "Runtime is not available, reporting error locally\n"
                          << boost::diagnostic_information(boost::current_exception()); 
            }

            std::terminate();
        }
    }
}}

