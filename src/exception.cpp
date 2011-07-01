//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#if defined(HPX_STACKTRACES)
    #include <boost/backtrace.hpp>
#endif
#include <stdexcept>

namespace boost
{

HPX_EXPORT void assertion_failed(
    char const* expr
  , char const* function
  , char const* file
  , long line
) {
    boost::filesystem::path p(hpx::util::create_path(file));
    hpx::exception e(hpx::assertion_failed, expr);
    hpx::detail::throw_exception(e, function, p.string(), line);
}

HPX_EXPORT void assertion_failed_msg(
    char const* msg
  , char const* expr
  , char const* function
  , char const* file
  , long line
) {
    boost::filesystem::path p(hpx::util::create_path(file));
    hpx::exception e(hpx::assertion_failed, msg);
    hpx::detail::throw_exception(e, function, p.string(), line);
}

}

namespace hpx { namespace detail
{
    template <typename Exception>
    HPX_EXPORT void throw_exception(Exception const& e, char const* func, 
        std::string const& file, int line)
    {
        threads::thread_self* self = threads::get_self_ptr();
        if (NULL != self) {
            threads::thread_id_type id = self->get_thread_id();
            throw boost::enable_current_exception(
                boost::enable_error_info(e) 
#if defined(HPX_STACKTRACES)
                    << throw_stacktrace(boost::trace())
#endif
                    << boost::throw_function(func) 
                    << throw_thread_name(threads::get_thread_description(id))
                    << throw_file(file) << throw_line(line));
        }
        else {
            throw boost::enable_current_exception(
                boost::enable_error_info(e) 
#if defined(HPX_STACKTRACES)
                    << throw_stacktrace(boost::trace())
#endif
                    << boost::throw_function(func) 
                    << throw_file(file) << throw_line(line));
        }
    }

    template HPX_EXPORT void throw_exception(hpx::exception const&, 
        char const*, std::string const&, int);

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
#if defined(HPX_STACKTRACES)
                    << throw_stacktrace(boost::trace())
#endif
                    << throw_function(func) 
                    << throw_thread_name(threads::get_thread_description(id))
                    << throw_file(file) << throw_line(line));
        }
        else {
            throw boost::enable_current_exception(
                boost::enable_error_info(e) 
                    << throw_stacktrace(boost::trace())
                    << throw_function(func) 
                    << throw_file(file) << throw_line(line));
        }
    }

    template HPX_EXPORT void throw_exception<hpx::exception>(hpx::exception const&, 
        std::string const&, std::string const&, int);

    template HPX_EXPORT void throw_exception(boost::system::system_error const&, 
        std::string const&, std::string const&, int);

    template HPX_EXPORT void throw_exception(std::exception const&, 
        std::string const&, std::string const&, int);
    template HPX_EXPORT void throw_exception(std::bad_exception const&, 
        std::string const&, std::string const&, int);
    template HPX_EXPORT void throw_exception(std::bad_typeid const&, 
        std::string const&, std::string const&, int);
    template HPX_EXPORT void throw_exception(std::bad_cast const&, 
        std::string const&, std::string const&, int);
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
}}

