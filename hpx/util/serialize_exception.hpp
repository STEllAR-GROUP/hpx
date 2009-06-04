//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SERIALIZE_EXCEPTION_JAN_23_2009_0108PM)
#define HPX_UTIL_SERIALIZE_EXCEPTION_JAN_23_2009_0108PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>

#include <boost/config.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_free.hpp>

#include <stdexcept>
#ifndef BOOST_NO_TYPEID
#include <typeinfo>
#endif

namespace hpx { namespace util
{
    enum exception_type
    {
        // unknown exception
        unknown_exception = 0,

        // standard exceptions
        std_runtime_error = 1,
        std_invalid_argument = 2,
        std_out_of_range = 3,
        std_logic_error = 4,
        std_bad_alloc = 5,
#ifndef BOOST_NO_TYPEID
        std_bad_cast = 6,
        std_bad_typeid = 7,
#endif
        std_bad_exception = 8,
        std_exception = 9,

        // boost exceptions
        boost_exception = 10,

        // boost::system::system_error
        boost_system_error = 11,

        // hpx::exception
        hpx_exception = 12
    };

}}  // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void save(Archive& ar, boost::exception_ptr const& ep, unsigned int)
    {
        hpx::util::exception_type type;
        std::string what;
        int err_value;
        std::string err_message;

        std::string throw_function;
        std::string throw_file;
        int throw_line = -1;

        // retrieve information related to boost::exception
        try {
            boost::rethrow_exception(ep);
        }
        catch (boost::exception const& e) {
            boost::shared_ptr<char const* const> func(
                boost::get_error_info<boost::throw_function>(e));
            if (func) {
                throw_function = *func;
            }
            else {
                boost::shared_ptr<std::string const> func(
                    boost::get_error_info<hpx::detail::throw_function>(e));
                if (func)
                    throw_function = *func;
            }

            boost::shared_ptr<std::string const> file(
                boost::get_error_info<hpx::detail::throw_file>(e));
            if (file)
                throw_file = *file;

            boost::shared_ptr<int const> line(
                boost::get_error_info<hpx::detail::throw_line>(e));
            if (line)
                throw_line = *line;
        }

        // figure out concrete underlying exception type
        try {
            boost::rethrow_exception(ep);
        }
        catch (hpx::exception const& e) {
            type = hpx::util::hpx_exception;
            what = e.what();
            err_value = e.get_error();
        }
        catch (boost::system::system_error const& e) {
            type = hpx::util::boost_system_error;
            what = e.what();
            err_value = e.code().value();
            err_message = e.code().message();
        }
        catch (std::runtime_error const& e) {
            type = hpx::util::std_runtime_error;
            what = e.what();
        }
        catch (std::invalid_argument const& e) {
            type = hpx::util::std_invalid_argument;
            what = e.what();
        }
        catch (std::out_of_range const& e) {
            type = hpx::util::std_out_of_range;
            what = e.what();
        }
        catch (std::logic_error const& e) {
            type = hpx::util::std_logic_error;
            what = e.what();
        }
        catch (std::bad_alloc const& e) {
            type = hpx::util::std_bad_alloc;
            what = e.what();
        }
#ifndef BOOST_NO_TYPEID
        catch (std::bad_cast const& e) {
            type = hpx::util::std_bad_cast;
            what = e.what();
        }
        catch (std::bad_typeid const& e) {
            type = hpx::util::std_bad_typeid;
            what = e.what();
        }
#endif
        catch (std::bad_exception const& e) {
            type = hpx::util::std_bad_exception;
            what = e.what();
        }
        catch (std::exception const& e) {
            type = hpx::util::std_exception;
            what = e.what();
        }
        catch (boost::exception const& e) {
            type = hpx::util::boost_exception;
            what = boost::diagnostic_information(e);
        }
        catch (...) {
            type = hpx::util::unknown_exception;
            what = "unknown exception";
        }

        ar & type & what & throw_function & throw_file & throw_line;
        if (hpx::util::hpx_exception == type) {
            ar & err_value;
        }
        else if (hpx::util::boost_system_error == type) {
            ar & err_value & err_message;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void load(Archive& ar, boost::exception_ptr& e, unsigned int)
    {
        hpx::util::exception_type type;
        std::string what;
        int err_value;
        std::string err_message;

        std::string throw_function;
        std::string throw_file;
        int throw_line = 0;

        ar & type & what & throw_function & throw_file & throw_line;
        if (hpx::util::hpx_exception == type) {
            ar & err_value;
        }
        else if (hpx::util::boost_system_error == type) {
            ar & err_value & err_message;
        }

        try {
            switch (type) {
            case hpx::util::std_exception:
            case hpx::util::unknown_exception:
                hpx::detail::throw_exception(std::exception(),
                    throw_function, throw_file, throw_line);
                break;

            // standard exceptions
            case hpx::util::std_runtime_error:
                hpx::detail::throw_exception(std::runtime_error(what),
                    throw_function, throw_file, throw_line);
                break;

            case hpx::util::std_invalid_argument:
                hpx::detail::throw_exception(std::invalid_argument(what),
                    throw_function, throw_file, throw_line);
                break;

            case hpx::util::std_out_of_range:
                hpx::detail::throw_exception(std::out_of_range(what),
                    throw_function, throw_file, throw_line);
                break;

            case hpx::util::std_logic_error:
                hpx::detail::throw_exception(std::logic_error(what),
                    throw_function, throw_file, throw_line);
                break;

            case hpx::util::std_bad_alloc:
                hpx::detail::throw_exception(std::bad_alloc(),
                    throw_function, throw_file, throw_line);
                break;

#ifndef BOOST_NO_TYPEID
            case hpx::util::std_bad_cast:
                hpx::detail::throw_exception(std::bad_cast(),
                    throw_function, throw_file, throw_line);
                break;

            case hpx::util::std_bad_typeid:
                hpx::detail::throw_exception(std::bad_typeid(),
                    throw_function, throw_file, throw_line);
                break;
#endif
            case hpx::util::std_bad_exception:
                hpx::detail::throw_exception(std::bad_exception(),
                    throw_function, throw_file, throw_line);
                break;

            // boost exceptions
            case hpx::util::boost_exception:
                BOOST_ASSERT(false);    // shouldn't happen
                break;

            // boost::system::system_error
            case hpx::util::boost_system_error:
                hpx::detail::throw_exception(
                    boost::system::system_error(err_value, 
                        boost::system::get_system_category(), err_message),
                    throw_function, throw_file, throw_line);
                break;

            // hpx::exception
            case hpx::util::hpx_exception:
                hpx::detail::throw_exception(
                    hpx::exception((hpx::error)err_value, what, hpx::rethrow),
                    throw_function, throw_file, throw_line);
                break;
            }
        }
        catch (...) {
            e = boost::current_exception();
        }
    }

}}

BOOST_SERIALIZATION_SPLIT_FREE(boost::exception_ptr);

#endif
