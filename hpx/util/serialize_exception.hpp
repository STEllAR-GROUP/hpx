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
    void save(Archive& ar, boost::exception_ptr const& e, unsigned int)
    {
        hpx::util::exception_type type;
        std::string what;
        int err_value;
        std::string err_message;

        try {
            boost::rethrow_exception(e);
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
        catch (boost::exception const& e) {
            type = hpx::util::boost_exception;
            what = boost::diagnostic_information(e);
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
        catch (...) {
            type = hpx::util::unknown_exception;
            what = "unknown exception";
        }

        ar & type & what;
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

        ar & type & what;
        if (hpx::util::hpx_exception == type) {
            ar & err_value;
        }
        else if (hpx::util::boost_system_error == type) {
            ar & err_value & err_message;
        }

        try {
            switch (type) {
            case hpx::util::unknown_exception:
                boost::throw_exception(std::exception());
                break;

            // standard exceptions
            case hpx::util::std_runtime_error:
                boost::throw_exception(std::runtime_error(what));
                break;

            case hpx::util::std_invalid_argument:
                boost::throw_exception(std::invalid_argument(what));
                break;

            case hpx::util::std_out_of_range:
                boost::throw_exception(std::out_of_range(what));
                break;

            case hpx::util::std_logic_error:
                boost::throw_exception(std::logic_error(what));
                break;

            case hpx::util::std_bad_alloc:
                boost::throw_exception(std::bad_alloc());
                break;

#ifndef BOOST_NO_TYPEID
            case hpx::util::std_bad_cast:
                boost::throw_exception(std::bad_cast());
                break;

            case hpx::util::std_bad_typeid:
                boost::throw_exception(std::bad_typeid());
                break;
#endif
            case hpx::util::std_bad_exception:
                boost::throw_exception(std::bad_exception());
                break;

            case hpx::util::std_exception:
                boost::throw_exception(std::exception());
                break;

            // boost exceptions
            case hpx::util::boost_exception:
                break;

            // boost::system::system_error
            case hpx::util::boost_system_error:
                boost::throw_exception(boost::system::system_error(err_value, 
                    boost::system::get_system_category(), err_message));
                break;

            // hpx::exception
            case hpx::util::hpx_exception:
                HPX_RETHROW_EXCEPTION(err_value, "load(exception_ptr)", what);
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
