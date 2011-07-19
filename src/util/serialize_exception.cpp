//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/serialize_exception.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/version.hpp>
#include <boost/config.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_free.hpp>

#include <stdexcept>
#ifndef BOOST_NO_TYPEID
#include <typeinfo>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void save(Archive& ar, boost::exception_ptr const& ep, unsigned int)
    {
        hpx::util::exception_type type(hpx::util::unknown_exception);
        std::string what;
        int err_value(0);
        std::string err_message;

        std::string throw_function_;
        std::string throw_file_;
        int throw_line_ = -1;

#if HPX_STACKTRACES != 0
        std::string back_trace_;
#endif

        // retrieve information related to boost::exception
        try {
            boost::rethrow_exception(ep);
        }
        catch (boost::exception const& e) {
#if BOOST_VERSION >= 103900
            char const* const* func =
                boost::get_error_info<boost::throw_function>(e);
            if (func) {
                throw_function_ = *func;
            }
            else {
                std::string const* s = 
                    boost::get_error_info<hpx::detail::throw_function>(e);
                if (s)
                    throw_function_ = *s;
            }

            std::string const* file = 
                boost::get_error_info<hpx::detail::throw_file>(e);
            if (file)
                throw_file_ = *file;

            int const* line = 
                boost::get_error_info<hpx::detail::throw_line>(e);
            if (line)
                throw_line_ = *line;

#if HPX_STACKTRACES != 0
            std::string const* back_trace =
                boost::get_error_info<hpx::detail::throw_stacktrace>(e);
            if (back_trace)
                back_trace_ = *back_trace;
#endif
#else
            boost::shared_ptr<char const* const> func(
                boost::get_error_info<boost::throw_function>(e));
            if (func) {
                throw_function_ = *func;
            }
            else {
                boost::shared_ptr<std::string const> s(
                    boost::get_error_info<hpx::detail::throw_function>(e));
                if (s)
                    throw_function_ = *s;
            }

            boost::shared_ptr<std::string const> file(
                boost::get_error_info<hpx::detail::throw_file>(e));
            if (file)
                throw_file_ = *file;

            boost::shared_ptr<int const> line(
                boost::get_error_info<hpx::detail::throw_line>(e));
            if (line)
                throw_line_ = *line;

#if HPX_STACKTRACES != 0
            boost::shared_ptr<std::string const> back_trace(
                boost::get_error_info<hpx::detail::throw_stacktrace>(e));
            if (back_trace)
                back_trace_ = *back_trace;
#endif
#endif
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

        ar & type & what & throw_function_ & throw_file_ & throw_line_;
        if (hpx::util::hpx_exception == type) {
            ar & err_value;
        }
        else if (hpx::util::boost_system_error == type) {
            ar & err_value & err_message;
        }

#if HPX_STACKTRACES != 0
        ar & back_trace_;
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void load(Archive& ar, boost::exception_ptr& e, unsigned int)
    {
        hpx::util::exception_type type(hpx::util::unknown_exception);
        std::string what;
        int err_value(0);
        std::string err_message;

        std::string throw_function_;
        std::string throw_file_;
        std::string back_trace_;
        int throw_line_ = 0;

        ar & type & what & throw_function_ & throw_file_ & throw_line_;
        if (hpx::util::hpx_exception == type) {
            ar & err_value;
        }
        else if (hpx::util::boost_system_error == type) {
            ar & err_value & err_message;
        }
#if HPX_STACKTRACES != 0
        ar & back_trace_;
#endif

        try {
            switch (type) {
            case hpx::util::std_exception:
            case hpx::util::unknown_exception:
                hpx::detail::throw_exception(std::exception(),
                    throw_function_, throw_file_, throw_line_, back_trace_);
                break;

            // standard exceptions
            case hpx::util::std_runtime_error:
                hpx::detail::throw_exception(std::runtime_error(what),
                    throw_function_, throw_file_, throw_line_, back_trace_);
                break;

            case hpx::util::std_invalid_argument:
                hpx::detail::throw_exception(std::invalid_argument(what),
                    throw_function_, throw_file_, throw_line_, back_trace_);
                break;

            case hpx::util::std_out_of_range:
                hpx::detail::throw_exception(std::out_of_range(what),
                    throw_function_, throw_file_, throw_line_, back_trace_);
                break;

            case hpx::util::std_logic_error:
                hpx::detail::throw_exception(std::logic_error(what),
                    throw_function_, throw_file_, throw_line_, back_trace_);
                break;

            case hpx::util::std_bad_alloc:
                hpx::detail::throw_exception(std::bad_alloc(),
                    throw_function_, throw_file_, throw_line_, back_trace_);
                break;

#ifndef BOOST_NO_TYPEID
            case hpx::util::std_bad_cast:
                hpx::detail::throw_exception(std::bad_cast(),
                    throw_function_, throw_file_, throw_line_, back_trace_);
                break;

            case hpx::util::std_bad_typeid:
                hpx::detail::throw_exception(std::bad_typeid(),
                    throw_function_, throw_file_, throw_line_, back_trace_);
                break;
#endif
            case hpx::util::std_bad_exception:
                hpx::detail::throw_exception(std::bad_exception(),
                    throw_function_, throw_file_, throw_line_, back_trace_);
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
                    throw_function_, throw_file_, throw_line_, back_trace_);
                break;

            // hpx::exception
            case hpx::util::hpx_exception:
                hpx::detail::throw_exception(
                    hpx::exception((hpx::error)err_value, what, hpx::rethrow),
                    throw_function_, throw_file_, throw_line_, back_trace_);
                break;
            }
        }
        catch (...) {
            e = boost::current_exception();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // explicit instantiation for the correct archive types
#if HPX_USE_PORTABLE_ARCHIVES != 0
    template HPX_EXPORT void 
    save(hpx::util::portable_binary_oarchive&, boost::exception_ptr const&, 
        unsigned int);

    template HPX_EXPORT void 
    load(hpx::util::portable_binary_iarchive&, boost::exception_ptr&, 
        unsigned int);
#else
    template HPX_EXPORT void 
    save(boost::archive::binary_oarchive&, boost::exception_ptr const&, 
        unsigned int);

    template HPX_EXPORT void 
    load(boost::archive::binary_iarchive&, boost::exception_ptr&, 
        unsigned int);
#endif
}}

