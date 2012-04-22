//  Copyright (c) 2007-2012 Hartmut Kaiser
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
    // TODO: This is not scalable, and painful to update.
    template <typename Archive>
    void save(Archive& ar, boost::exception_ptr const& ep, unsigned int)
    {
        hpx::util::exception_type type(hpx::util::unknown_exception);
        std::string what;
        int err_value = hpx::success;
        std::string err_message;

        boost::uint32_t throw_locality_ = 0;
        std::string throw_hostname_;
        boost::int64_t throw_pid_ = -1;
        std::size_t throw_shepherd_ = 0;
        std::size_t throw_thread_id_ = 0;
        std::string throw_thread_name_;
        std::string throw_function_;
        std::string throw_file_;
        std::string back_trace_;
        int throw_line_ = 0;

        // retrieve information related to boost::exception
        try {
            boost::rethrow_exception(ep);
        }
        catch (boost::exception const& e) {
            char const* const* func =
                boost::get_error_info<boost::throw_function>(e);
            if (func) {
                throw_function_ = *func;
            }
            else {
                std::string const* s =
                    boost::get_error_info<hpx::throw_function>(e);
                if (s)
                    throw_function_ = *s;
            }

            char const* const* file =
                boost::get_error_info<boost::throw_file>(e);
            if (file) {
                throw_file_ = *file;
            }
            else {
                std::string const* s =
                    boost::get_error_info<hpx::throw_file>(e);
                if (s)
                    throw_file_ = *s;
            }

            int const* line =
                boost::get_error_info<boost::throw_line>(e);
            if (line) {
                throw_line_ = *line;
            }

            boost::uint32_t const* locality =
                boost::get_error_info<hpx::throw_locality>(e);
            if (locality) {
                throw_locality_ = *locality;
            }

            std::string const* hostname_ =
                boost::get_error_info<hpx::throw_hostname>(e);
            if (hostname_) {
                throw_hostname_ = *hostname_;
            }

            boost::int64_t const* pid_ =
                boost::get_error_info<hpx::throw_pid>(e);
            if (pid_) {
                throw_pid_ = *pid_;
            }

            boost::int64_t const* shepherd =
                boost::get_error_info<hpx::throw_shepherd>(e);
            if (shepherd) {
                throw_shepherd_ = *shepherd;
            }

            std::size_t const* thread_id =
                boost::get_error_info<hpx::throw_thread_id>(e);
            if (thread_id) {
                throw_thread_id_ = *thread_id;
            }

            std::string const* thread_name =
                boost::get_error_info<hpx::throw_thread_name>(e);
            if (thread_name) {
                throw_thread_name_ = *thread_name;
            }

#if defined(HPX_HAVE_STACKTRACES)
            std::string const* back_trace =
                boost::get_error_info<hpx::throw_stacktrace>(e);
            if (back_trace)
                back_trace_ = *back_trace;
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

        ar & type & what & throw_function_ & throw_file_ & throw_line_
           & throw_locality_ & throw_hostname_ & throw_pid_ & throw_shepherd_
           & throw_thread_id_ & throw_thread_name_ & back_trace_;
        if (hpx::util::hpx_exception == type) {
            ar & err_value;
        }
        else if (hpx::util::boost_system_error == type) {
            ar & err_value & err_message;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // TODO: This is not scalable, and painful to update.
    template <typename Archive>
    void load(Archive& ar, boost::exception_ptr& e, unsigned int)
    {
        hpx::util::exception_type type(hpx::util::unknown_exception);
        std::string what;
        int err_value = hpx::success;
        std::string err_message;

        boost::uint32_t throw_locality_ = 0;
        std::string throw_hostname_;
        boost::int64_t throw_pid_ = -1;
        std::size_t throw_shepherd_ = 0;
        std::size_t throw_thread_id_ = 0;
        std::string throw_thread_name_;
        std::string throw_function_;
        std::string throw_file_;
        std::string back_trace_;
        int throw_line_ = 0;

        ar & type & what & throw_function_ & throw_file_ & throw_line_
           & throw_locality_ & throw_hostname_ & throw_pid_ & throw_shepherd_
           & throw_thread_id_ & throw_thread_name_ & back_trace_;
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
                hpx::detail::rethrow_exception(hpx::detail::std_exception(what),
                    throw_function_, throw_file_, throw_line_, back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_);
                break;

            // standard exceptions
            case hpx::util::std_runtime_error:
                hpx::detail::rethrow_exception(std::runtime_error(what),
                    throw_function_, throw_file_, throw_line_, back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_);
                break;

            case hpx::util::std_invalid_argument:
                hpx::detail::rethrow_exception(std::invalid_argument(what),
                    throw_function_, throw_file_, throw_line_, back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_);
                break;

            case hpx::util::std_out_of_range:
                hpx::detail::rethrow_exception(std::out_of_range(what),
                    throw_function_, throw_file_, throw_line_, back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_);
                break;

            case hpx::util::std_logic_error:
                hpx::detail::rethrow_exception(std::logic_error(what),
                    throw_function_, throw_file_, throw_line_, back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_);
                break;

            case hpx::util::std_bad_alloc:
                hpx::detail::rethrow_exception(hpx::detail::bad_alloc(what),
                    throw_function_, throw_file_, throw_line_, back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_);
                break;

#ifndef BOOST_NO_TYPEID
            case hpx::util::std_bad_cast:
                hpx::detail::rethrow_exception(hpx::detail::bad_cast(what),
                    throw_function_, throw_file_, throw_line_, back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_);
                break;

            case hpx::util::std_bad_typeid:
                hpx::detail::rethrow_exception(hpx::detail::bad_typeid(what),
                    throw_function_, throw_file_, throw_line_, back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_);
                break;
#endif
            case hpx::util::std_bad_exception:
                hpx::detail::rethrow_exception(hpx::detail::bad_exception(what),
                    throw_function_, throw_file_, throw_line_, back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_);
                break;

            // boost exceptions
            case hpx::util::boost_exception:
                BOOST_ASSERT(false);    // shouldn't happen
                break;

            // boost::system::system_error
            case hpx::util::boost_system_error:
                hpx::detail::rethrow_exception(
                    boost::system::system_error(err_value,
                        boost::system::get_system_category(), err_message),
                    throw_function_, throw_file_, throw_line_, back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_);
                break;

            // hpx::exception
            case hpx::util::hpx_exception:
                hpx::detail::rethrow_exception(
                    hpx::exception(static_cast<hpx::error>(err_value),
                        what, hpx::rethrow),
                    throw_function_, throw_file_, throw_line_, back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_);
                break;
            }
        }
        catch (...) {
            e = boost::current_exception();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // explicit instantiation for the correct archive types
    template HPX_EXPORT void
    save(hpx::util::portable_binary_oarchive&, boost::exception_ptr const&,
        unsigned int);

    template HPX_EXPORT void
    load(hpx::util::portable_binary_iarchive&, boost::exception_ptr&,
        unsigned int);
}}

