//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/exception.hpp>
#include <hpx/exception_info.hpp>
#include <hpx/util/serialize_exception.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

#include <boost/exception/diagnostic_information.hpp>
#include <boost/exception/exception.hpp>
#include <boost/version.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <typeinfo>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    // TODO: This is not scalable, and painful to update.
    template <typename Archive>
    void save(Archive& ar, std::exception_ptr const& ep, unsigned int)
    {
        hpx::util::exception_type type(hpx::util::unknown_exception);
        std::string what;
        int err_value = hpx::success;
        std::string err_message;

        std::uint32_t throw_locality_ = 0;
        std::string throw_hostname_;
        std::int64_t throw_pid_ = -1;
        std::size_t throw_shepherd_ = 0;
        std::size_t throw_thread_id_ = 0;
        std::string throw_thread_name_;
        std::string throw_function_;
        std::string throw_file_;
        std::string throw_back_trace_;
        int throw_line_ = 0;
        std::string throw_env_;
        std::string throw_config_;
        std::string throw_state_;
        std::string throw_auxinfo_;

        // retrieve information related to exception_info
        try {
            std::rethrow_exception(ep);
        }
        catch (exception_info const& xi) {
            std::string const* function =
                xi.get<hpx::detail::throw_function>();
            if (function)
                throw_function_ = *function;

            std::string const* file =
                xi.get<hpx::detail::throw_file>();
            if (file)
                throw_file_ = *file;

            long const* line =
                xi.get<hpx::detail::throw_line>();
            if (line)
                throw_line_ = *line;

            std::uint32_t const* locality =
                xi.get<hpx::detail::throw_locality>();
            if (locality)
                throw_locality_ = *locality;

            std::string const* hostname_ =
                xi.get<hpx::detail::throw_hostname>();
            if (hostname_)
                throw_hostname_ = *hostname_;

            std::int64_t const* pid_ =
                xi.get<hpx::detail::throw_pid>();
            if (pid_)
                throw_pid_ = *pid_;

            std::size_t const* shepherd =
                xi.get<hpx::detail::throw_shepherd>();
            if (shepherd)
                throw_shepherd_ = *shepherd;

            std::size_t const* thread_id =
                xi.get<hpx::detail::throw_thread_id>();
            if (thread_id)
                throw_thread_id_ = *thread_id;

            std::string const* thread_name =
                xi.get<hpx::detail::throw_thread_name>();
            if (thread_name)
                throw_thread_name_ = *thread_name;

            std::string const* back_trace =
                xi.get<hpx::detail::throw_stacktrace>();
            if (back_trace)
                throw_back_trace_ = *back_trace;

            std::string const* env_ =
                xi.get<hpx::detail::throw_env>();
            if (env_)
                throw_env_ = *env_;

            std::string const* config_ =
                xi.get<hpx::detail::throw_config>();
            if (config_)
                throw_config_ = *config_;

            std::string const* state_ =
                xi.get<hpx::detail::throw_state>();
            if (state_)
                throw_state_ = *state_;

            std::string const* auxinfo_ =
                xi.get<hpx::detail::throw_auxinfo>();
            if (auxinfo_)
                throw_auxinfo_ = *auxinfo_;
        }

        // figure out concrete underlying exception type
        try {
            std::rethrow_exception(ep);
        }
        catch (hpx::thread_interrupted const&) {
            type = hpx::util::hpx_thread_interrupted_exception;
            what = "hpx::thread_interrupted";
            err_value = hpx::thread_cancelled;
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
        catch (std::bad_cast const& e) {
            type = hpx::util::std_bad_cast;
            what = e.what();
        }
        catch (std::bad_typeid const& e) {
            type = hpx::util::std_bad_typeid;
            what = e.what();
        }
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
           & throw_thread_id_ & throw_thread_name_ & throw_back_trace_
           & throw_env_ & throw_config_ & throw_state_ & throw_auxinfo_;

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
    void load(Archive& ar, std::exception_ptr& e, unsigned int)
    {
        hpx::util::exception_type type(hpx::util::unknown_exception);
        std::string what;
        int err_value = hpx::success;
        std::string err_message;

        std::uint32_t throw_locality_ = 0;
        std::string throw_hostname_;
        std::int64_t throw_pid_ = -1;
        std::size_t throw_shepherd_ = 0;
        std::size_t throw_thread_id_ = 0;
        std::string throw_thread_name_;
        std::string throw_function_;
        std::string throw_file_;
        std::string throw_back_trace_;
        int throw_line_ = 0;
        std::string throw_env_;
        std::string throw_config_;
        std::string throw_state_;
        std::string throw_auxinfo_;

        ar & type & what & throw_function_ & throw_file_ & throw_line_
           & throw_locality_ & throw_hostname_ & throw_pid_ & throw_shepherd_
           & throw_thread_id_ & throw_thread_name_ & throw_back_trace_
           & throw_env_ & throw_config_ & throw_state_ & throw_auxinfo_;

        if (hpx::util::hpx_exception == type) {
            ar & err_value;
        }
        else if (hpx::util::boost_system_error == type) {
            ar & err_value & err_message;
        }

        switch (type) {
        default:
        case hpx::util::std_exception:
        case hpx::util::unknown_exception:
            e = hpx::detail::construct_exception(
                    hpx::detail::std_exception(what),
                    throw_function_, throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_);
            break;

        // standard exceptions
        case hpx::util::std_runtime_error:
            e = hpx::detail::construct_exception(
                    std::runtime_error(what),
                    throw_function_, throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_);
            break;

        case hpx::util::std_invalid_argument:
            e = hpx::detail::construct_exception(
                    std::invalid_argument(what),
                    throw_function_, throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_);
            break;

        case hpx::util::std_out_of_range:
            e = hpx::detail::construct_exception(
                    std::out_of_range(what),
                    throw_function_, throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_);
            break;

        case hpx::util::std_logic_error:
            e = hpx::detail::construct_exception(
                    std::logic_error(what),
                    throw_function_, throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_);
            break;

        case hpx::util::std_bad_alloc:
            e = hpx::detail::construct_exception(
                    hpx::detail::bad_alloc(what),
                    throw_function_, throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_);
            break;

        case hpx::util::std_bad_cast:
            e = hpx::detail::construct_exception(
                    hpx::detail::bad_cast(what),
                    throw_function_, throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_);
            break;

        case hpx::util::std_bad_typeid:
            e = hpx::detail::construct_exception(hpx::detail::bad_typeid(what),
                    throw_function_, throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_);
            break;
        case hpx::util::std_bad_exception:
            e = hpx::detail::construct_exception(
                    hpx::detail::bad_exception(what),
                    throw_function_, throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_);
            break;

        // boost exceptions
        case hpx::util::boost_exception:
            HPX_ASSERT(false);    // shouldn't happen
            break;

        // boost::system::system_error
        case hpx::util::boost_system_error:
            e = hpx::detail::construct_exception(
                    boost::system::system_error(err_value,
#ifndef BOOST_SYSTEM_NO_DEPRECATED
                        boost::system::get_system_category()
#else
                        boost::system::system_category()
#endif
                      , err_message
                    )
                  , throw_function_, throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_);
            break;

        // hpx::exception
        case hpx::util::hpx_exception:
            e = hpx::detail::construct_exception(
                    hpx::exception(static_cast<hpx::error>(err_value),
                        what, hpx::rethrow),
                    throw_function_, throw_file_, throw_line_, throw_back_trace_,
                    throw_locality_, throw_hostname_, throw_pid_,
                    throw_shepherd_, throw_thread_id_, throw_thread_name_,
                    throw_env_, throw_config_, throw_state_, throw_auxinfo_);
            break;

        // hpx::thread_interrupted
        case hpx::util::hpx_thread_interrupted_exception:
            e = hpx::detail::construct_lightweight_exception(
                hpx::thread_interrupted());
            break;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // explicit instantiation for the correct archive types
    template HPX_EXPORT void
    save(hpx::serialization::output_archive&, std::exception_ptr const&,
        unsigned int);

    template HPX_EXPORT void
    load(hpx::serialization::input_archive&, std::exception_ptr&,
        unsigned int);
}}

