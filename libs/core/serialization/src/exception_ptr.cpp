//  Copyright (c)      2020 ETH Zurich
//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/serialization/exception_ptr.hpp>
#include <hpx/serialization/serialize.hpp>

#if BOOST_ASIO_HAS_BOOST_THROW_EXCEPTION != 0
#include <boost/exception/diagnostic_information.hpp>
#include <boost/exception/exception.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <typeinfo>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization {
    namespace detail {
        ///////////////////////////////////////////////////////////////////////////
        // TODO: This is not scalable, and painful to update.
        void save(output_archive& ar, std::exception_ptr const& ep,
            unsigned int version)
        {
            hpx::util::exception_type type(hpx::util::unknown_exception);
            std::string what;
            int err_value = hpx::success;
            std::string err_message;

            std::string throw_function_;
            std::string throw_file_;
            long throw_line_ = 0;

            // retrieve information related to exception_info
            try
            {
                std::rethrow_exception(ep);
            }
            catch (exception_info const& xi)
            {
                std::string const* function =
                    xi.get<hpx::detail::throw_function>();
                if (function)
                    throw_function_ = *function;

                std::string const* file = xi.get<hpx::detail::throw_file>();
                if (file)
                    throw_file_ = *file;

                long const* line = xi.get<hpx::detail::throw_line>();
                if (line)
                    throw_line_ = *line;
            }

            // figure out concrete underlying exception type
            try
            {
                std::rethrow_exception(ep);
            }
            catch (hpx::thread_interrupted const&)
            {
                type = hpx::util::hpx_thread_interrupted_exception;
                what = "hpx::thread_interrupted";
                err_value = hpx::thread_cancelled;
            }
            catch (hpx::exception const& e)
            {
                type = hpx::util::hpx_exception;
                what = e.what();
                err_value = e.get_error();
            }
            catch (boost::system::system_error const& e)
            {
                type = hpx::util::boost_system_error;
                what = e.what();
                err_value = e.code().value();
                err_message = e.code().message();
            }
            catch (std::runtime_error const& e)
            {
                type = hpx::util::std_runtime_error;
                what = e.what();
            }
            catch (std::invalid_argument const& e)
            {
                type = hpx::util::std_invalid_argument;
                what = e.what();
            }
            catch (std::out_of_range const& e)
            {
                type = hpx::util::std_out_of_range;
                what = e.what();
            }
            catch (std::logic_error const& e)
            {
                type = hpx::util::std_logic_error;
                what = e.what();
            }
            catch (std::bad_alloc const& e)
            {
                type = hpx::util::std_bad_alloc;
                what = e.what();
            }
            catch (std::bad_cast const& e)
            {
                type = hpx::util::std_bad_cast;
                what = e.what();
            }
            catch (std::bad_typeid const& e)
            {
                type = hpx::util::std_bad_typeid;
                what = e.what();
            }
            catch (std::bad_exception const& e)
            {
                type = hpx::util::std_bad_exception;
                what = e.what();
            }
            catch (std::exception const& e)
            {
                type = hpx::util::std_exception;
                what = e.what();
            }
#if BOOST_ASIO_HAS_BOOST_THROW_EXCEPTION != 0
            catch (boost::exception const& e)
            {
                type = hpx::util::boost_exception;
                what = boost::diagnostic_information(e);
            }
#endif
            catch (...)
            {
                type = hpx::util::unknown_exception;
                what = "unknown exception";
            }

            // clang-format off
            ar & type & what & throw_function_ & throw_file_ & throw_line_;
            // clang-format on

            if (hpx::util::hpx_exception == type)
            {
                // clang-format off
                ar & err_value;
                // clang-format on
            }
            else if (hpx::util::boost_system_error == type)
            {
                // clang-format off
                ar & err_value & err_message;
                // clang-format on
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // TODO: This is not scalable, and painful to update.
        void load(
            input_archive& ar, std::exception_ptr& e, unsigned int version)
        {
            hpx::util::exception_type type(hpx::util::unknown_exception);
            std::string what;
            int err_value = hpx::success;
            std::string err_message;

            std::string throw_function_;
            std::string throw_file_;
            int throw_line_ = 0;

            // clang-format off
            ar & type & what & throw_function_ & throw_file_ & throw_line_;
            // clang-format on

            if (hpx::util::hpx_exception == type)
            {
                // clang-format off
                ar & err_value;
                // clang-format on
            }
            else if (hpx::util::boost_system_error == type)
            {
                // clang-format off
                ar & err_value& err_message;
                // clang-format on
            }

            switch (type)
            {
            default:
            case hpx::util::std_exception:
            case hpx::util::unknown_exception:
                e = hpx::detail::get_exception(hpx::detail::std_exception(what),
                    throw_function_, throw_file_, throw_line_);
                break;

            // standard exceptions
            case hpx::util::std_runtime_error:
                e = hpx::detail::get_exception(std::runtime_error(what),
                    throw_function_, throw_file_, throw_line_);
                break;

            case hpx::util::std_invalid_argument:
                e = hpx::detail::get_exception(std::invalid_argument(what),
                    throw_function_, throw_file_, throw_line_);
                break;

            case hpx::util::std_out_of_range:
                e = hpx::detail::get_exception(std::out_of_range(what),
                    throw_function_, throw_file_, throw_line_);
                break;

            case hpx::util::std_logic_error:
                e = hpx::detail::get_exception(std::logic_error(what),
                    throw_function_, throw_file_, throw_line_);
                break;

            case hpx::util::std_bad_alloc:
                e = hpx::detail::get_exception(hpx::detail::bad_alloc(what),
                    throw_function_, throw_file_, throw_line_);
                break;

            case hpx::util::std_bad_cast:
                e = hpx::detail::get_exception(hpx::detail::bad_cast(what),
                    throw_function_, throw_file_, throw_line_);
                break;

            case hpx::util::std_bad_typeid:
                e = hpx::detail::get_exception(hpx::detail::bad_typeid(what),
                    throw_function_, throw_file_, throw_line_);
                break;
            case hpx::util::std_bad_exception:
                e = hpx::detail::get_exception(hpx::detail::bad_exception(what),
                    throw_function_, throw_file_, throw_line_);
                break;

#if BOOST_ASIO_HAS_BOOST_THROW_EXCEPTION != 0
            // boost exceptions
            case hpx::util::boost_exception:
                HPX_ASSERT(false);    // shouldn't happen
                break;
#endif

            // boost::system::system_error
            case hpx::util::boost_system_error:
                e = hpx::detail::get_exception(
                    boost::system::system_error(err_value,
#if BOOST_VERSION < 106600 && !defined(BOOST_SYSTEM_NO_DEPRECATED)
                        boost::system::get_system_category()
#else
                        boost::system::system_category()
#endif
                            ,
                        err_message),
                    throw_function_, throw_file_, throw_line_);
                break;

            // hpx::exception
            case hpx::util::hpx_exception:
                e = hpx::detail::get_exception(
                    hpx::exception(
                        static_cast<hpx::error>(err_value), what, hpx::rethrow),
                    throw_function_, throw_file_, throw_line_);
                break;

            // hpx::thread_interrupted
            case hpx::util::hpx_thread_interrupted_exception:
                e = hpx::detail::construct_lightweight_exception(
                    hpx::thread_interrupted());
                break;
            }
        }

        save_custom_exception_handler_type& get_save_custom_exception_handler()
        {
            static save_custom_exception_handler_type f = save;
            return f;
        }

        HPX_CORE_EXPORT void set_save_custom_exception_handler(
            save_custom_exception_handler_type f)
        {
            get_save_custom_exception_handler() = f;
        }

        load_custom_exception_handler_type& get_load_custom_exception_handler()
        {
            static load_custom_exception_handler_type f = load;
            return f;
        }

        HPX_CORE_EXPORT void set_load_custom_exception_handler(
            load_custom_exception_handler_type f)
        {
            get_load_custom_exception_handler() = f;
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void save(Archive& ar, std::exception_ptr const& ep, unsigned int version)
    {
        if (detail::get_save_custom_exception_handler())
        {
            detail::get_save_custom_exception_handler()(ar, ep, version);
        }
        else
        {
            HPX_THROW_EXCEPTION(invalid_status, "hpx::serialization::save",
                "Attempted to save a std::exception_ptr, but there is no "
                "handler installed. Set one with "
                "hpx::serialization::detail::set_save_custom_exception_"
                "handler.");
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void load(Archive& ar, std::exception_ptr& ep, unsigned int version)
    {
        if (detail::get_load_custom_exception_handler())
        {
            detail::get_load_custom_exception_handler()(ar, ep, version);
        }
        else
        {
            HPX_THROW_EXCEPTION(invalid_status, "hpx::serialization::load",
                "Attempted to load a std::exception_ptr, but there is no "
                "handler installed. Set one with "
                "hpx::serialization::detail::set_load_custom_exception_"
                "handler.");
        }
    }

    template HPX_CORE_EXPORT void save(hpx::serialization::output_archive&,
        std::exception_ptr const&, unsigned int);

    template HPX_CORE_EXPORT void load(
        hpx::serialization::input_archive&, std::exception_ptr&, unsigned int);
}}    // namespace hpx::serialization
