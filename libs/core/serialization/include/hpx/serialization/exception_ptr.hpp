//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <exception>
#include <functional>

namespace hpx::util {

    enum class exception_type
    {
        // unknown exception
        unknown_exception = 0,

        // standard exceptions
        std_runtime_error = 1,
        std_invalid_argument = 2,
        std_out_of_range = 3,
        std_logic_error = 4,
        std_bad_alloc = 5,
        std_bad_cast = 6,
        std_bad_typeid = 7,
        std_bad_exception = 8,
        std_exception = 9,

        // boost::system::system_error
        boost_system_error = 10,

        // hpx::exception
        hpx_exception = 11,
        hpx_thread_interrupted_exception = 12,

#if ASIO_HAS_BOOST_THROW_EXCEPTION != 0
        // boost exceptions
        boost_exception = 13,
#endif

        std_system_error = 14
    };
}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
namespace hpx::serialization {

    namespace detail {

        using save_custom_exception_handler_type =
            std::function<void(hpx::serialization::output_archive&,
                std::exception_ptr const&, unsigned int)>;
        using load_custom_exception_handler_type =
            std::function<void(hpx::serialization::input_archive&,
                std::exception_ptr&, unsigned int)>;

        HPX_CORE_EXPORT void set_save_custom_exception_handler(
            save_custom_exception_handler_type f);
        HPX_CORE_EXPORT void set_load_custom_exception_handler(
            load_custom_exception_handler_type f);
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void save(Archive& ar, std::exception_ptr const& e, unsigned int);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void load(Archive& ar, std::exception_ptr& e, unsigned int);

    HPX_SERIALIZATION_SPLIT_FREE(std::exception_ptr)
}    // namespace hpx::serialization
