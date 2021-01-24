//  Copyright (c) 2020 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/export_definitions.hpp>

#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hpx { namespace resiliency { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    struct HPX_ALWAYS_EXPORT abort_replicate_exception : std::exception
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    struct HPX_ALWAYS_EXPORT abort_replay_exception : std::exception
    {
    };

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        struct replicate_voter
        {
            template <typename T>
            constexpr T operator()(std::vector<T>&& vect) const
            {
                return std::move(vect.at(0));
            }
        };

        struct replicate_validator
        {
            template <typename T>
            constexpr bool operator()(T&&) const noexcept
            {
                return true;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        struct replay_validator
        {
            template <typename T>
            constexpr bool operator()(T&&) const noexcept
            {
                return true;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Future>
        std::exception_ptr rethrow_on_abort_replicate(Future& f)
        {
            std::exception_ptr ex;
            try
            {
                f.get();
            }
            catch (abort_replicate_exception const&)
            {
                throw;
            }
            catch (...)
            {
                ex = std::current_exception();
            }
            return ex;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Future>
        std::exception_ptr rethrow_on_abort_replay(Future& f)
        {
            std::exception_ptr ex;
            try
            {
                f.get();
            }
            catch (abort_replay_exception const&)
            {
                throw;
            }
            catch (...)
            {
                ex = std::current_exception();
            }
            return ex;
        }

    }    // namespace detail
}}}      // namespace hpx::resiliency::experimental
