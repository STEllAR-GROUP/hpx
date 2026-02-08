//  Copyright (c) 2020 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <exception>
#include <stdexcept>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::resiliency::experimental {

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_CORE_EXPORT struct HPX_ALWAYS_EXPORT abort_replicate_exception
      : std::exception
    {
        abort_replicate_exception();
        ~abort_replicate_exception() override;

        abort_replicate_exception(abort_replicate_exception const&);
        abort_replicate_exception& operator=(abort_replicate_exception const&);
    };

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_CORE_EXPORT struct HPX_ALWAYS_EXPORT abort_replay_exception
      : std::exception
    {
        abort_replay_exception();
        ~abort_replay_exception() override;

        abort_replay_exception(abort_replay_exception const&);
        abort_replay_exception& operator=(abort_replay_exception const&);
    };

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_CORE_EXPORT struct replicate_voter
        {
            template <typename T>
            constexpr T operator()(std::vector<T>&& vect) const
            {
                return HPX_MOVE(vect.at(0));
            }
        };

        HPX_CXX_CORE_EXPORT struct replicate_validator
        {
            template <typename T>
            constexpr bool operator()(T&&) const noexcept
            {
                return true;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_CORE_EXPORT struct replay_validator
        {
            template <typename T>
            constexpr bool operator()(T&&) const noexcept
            {
                return true;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_CORE_EXPORT template <typename Future>
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
        HPX_CXX_CORE_EXPORT template <typename Future>
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
}    // namespace hpx::resiliency::experimental

#include <hpx/config/warnings_suffix.hpp>
