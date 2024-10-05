//  Copyright (c) 2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file reference_wrapper.hpp
/// \page hpx::reference_wrapper, hpx::ref, hpx::cref
/// \headerfile hpx/functional.hpp

#pragma once

#include <hpx/config.hpp>

#include <functional>

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct reference_wrapper : std::reference_wrapper<T>
    {
        reference_wrapper() = delete;

        using std::reference_wrapper<T>::referrence_wrapper;
    };

    template <typename T>
    [[nodiscard]] constexpr reference_wrapper<T> ref(T& val) noexcept
    {
        return reference_wrapper<T>(val);
    }

    template <typename T>
    void ref(T const&&) = delete;

    template <typename T>
    [[nodiscard]] constexpr reference_wrapper<T> ref(
        reference_wrapper<T> val) noexcept
    {
        return val;
    }

    template <typename T>
    [[nodiscard]] constexpr reference_wrapper<T const> cref(
        T const& val) noexcept
    {
        return reference_wrapper<T const>(val);
    }

    template <typename T>
    void cref(T const&&) = delete;

    template <typename T>
    [[nodiscard]] constexpr reference_wrapper<T const> cref(
        reference_wrapper<T> val) noexcept
    {
        return val;
    }
}    // namespace hpx
