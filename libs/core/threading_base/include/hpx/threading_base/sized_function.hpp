//  Copyright (c) 2026 Vansh Dobhal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \page hpx::threads::with_task_size
/// \headerfile hpx/functional.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/type_support.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx::threads {

    ///////////////////////////////////////////////////////////////////////////
    /// A function that carries the expected run time of the task it runs in.
    /// The size is a template argument, so the wrapper adds no storage.
    HPX_CXX_CORE_EXPORT template <typename F, task_size Size>
    struct sized_function
    {
        F f;

        template <typename... Ts>
        hpx::util::invoke_result_t<F, Ts...> operator()(Ts&&... ts)
        {
            return HPX_INVOKE(f, HPX_FORWARD(Ts, ts)...);
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const /*version*/)
        {
            // clang-format off
            ar & f;
            // clang-format on
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns a function that carries the given task size.
    ///
    /// \param f Function whose task is expected to run for about \a Size
    HPX_CXX_CORE_EXPORT template <task_size Size, typename F>
    constexpr sized_function<std::decay_t<F>, Size> with_task_size(F&& f)
    {
        return {HPX_FORWARD(F, f)};
    }
}    // namespace hpx::threads

///////////////////////////////////////////////////////////////////////////////
template <typename F, hpx::threads::task_size Size>
struct hpx::traits::get_task_size<hpx::threads::sized_function<F, Size>>
{
    static constexpr threads::task_size value = Size;
};

// The wrapped function keeps its address and annotation.
template <typename F, hpx::threads::task_size Size>
struct hpx::traits::get_function_address<hpx::threads::sized_function<F, Size>>
{
    static constexpr std::size_t call(
        hpx::threads::sized_function<F, Size> const& f) noexcept
    {
        return get_function_address<F>::call(f.f);
    }
};

template <typename F, hpx::threads::task_size Size>
struct hpx::traits::get_function_annotation<
    hpx::threads::sized_function<F, Size>>
{
    static constexpr char const* call(
        hpx::threads::sized_function<F, Size> const& f) noexcept
    {
        return get_function_annotation<F>::call(f.f);
    }
};
