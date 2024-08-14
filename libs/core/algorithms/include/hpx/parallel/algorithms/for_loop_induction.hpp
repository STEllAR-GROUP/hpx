//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/for_loop_induction.hpp
/// \page hpx::experimental::induction
/// \headerfile hpx/algorithm.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/detail/execution_parameter_callbacks.hpp>
#include <hpx/threading_base/thread_num_tss.hpp>

#if !defined(HPX_HAVE_CXX17_SHARED_PTR_ARRAY)
#include <boost/shared_array.hpp>
#else
#include <memory>
#endif

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "hpx/concepts/concepts.hpp"

namespace hpx::parallel::detail {
    /// \cond NOINTERNAL

    template <typename T>
    struct hpx_thread_local
    {
    private:
        using element_type = hpx::util::cache_line_data<T>;

#if defined(HPX_HAVE_CXX17_SHARED_PTR_ARRAY)
        using array_type = std::shared_ptr<element_type[]>;
#else
        using array_type = boost::shared_array<element_type>;
#endif

    public:
        constexpr explicit hpx_thread_local(T const& init)
        {
            const std::size_t threads =
                hpx::parallel::execution::detail::get_os_thread_count();
            data_.reset(new element_type[threads]);
            std::fill_n(data_.get(), threads, element_type{init});
        }

        // clang-format off
        template<typename O,
            HPX_CONCEPT_REQUIRES_(
                std::is_assignable_v<T&, O const&>
            )>
        // clang-format on
        constexpr hpx_thread_local& operator=(O const& other)
        {
            data_[hpx::get_worker_thread_num()].data_ = other;
            return *this;
        }

        // clang-format off
        constexpr operator T const&() const
        // clang-format on
        {
            return data_[hpx::get_worker_thread_num()].data_;
        }

        constexpr operator T&()
        {
            return data_[hpx::get_worker_thread_num()].data_;
        }

    private:
        array_type data_;
    };

    template <typename Iterable, typename Stride>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Iterable next(
        const hpx_thread_local<Iterable>& val, Stride offset)
    {
        return hpx::parallel::detail::next(
            static_cast<Iterable const&>(val), offset);
    }

    ///////////////////////////////////////////////////////////////////////
    template <typename T>
    struct induction_helper
    {
        explicit constexpr induction_helper(T var) noexcept
          : var_(var)
          , curr_(var)
        {
        }

        HPX_HOST_DEVICE
        constexpr void init_iteration(std::size_t index) noexcept
        {
            curr_ = parallel::detail::next(var_, index);
        }

        HPX_HOST_DEVICE
        constexpr T const& iteration_value() const noexcept
        {
            return curr_;
        }

        HPX_HOST_DEVICE
        constexpr void next_iteration() noexcept
        {
            ++curr_;
        }

        HPX_HOST_DEVICE
        static constexpr void exit_iteration(std::size_t /*index*/) noexcept {}

    private:
        std::decay_t<T> var_;
        hpx_thread_local<T> curr_;
    };

    template <typename T>
    struct induction_helper<T&>
    {
        explicit constexpr induction_helper(T& var) noexcept
          : live_out_var_(var)
          , var_(var)
          , curr_(var)
        {
        }

        HPX_HOST_DEVICE
        constexpr void init_iteration(std::size_t index) noexcept
        {
            curr_ = parallel::detail::next(var_, index);
        }

        HPX_HOST_DEVICE
        constexpr T const& iteration_value() const noexcept
        {
            return curr_;
        }

        HPX_HOST_DEVICE
        constexpr void next_iteration() noexcept
        {
            ++curr_;
        }

        HPX_HOST_DEVICE
        constexpr void exit_iteration(std::size_t index) noexcept
        {
            live_out_var_ = parallel::detail::next(live_out_var_, index);
        }

    private:
        T& live_out_var_;
        T var_;
        hpx_thread_local<T> curr_;
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename T>
    struct induction_stride_helper
    {
        constexpr induction_stride_helper(T var, std::size_t stride) noexcept
          : var_(var)
          , curr_(var)
          , stride_(stride)
        {
        }

        HPX_HOST_DEVICE
        constexpr void init_iteration(std::size_t index) noexcept
        {
            curr_ = parallel::detail::next(var_, stride_ * index);
        }

        HPX_HOST_DEVICE
        constexpr T const& iteration_value() const noexcept
        {
            return curr_;
        }

        HPX_HOST_DEVICE
        constexpr void next_iteration() noexcept
        {
            /*for (std::size_t i{}; i < stride_; ++i)
                ++curr_;*/
            curr_ = parallel::detail::next(curr_, stride_);
        }

        HPX_HOST_DEVICE
        static constexpr void exit_iteration(std::size_t /*index*/) noexcept {}

    private:
        std::decay_t<T> var_;
        hpx_thread_local<T> curr_;
        std::size_t stride_;
    };

    template <typename T>
    struct induction_stride_helper<T&>
    {
        constexpr induction_stride_helper(T& var, std::size_t stride) noexcept
          : live_out_var_(var)
          , var_(var)
          , curr_(var)
          , stride_(stride)
        {
        }

        HPX_HOST_DEVICE
        constexpr void init_iteration(std::size_t index) noexcept
        {
            curr_ = parallel::detail::next(var_, stride_ * index);
        }

        HPX_HOST_DEVICE
        constexpr T const& iteration_value() const noexcept
        {
            return curr_;
        }

        HPX_HOST_DEVICE
        constexpr void next_iteration() noexcept
        {
            curr_ = parallel::detail::next(curr_, stride_);
        }

        HPX_HOST_DEVICE
        constexpr void exit_iteration(std::size_t index) noexcept
        {
            live_out_var_ =
                parallel::detail::next(live_out_var_, stride_ * index);
        }

    private:
        T& live_out_var_;
        T var_;
        hpx_thread_local<T> curr_;
        std::size_t stride_;
    };

    /// \endcond
}    // namespace hpx::parallel::detail

namespace hpx::experimental {

    /// The function template returns an induction object of unspecified type
    /// having a value type and encapsulating an initial value \a value of that
    /// type and, optionally, a stride.
    ///
    /// For each element in the input range, a looping algorithm over input
    /// sequence \a S computes an induction value from an induction variable and
    /// ordinal position \a p within \a S by the formula i + p * stride if a
    /// stride was specified or i + p otherwise. This induction value is passed
    /// to the element access function.
    ///
    /// If the \a value argument to \a induction is a non-const lvalue, then
    /// that lvalue becomes the live-out object for the returned induction
    /// object. For each induction object that has a live-out object, the
    /// looping algorithm assigns the value of i + n * stride to the live-out
    /// object upon return, where \a n is the number of elements in the input
    /// range.
    ///
    /// \tparam T       The value type to be used by the induction object.
    ///
    /// \param value    [in] The initial value to use for the induction object
    /// \param stride   [in] The (optional) stride to use for the induction
    ///                 object (default: 1)
    ///
    /// \returns This returns an induction object with value type \a T, initial
    ///          value \a value, and (if specified) stride \a stride. If \a T is
    ///          an lvalue of non-const type, \a value is used as the live-out
    ///          object for the induction object; otherwise there is no live-out
    ///          object.
    ///
    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::induction_stride_helper<T>
    induction(T&& value, std::size_t stride)
    {
        return hpx::parallel::detail::induction_stride_helper<T>(
            HPX_FORWARD(T, value), stride);
    }

    /// \cond NOINTERNAL
    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::induction_helper<T>
    induction(T&& value)
    {
        return hpx::parallel::detail::induction_helper<T>(
            HPX_FORWARD(T, value));
    }
    /// \endcond
}    // namespace hpx::experimental

namespace hpx::parallel {
    /// \cond IGNORE_DEPRECATED

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::induction is deprecated. Please use "
        "hpx::experimental::induction instead.")
    constexpr decltype(auto) induction(T&& value, std::size_t stride)
    {
        return hpx::experimental::induction(HPX_FORWARD(T, value), stride);
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::induction is deprecated. Please use "
        "hpx::experimental::induction instead.")
    constexpr decltype(auto) induction(T&& value)
    {
        return hpx::experimental::induction(HPX_FORWARD(T, value));
    }
    /// \endcond
}    // namespace hpx::parallel
