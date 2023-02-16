//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/for_loop_reduction.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/detail/execution_parameter_callbacks.hpp>
#include <hpx/threading_base/thread_num_tss.hpp>

#if !defined(HPX_HAVE_CXX17_SHARED_PTR_ARRAY)
#include <boost/shared_array.hpp>
#else
#include <memory>
#endif

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace hpx::parallel::detail {

    ///////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    template <typename T, typename Op>
    struct reduction_helper
    {
        template <typename Op_>
        constexpr reduction_helper(T& var, T const& identity, Op_&& op)
          : var_(var)
          , op_(HPX_FORWARD(Op_, op))
        {
            std::size_t const cores =
                hpx::parallel::execution::detail::get_os_thread_count();
            data_.reset(new hpx::util::cache_line_data<T>[cores]);
            for (std::size_t i = 0; i != cores; ++i)
                data_[i].data_ = identity;
        }

        static constexpr void init_iteration(std::size_t)
        {
            HPX_ASSERT(hpx::get_worker_thread_num() <
                hpx::parallel::execution::detail::get_os_thread_count());
        }

        constexpr T& iteration_value()
        {
            return data_[hpx::get_worker_thread_num()].data_;
        }

        static constexpr void next_iteration() noexcept {}

        void exit_iteration(std::size_t /*index*/)
        {
            std::size_t const cores =
                hpx::parallel::execution::detail::get_os_thread_count();
            for (std::size_t i = 0; i != cores; ++i)
                var_ = op_(var_, data_[i].data_);
        }

    private:
        T& var_;
        Op op_;
#if defined(HPX_HAVE_CXX17_SHARED_PTR_ARRAY)
        std::shared_ptr<hpx::util::cache_line_data<T>[]> data_;
#else
        boost::shared_array<hpx::util::cache_line_data<T>> data_;
#endif
    };
    /// \endcond
}    // namespace hpx::parallel::detail

namespace hpx::experimental {

    /// The function template returns a reduction object of unspecified type
    /// having a value type and encapsulating an identity value for the
    /// reduction, a combiner function object, and a live-out object from which
    /// the initial value is obtained and into which the final value is stored.
    ///
    /// A parallel algorithm uses reduction objects by allocating an unspecified
    /// number of instances, called views, of the reduction's value type. Each
    /// view is initialized with the reduction object's identity value, except
    /// that the live-out object (which was initialized by the caller) comprises
    /// one of the views. The algorithm passes a reference to a view to each
    /// application of an element-access function, ensuring that no two
    /// concurrently-executing invocations share the same view. A view can be
    /// shared between two applications that do not execute concurrently, but
    /// initialization is performed only once per view.
    ///
    /// Modifications to the view by the application of element access functions
    /// accumulate as partial results. At some point before the algorithm returns,
    /// the partial results are combined, two at a time, using the reduction
    /// object's combiner operation until a single value remains, which is then
    /// assigned back to the live-out object.
    ///
    /// \tparam T       The value type to be used by the induction object.
    /// \tparam Op      The type of the binary function (object) used to
    ///                 perform the reduction operation.
    ///
    /// \param var      [in,out] The life-out value to use for the reduction
    ///                 object. This will hold the reduced value after the
    ///                 algorithm is finished executing.
    /// \param identity [in] The identity value to use for the reduction
    ///                 operation.
    /// \param combiner [in] The binary function (object) used to perform a
    ///                 pairwise reduction on the elements.
    ///
    /// T shall meet the requirements of \a CopyConstructible and
    /// \a MoveAssignable.
    /// The expression \code var = combiner(var, var) \endcode
    /// shall be well formed.
    ///
    /// \note In order to produce useful results, modifications to the view
    ///       should be limited to commutative operations closely related to
    ///       the combiner operation. For example if the combiner is plus<T>,
    ///       incrementing the view would be consistent with the combiner but
    ///       doubling it or assigning to it would not.
    ///
    /// \returns This returns a reduction object of unspecified type having a
    ///          value type of \a T. When the return value is used by an
    ///          algorithm, the reference to \a var is used as the live-out
    ///          object, new views are initialized to a copy of identity, and
    ///          views are combined by invoking the copy of combiner, passing
    ///          it the two views to be combined.
    ///
    template <typename T, typename Op>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        std::decay_t<Op>>
    reduction(T& var, T const& identity, Op&& combiner)
    {
        return hpx::parallel::detail::reduction_helper<T, std::decay_t<Op>>(
            var, identity, HPX_FORWARD(Op, combiner));
    }

    /// \cond NOINTERNAL
    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        std::plus<T>>
    reduction_plus(T& var)
    {
        return reduction(var, T(), std::plus<T>());
    }

    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        std::plus<T>>
    reduction_plus(T& var, T const& identity)
    {
        return reduction(var, identity, std::plus<T>());
    }

    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        std::multiplies<T>>
    reduction_multiplies(T& var)
    {
        return reduction(var, T(1), std::multiplies<T>());
    }

    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        std::multiplies<T>>
    reduction_multiplies(T& var, T const& identity)
    {
        return reduction(var, identity, std::multiplies<T>());
    }

    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        std::bit_and<T>>
    reduction_bit_and(T& var)
    {
        return reduction(var, ~T(), std::bit_and<T>());
    }

    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        std::bit_and<T>>
    reduction_bit_and(T& var, T const& identity)
    {
        return reduction(var, identity, std::bit_and<T>());
    }

    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        std::bit_or<T>>
    reduction_bit_or(T& var)
    {
        return reduction(var, T(), std::bit_or<T>());
    }

    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        std::bit_or<T>>
    reduction_bit_or(T& var, T const& identity)
    {
        return reduction(var, identity, std::bit_or<T>());
    }

    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        std::bit_xor<T>>
    reduction_bit_xor(T& var)
    {
        return reduction(var, T(), std::bit_xor<T>());
    }

    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        std::bit_xor<T>>
    reduction_bit_xor(T& var, T const& identity)
    {
        return reduction(var, identity, std::bit_xor<T>());
    }

    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        hpx::parallel::detail::min_of<T>>
    reduction_min(T& var)
    {
        return reduction(var, var, hpx::parallel::detail::min_of<T>());
    }

    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        hpx::parallel::detail::min_of<T>>
    reduction_min(T& var, T const& identity)
    {
        return reduction(var, identity, hpx::parallel::detail::min_of<T>());
    }

    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        hpx::parallel::detail::max_of<T>>
    reduction_max(T& var)
    {
        return reduction(var, var, hpx::parallel::detail::max_of<T>());
    }

    template <typename T>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        hpx::parallel::detail::max_of<T>>
    reduction_max(T& var, T const& identity)
    {
        return reduction(var, identity, hpx::parallel::detail::max_of<T>());
    }
    /// \endcond
}    // namespace hpx::experimental

namespace hpx::parallel {
    /// \cond IGNORE_DEPRECATED

    template <typename T, typename Op>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::reduction is deprecated. Please use "
        "hpx::experimental::reduction instead.")
    constexpr decltype(auto) reduction(T& var, T const& identity, Op&& combiner)
    {
        return hpx::experimental::reduction(
            var, identity, HPX_FORWARD(Op, combiner));
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::reduction_plus is deprecated. Please use "
        "hpx::experimental::reduction_plus instead.")
    constexpr decltype(auto) reduction_plus(T& var)
    {
        return hpx::experimental::reduction_plus(var);
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::reduction_plus is deprecated. Please use "
        "hpx::experimental::reduction_plus instead.")
    constexpr decltype(auto) reduction_plus(T& var, T const& identity)
    {
        return hpx::experimental::reduction_plus(var, identity);
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::reduction_multiplies is deprecated. Please use "
        "hpx::experimental::reduction_multiplies instead.")
    constexpr decltype(auto) reduction_multiplies(T& var)
    {
        return hpx::experimental::reduction_multiplies(var);
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::reduction_multiplies is deprecated. Please use "
        "hpx::experimental::reduction_multiplies instead.")
    constexpr decltype(auto) reduction_multiplies(T& var, T const& identity)
    {
        return hpx::experimental::reduction_multiplies(var, identity);
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::reduction_bit_and is deprecated. Please use "
        "hpx::experimental::reduction_bit_and instead.")
    constexpr decltype(auto) reduction_bit_and(T& var)
    {
        return hpx::experimental::reduction_bit_and(var);
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::reduction_bit_and is deprecated. Please use "
        "hpx::experimental::reduction_bit_and instead.")
    constexpr decltype(auto) reduction_bit_and(T& var, T const& identity)
    {
        return hpx::experimental::reduction_bit_and(var, identity);
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::reduction_bit_or is deprecated. Please use "
        "hpx::experimental::reduction_bit_or instead.")
    constexpr decltype(auto) reduction_bit_or(T& var)
    {
        return hpx::experimental::reduction_bit_or(var);
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::reduction_bit_or is deprecated. Please use "
        "hpx::experimental::reduction_bit_or instead.")
    constexpr decltype(auto) reduction_bit_or(T& var, T const& identity)
    {
        return hpx::experimental::reduction_bit_or(var, identity);
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::reduction_bit_xor is deprecated. Please use "
        "hpx::experimental::reduction_bit_xor instead.")
    constexpr decltype(auto) reduction_bit_xor(T& var)
    {
        return hpx::experimental::reduction_bit_xor(var);
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::reduction_bit_xor is deprecated. Please use "
        "hpx::experimental::reduction_bit_xor instead.")
    constexpr decltype(auto) reduction_bit_xor(T& var, T const& identity)
    {
        return hpx::experimental::reduction_bit_xor(var, identity);
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::reduction_min is deprecated. Please use "
        "hpx::experimental::reduction_min instead.")
    constexpr decltype(auto) reduction_min(T& var)
    {
        return hpx::experimental::reduction_min(var);
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::reduction_min is deprecated. Please use "
        "hpx::experimental::reduction_min instead.")
    constexpr decltype(auto) reduction_min(T& var, T const& identity)
    {
        return hpx::experimental::reduction_min(var, identity);
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::reduction_max is deprecated. Please use "
        "hpx::experimental::reduction_max instead.")
    constexpr decltype(auto) reduction_max(T& var)
    {
        return hpx::experimental::reduction_max(var);
    }

    template <typename T>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::reduction_max is deprecated. Please use "
        "hpx::experimental::reduction_max instead.")
    constexpr decltype(auto) reduction_max(T& var, T const& identity)
    {
        return hpx::experimental::reduction_max(var, identity);
    }
    /// \endcond
}    // namespace hpx::parallel
