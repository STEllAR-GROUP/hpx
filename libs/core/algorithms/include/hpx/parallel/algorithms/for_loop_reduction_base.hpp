//  Copyright (c) 2007-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/for_loop_reduction_base.hpp
/// \page hpx::experimental::reduction
/// \headerfile hpx/algorithm.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/execution/detail/execution_parameter_callbacks.hpp>

#if !defined(HPX_HAVE_CXX17_SHARED_PTR_ARRAY)
#include <boost/shared_array.hpp>
#else
#include <memory>
#endif

#include <cstddef>
#include <type_traits>
#include <utility>

/// \cond NOINTERNAL
namespace hpx::parallel::detail {

    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename Op>
    struct reduction_helper
    {
        using needs_current_thread_num = void;

        template <typename Op_>
        constexpr reduction_helper(T& var, T const& identity, Op_&& op)
          : var_(var)
          , op_(HPX_FORWARD(Op_, op))
        {
            std::size_t const cores =
                hpx::parallel::execution::detail::get_os_thread_count();
            data_.reset(new hpx::util::cache_line_data<T>[cores]);
            for (std::size_t i = 0; i != cores; ++i)
            {
                data_[i].data_ = identity;
            }
        }

        HPX_HOST_DEVICE static constexpr void init_iteration(
            std::size_t /*index*/,
            [[maybe_unused]] std::size_t current_thread) noexcept
        {
            HPX_ASSERT(current_thread <
                hpx::parallel::execution::detail::get_os_thread_count());
        }

        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr T& iteration_value(
            std::size_t current_thread) noexcept
        {
            return data_[current_thread].data_;
        }

        HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr void next_iteration(
            std::size_t /*current_thread*/) noexcept
        {
        }

        HPX_HOST_DEVICE void exit_iteration(std::size_t /*index*/)
        {
            std::size_t const cores =
                hpx::parallel::execution::detail::get_os_thread_count();
            for (std::size_t i = 0; i != cores; ++i)
            {
                var_ = op_(var_, data_[i].data_);
            }
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
}    // namespace hpx::parallel::detail
/// \endcond

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
    ///                 operation. This argument is optional and defaults to
    ///                 a copy of \a var.
    /// \param combiner [in] The binary function (object) used to perform a
    ///                 pairwise reduction on the elements.
    ///
    /// T shall meet the requirements of \a CopyConstructible and
    /// \a MoveAssignable.
    /// The expression \code var = combiner(var, var) \endcode
    /// shall be well-formed.
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

    template <typename T, typename Op>
    HPX_FORCEINLINE constexpr hpx::parallel::detail::reduction_helper<T,
        std::decay_t<Op>>
    reduction(T& var, Op&& combiner)
    {
        return hpx::parallel::detail::reduction_helper<T, std::decay_t<Op>>(
            var, var, HPX_FORWARD(Op, combiner));
    }
}    // namespace hpx::experimental
/// \endcond
