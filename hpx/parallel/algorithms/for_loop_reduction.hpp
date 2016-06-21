//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/for_loop_reduction.hpp

#if !defined(HPX_PARALLEL_ALGORITHM_FOR_LOOP_REDUCTION_MAR_05_2016_0837PM)
#define HPX_PARALLEL_ALGORITHM_FOR_LOOP_REDUCTION_MAR_05_2016_0837PM

#include <hpx/config.hpp>
#include <hpx/runtime/get_os_thread_count.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/decay.hpp>

#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>

#include <boost/shared_array.hpp>

#include <cstdlib>
#include <functional>
#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v2)
{
    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Op>
        struct reduction_helper
        {
            template <typename Op_>
            reduction_helper(T& var, T const& identity, Op_ && op)
              : var_(var), op_(std::forward<Op_>(op)), curr_(0)
            {
                std::size_t cores = hpx::get_os_thread_count();
                data_.reset(new T[cores]);
                for (std::size_t i = 0; i != cores; ++i)
                    data_[i] = identity;
            }

            void init_iteration(std::size_t)
            {
                curr_ = hpx::get_worker_thread_num();
                HPX_ASSERT(curr_ < hpx::get_os_thread_count());
            }

            T& iteration_value()
            {
                return data_[curr_];
            }

            void next_iteration() HPX_NOEXCEPT {}

            void exit_iteration(std::size_t /*index*/)
            {
                std::size_t cores = hpx::get_os_thread_count();
                for (std::size_t i = 0; i != cores; ++i)
                    var_ = op_(var_, data_[i]);
            }

        private:
            T& var_;
            Op op_;
            std::size_t curr_;
            boost::shared_array<T> data_;
        };

        /// \endcond
    }

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
    /// T shall meet the requirements of CopyConstructible and MoveAssignable.
    /// The expression var = combiner(var, var) shall be well formed.
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
    HPX_FORCEINLINE detail::reduction_helper<T, typename std::decay<Op>::type>
    reduction(T& var, T const& identity, Op && combiner)
    {
        return detail::reduction_helper<
                T, typename std::decay<Op>::type
            >(var, identity, std::forward<Op>(combiner));
    }

    /// \cond NOINTERNAL
    template <typename T>
    HPX_FORCEINLINE detail::reduction_helper<T, std::plus<T> >
    reduction_plus(T& var)
    {
        return reduction(var, T(), std::plus<T>());
    }

    template <typename T>
    HPX_FORCEINLINE detail::reduction_helper<T, std::multiplies<T> >
    reduction_multiplies(T& var)
    {
        return reduction(var, T(1), std::multiplies<T>());
    }

    template <typename T>
    HPX_FORCEINLINE detail::reduction_helper<T, std::bit_and<T> >
    reduction_bit_and(T& var)
    {
        return reduction(var, ~(T()), std::bit_and<T>());
    }

    template <typename T>
    HPX_FORCEINLINE detail::reduction_helper<T, std::bit_or<T> >
    reduction_bit_or(T& var)
    {
        return reduction(var, T(), std::bit_or<T>());
    }

    template <typename T>
    HPX_FORCEINLINE detail::reduction_helper<T, std::bit_xor<T> >
    reduction_bit_xor(T& var)
    {
        return reduction(var, T(), std::bit_xor<T>());
    }

    template <typename T>
    HPX_FORCEINLINE detail::reduction_helper<T, v1::detail::min_of<T> >
    reduction_min(T& var)
    {
        return reduction(var, var, v1::detail::min_of<T>());
    }

    template <typename T>
    HPX_FORCEINLINE detail::reduction_helper<T, v1::detail::max_of<T> >
    reduction_max(T& var)
    {
        return reduction(var, var, v1::detail::max_of<T>());
    }
    /// \endcond
}}}

#endif

