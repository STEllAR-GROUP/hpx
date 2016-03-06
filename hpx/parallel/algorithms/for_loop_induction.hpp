//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/for_loop_induction.hpp

#if !defined(HPX_PARALLEL_ALGORITH_FOR_LOOP_INDUCTION_MAR_05_2016_0305PM)
#define HPX_PARALLEL_ALGORITH_FOR_LOOP_INDUCTION_MAR_05_2016_0305PM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pack.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <boost/mpl/bool.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v2)
{
    // for_loop
    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct induction_helper
        {
            induction_helper(T var) HPX_NOEXCEPT
              : var_(var), curr_(var)
            {}

            void init_iteration(std::size_t index) HPX_NOEXCEPT
            {
                curr_ = parallel::v1::detail::next(var_, index);
            }

            T next_iteration() HPX_NOEXCEPT
            {
                return curr_++;
            }

            void exit_iteration(std::size_t /*index*/) HPX_NOEXCEPT
            {
            }

        private:
            typename std::decay<T>::type var_;
            T curr_;
        };

        template <typename T>
        struct induction_helper<T&>
        {
            induction_helper(T& var) HPX_NOEXCEPT
              : live_out_var_(var), var_(var), curr_(var)
            {}

            void init_iteration(std::size_t index) HPX_NOEXCEPT
            {
                curr_ = parallel::v1::detail::next(var_, index);
            }

            T next_iteration() HPX_NOEXCEPT
            {
                return curr_++;
            }

            void exit_iteration(std::size_t index) HPX_NOEXCEPT
            {
                live_out_var_ = parallel::v1::detail::next(
                    live_out_var_, index);
            }

        private:
            T& live_out_var_;
            T var_;
            T curr_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct induction_stride_helper
        {
            induction_stride_helper(T var, std::size_t stride) HPX_NOEXCEPT
              : var_(var), curr_(var), stride_(stride)
            {}

            void init_iteration(std::size_t index) HPX_NOEXCEPT
            {
                curr_ = parallel::v1::detail::next(var_, stride_*index);
            }

            T next_iteration() HPX_NOEXCEPT
            {
                T curr = curr_;
                curr_ = parallel::v1::detail::next(curr_, stride_);
                return curr;
            }

            void exit_iteration(std::size_t /*index*/) HPX_NOEXCEPT
            {
            }

        private:
            typename std::decay<T>::type var_;
            T curr_;
            std::size_t stride_;
        };

        template <typename T>
        struct induction_stride_helper<T&>
        {
            induction_stride_helper(T& var, std::size_t stride) HPX_NOEXCEPT
              : live_out_var_(var), var_(var), curr_(var), stride_(stride)
            {}

            void init_iteration(std::size_t index) HPX_NOEXCEPT
            {
                curr_ = parallel::v1::detail::next(var_, stride_*index);
            }

            T next_iteration() HPX_NOEXCEPT
            {
                T curr = curr_;
                curr_ = parallel::v1::detail::next(curr_, stride_);
                return curr;
            }

            void exit_iteration(std::size_t index) HPX_NOEXCEPT
            {
                live_out_var_ = parallel::v1::detail::next(
                    live_out_var_, stride_*index);
            }

        private:
            T& live_out_var_;
            T var_;
            T curr_;
            std::size_t stride_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Modifier>
        HPX_FORCEINLINE void init_iteration(Modifier& mod, std::size_t index)
        {
            mod.init_iteration(index);
        }

        template <typename Modifier>
        HPX_FORCEINLINE void exit_iteration(Modifier& mod, std::size_t index)
        {
            mod.exit_iteration(index);
        }

        template <typename Modifier>
        HPX_FORCEINLINE auto next_iteration(Modifier& mod)
        ->  decltype(mod.next_iteration())
        {
            return mod.next_iteration();
        }

        /// \endcond
    }

    /// The function template returns an induction object of unspecified type
    /// having a value type and encapsulating an initial value \a value of that
    /// type and, optionally, a stride.
    ///
    /// For each element in the input range, a looping algorithm over input
    /// sequence \a S computes an induction value from an induction variable
    /// and ordinal position \a p within \a S by the formula
    /// \a i + p * stride if a stride was specified or \a i + p otherwise.
    /// This induction value is passed to the element access function.
    ///
    /// If the \a value argument to \a induction is a non-const lvalue, then
    /// that lvalue becomes the live-out object for the returned induction
    /// object. For each induction object that has a live-out object, the
    /// looping algorithm assigns the value of \a i + n * stride to the live-out
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
    ///          value \a value, and (if specified) stride \a stride. If \a T
    ///          is an lvalue of non-const type, \a value is used as the live-out
    ///          object for the induction object; otherwise there is no live-out
    ///          object.
    ///
    template <typename T>
    HPX_FORCEINLINE detail::induction_stride_helper<T>
    induction(T && value, std::size_t stride)
    {
        return detail::induction_stride_helper<T>(std::forward<T>(value), stride);
    }

    /// \cond NOINTERNAL
    template <typename T>
    HPX_FORCEINLINE detail::induction_helper<T>
    induction(T && value)
    {
        return detail::induction_helper<T>(std::forward<T>(value));
    }
    /// \endcond
}}}

#endif

