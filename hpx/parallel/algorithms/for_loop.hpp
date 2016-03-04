//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/for_loop.hpp

#if !defined(HPX_PARALLEL_ALGORITH_FOR_LOOP_MAR_02_2016_1256PM)
#define HPX_PARALLEL_ALGORITH_FOR_LOOP_MAR_02_2016_1256PM

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
#include <hpx/parallel/util/foreach_partitioner.hpp>
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
        template <std::size_t I, typename... Args>
        HPX_FORCEINLINE typename hpx::util::tuple_element<
            I, hpx::util::tuple<Args&& ...>
        >::type
        nth(Args &&... args)
        {
            return hpx::util::get<I>(
                hpx::util::forward_as_tuple(std::forward<Args>(args)...)
            );
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct induction_helper
        {
            induction_helper(T var, std::size_t stride) HPX_NOEXCEPT
              : var_(var), curr_(var), stride_(stride)
            {}

            void init_iteration(std::size_t index) HPX_NOEXCEPT
            {
                curr_ = parallel::v1::detail::next(var_, index);
            }

            void next_iteration() HPX_NOEXCEPT
            {
                curr_ = parallel::v1::detail::next(curr_, stride_);
            }

            T iteration_value() const HPX_NOEXCEPT
            {
                return curr_;
            }

        private:
            typename std::decay<T>::type var_;
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
        HPX_FORCEINLINE void next_iteration(Modifier& mod)
        {
            mod.next_iteration();
        }

        template <typename Modifier>
        HPX_FORCEINLINE auto iteration_value(Modifier const& mod)
        ->  decltype(mod.iteration_value())
        {
            return mod.iteration_value();
        }

        ///////////////////////////////////////////////////////////////////////
        struct for_loop_n : public v1::detail::algorithm<for_loop_n>
        {
            for_loop_n()
              : for_loop_n::algorithm("for_loop_n")
            {}

            template <typename ExPolicy, typename B, typename E, typename F,
                typename... Args>
            static hpx::util::unused_type
            sequential(ExPolicy, B first, E last, F && f, Args &&... args)
            {
                int init_sequencer[] = {
                    ( init_iteration(args, 0), 0 )..., 0
                };

                for (/**/; first != last; ++first)
                {
                    int next_sequencer[] = {
                        ( next_iteration(args), 0 )..., 0
                    };
                    hpx::util::invoke(f, first, iteration_value(args)...);
                }

                return hpx::util::unused;
            }

            template <typename ExPolicy, typename B, typename E, typename F,
                typename... Args>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy policy, B first, E last, F && f, Args &&... args)
            {
                if (first == last)
                    return util::detail::algorithm_result<ExPolicy>::get();

                std::size_t size = parallel::v1::detail::distance(first, last);
                auto result = util::foreach_partitioner<ExPolicy>::call(
                    policy, first, size,
                    [=](std::size_t part_index,
                        B part_begin, std::size_t part_size) mutable
                    {
                        int init_sequencer[] = {
                            ( init_iteration(args, part_index), 0 )..., 0
                        };

                        for (/**/; part_size != 0; (void)--part_size, ++part_begin)
                        {
                            int next_sequencer[] = {
                                ( next_iteration(args), 0 )..., 0
                            };
                            hpx::util::invoke(f, part_begin,
                                iteration_value(args)...);
                        }
                    });

                //  make sure live-out variables are properly set on return
                int exit_sequencer[] = {
                    ( init_iteration(args, size), 0 )..., 0
                };

                return util::detail::algorithm_result<ExPolicy>::get(
                    std::move(result));
            }
        };

        // reshuffle arguments, last argument is function object, will go first
        template <typename ExPolicy, typename B, typename E, std::size_t... Is,
            typename... Args>
        typename util::detail::algorithm_result<ExPolicy>::type
        for_loop(ExPolicy && policy, B first, E last,
            hpx::util::detail::pack_c<std::size_t, Is...>, Args &&... args)
        {
            typedef typename boost::mpl::bool_<
                is_sequential_execution_policy<ExPolicy>::value ||
                hpx::traits::is_input_iterator<B>::value
            >::type is_seq;

            return for_loop_n().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                nth<sizeof...(Args)-1>(std::forward<Args>(args)...),
                nth<Is>(std::forward<Args>(args)...)...);
        }

        /// \endcond
    }

    /// Requires: \a I shall be an integral type or meet the requirements
    ///           of an input iterator type. The \a rest parameter pack shall
    ///           have at least one element, comprising objects returned by
    ///           invocations of \a reduction ([parallel.alg.reduction]) and/or
    ///           \a induction ([parallel.alg.induction]) function templates
    ///           followed by exactly one element invocable element-access
    ///           function, \a f. \a f shall meet the requirements of
    ///           MoveConstructible.
    ///
    /// Effects:  Applies \a f to each element in the input sequence, with
    ///           additional arguments corresponding to the reductions and
    ///           inductions in the rest parameter pack. The length of the input
    ///           sequence is last - first.
    ///
    /// The first element in the input sequence is specified by \a first. Each
    /// subsequent element is generated by incrementing the previous element.
    ///
    /// \note As described in the C++ standard, section [algorithms.general],
    ///       arithmetic on non-random-access iterators is performed using
    ///       advance and distance.
    ///
    /// \note The order of the elements of the input sequence is important for
    ///       determining ordinal position of an application of \a f, even
    ///       though the applications themselves may be unordered.
    ///
    /// Along with an element from the input sequence, for each member of the
    /// rest parameter pack excluding \a f, an additional argument is passed to
    /// each application of \a f as follows:
    ///
    /// If the pack member is an object returned by a call to a reduction
    /// function listed in section [parallel.alg.reductions], then the
    /// additional argument is a reference to a view of that reduction object.
    /// If the pack member is an object returned by a call to induction, then
    /// the additional argument is the induction value for that induction object
    /// corresponding to the position of the application of \a f in the input
    /// sequence.
    ///
    /// Complexity: Applies \a f exactly once for each element of the input
    ///             sequence.
    ///
    /// Remarks: If \a f returns a result, the result is ignored.
    ///
    template <typename ExPolicy, typename I, typename... Args>
    typename util::detail::algorithm_result<ExPolicy>::type
    for_loop(ExPolicy && policy,
        typename std::decay<I>::type first, I last, Args &&... args)
    {
        static_assert(sizeof...(Args) >= 1,
            "for_loop must be called with at least a function object");

        using hpx::util::detail::make_index_pack;
        return detail::for_loop(
            std::forward<ExPolicy>(policy), first, last,
            typename make_index_pack<sizeof...(Args)-1>::type(),
            std::forward<Args>(args)...);
    }

    template <typename I, typename... Args>
    void for_loop(typename std::decay<I>::type first, I && last,
        Args &&... args)
    {
        static_assert(sizeof...(Args) >= 1,
            "for_loop must be called with at least a function object");

        return for_loop(parallel::seq, first, std::forward<I>(last),
            std::forward<Args>(args)...);
    }

    ///////////////////////////////////////////////////////////////////////////
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
    HPX_FORCEINLINE detail::induction_helper<T>
    induction(T && value, std::size_t stride = 1)
    {
        return detail::induction_helper<T>(std::forward<T>(value), stride);
    }
}}}

#endif

