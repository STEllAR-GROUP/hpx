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
        typename hpx::util::tuple_element<I, hpx::util::tuple<Args&& ...> >::type
        nth(Args &&... args)
        {
            return hpx::util::get<I>(
                hpx::util::forward_as_tuple(std::forward<Args>(args)...)
            );
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Iterable, typename Enable = void>
        struct calculate_distance
        {
            template <typename T1, typename T2>
            static std::size_t call(T1 t1, T2 t2)
            {
                return t2 - t1;
            }
        };

        template <typename Iterable>
        struct calculate_distance<Iterable,
            typename std::enable_if<
                hpx::traits::is_iterator<Iterable>::value
            >::type>
        {
            template <typename Iter1, typename Iter2>
            static std::size_t call(Iter1 iter1, Iter2 iter2)
            {
                return std::distance(iter1, iter2);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        struct for_loop_n : public detail::algorithm<for_loop_n>
        {
            for_loop_n()
              : for_loop_n::algorithm("for_loop_n")
            {}

            template <typename ExPolicy, typename B, typename E, typename F,
                typename... Args>
            static void sequential(ExPolicy, B first, E last, F && f,
                Args&&... args)
            {
                return util::loop(first, last,
                    [&](B curr)
                    {
                        hpx::util::invoke(f, curr, next_iteration(args)...);
                    });
            }

            template <typename ExPolicy, typename B, typename E, typename F,
                typename... Args>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy policy, B first, E last, F && f, Args&&... args)
            {
                if (first == last)
                    return util::detail::algorithm_result<ExPolicy>::get();

                return util::foreach_partitioner<ExPolicy>::call(
                    policy, first, calculate_distance<I>::call(first, last),
                    [=](B part_begin, std::size_t part_size, std::size_t part_index)
                    {
                        util::loop_n(
                            part_begin, part_size,
                            [&](Iter curr) mutable
                            {
                                hpx::util::invoke(f, curr,
                                    next_iteration(args, part_index)...);
                            });
                    });
            }
        };

        // reshuffle arguments, last argument is function object, will go first
        template <typename ExPolicy, typename B, typename E, std::size_t... Is,
            typename... Args>
        typename util::detail::algorithm_result<ExPolicy>::type
        for_loop(ExPolicy && policy, B first, E last,
            hpx::util::detail::pack_c<std::size_t, Is...>, Args && args...)
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
    typename util::detail::algorithm_result<ExPolicy>::type
    for_loop(typename std::decay<I>::type first, I && last,
        Args &&... args)
    {
        static_assert(sizeof...(Args) >= 1,
            "for_loop must be called with at least a function object");

        return for_loop(parallel::par, first, std::forward<I>(last),
            std::forward<Args>(args)...);
    }
}}}

#endif

