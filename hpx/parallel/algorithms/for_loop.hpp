//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/for_loop.hpp

#if !defined(HPX_PARALLEL_ALGORITHM_FOR_LOOP_MAR_02_2016_1256PM)
#define HPX_PARALLEL_ALGORITHM_FOR_LOOP_MAR_02_2016_1256PM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/unused.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/algorithms/for_loop_induction.hpp>
#include <hpx/parallel/algorithms/for_loop_reduction.hpp>
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
        template <typename ... Ts, std::size_t ... Is>
        HPX_FORCEINLINE void init_iteration(hpx::util::tuple<Ts...>& args,
            hpx::util::detail::pack_c<std::size_t, Is...>,
            std::size_t part_index)
        {
            int _sequencer[] =
            {
                0, (hpx::util::get<Is>(args).init_iteration(part_index), 0)...
            };
            (void)_sequencer;
        }

        template <typename ... Ts, std::size_t ... Is, typename F, typename B>
        HPX_FORCEINLINE void invoke_iteration(hpx::util::tuple<Ts...>& args,
            hpx::util::detail::pack_c<std::size_t, Is...>, F && f, B part_begin)
        {
            hpx::util::invoke(std::forward<F>(f), part_begin,
                hpx::util::get<Is>(args).iteration_value()...);
        }

        template <typename ... Ts, std::size_t ... Is>
        HPX_FORCEINLINE void next_iteration(hpx::util::tuple<Ts...>& args,
            hpx::util::detail::pack_c<std::size_t, Is...>)
        {
            int _sequencer[] =
            {
                0, (hpx::util::get<Is>(args).next_iteration(), 0)...
            };
            (void)_sequencer;
        }

        template <typename ... Ts, std::size_t ... Is>
        HPX_FORCEINLINE void exit_iteration(hpx::util::tuple<Ts...>& args,
            hpx::util::detail::pack_c<std::size_t, Is...>,
            std::size_t size)
        {
            int _sequencer[] =
            {
                0, (hpx::util::get<Is>(args).exit_iteration(size), 0)...
            };
            (void)_sequencer;
        }

        ///////////////////////////////////////////////////////////////////////
        struct for_loop_n : public v1::detail::algorithm<for_loop_n>
        {
            for_loop_n()
              : for_loop_n::algorithm("for_loop_n")
            {}

            template <typename ExPolicy, typename B, typename E, typename S,
                typename F, typename... Args>
            static hpx::util::unused_type
            sequential(ExPolicy policy, B first, E last, S stride, F && f,
                Args &&... args)
            {
                int init_sequencer[] = {
                    0, (args.init_iteration(0), 0)...
                };
                (void)init_sequencer;

                std::size_t size = parallel::v1::detail::distance(first, last);

                std::size_t count = size;
                for (/* */; first != last; count -= abs(stride))
                {
                    hpx::util::invoke(f, first, args.iteration_value()...);

                    int next_sequencer[] = {
                        0, (args.next_iteration(), 0)...
                    };
                    (void)next_sequencer;

                    // modifies stride
                    first = parallel::v1::detail::next(first, count, stride);
                }

                // make sure live-out variables are properly set on
                // return
                int exit_sequencer[] = {
                    0, (args.exit_iteration(size), 0)...
                };
                (void)exit_sequencer;

                return hpx::util::unused;
            }

            template <typename ExPolicy, typename B, typename E, typename S,
                typename F, typename... Ts>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy policy, B first, E last, S stride, F && f,
                Ts &&... ts)
            {
                if (first == last)
                    return util::detail::algorithm_result<ExPolicy>::get();

                // gcc does not support binding parameter packs as lambda closures
                auto args = hpx::util::make_tuple(std::forward<Ts>(ts)...);
                auto pack = typename hpx::util::detail::make_index_pack<
                    sizeof...(Ts)>::type();

                std::size_t size = parallel::v1::detail::distance(first, last);
                return util::partitioner<ExPolicy>::call_with_index(
                    policy, first, size, stride,
                    [=](std::size_t part_index,
                        B part_begin, std::size_t part_steps) mutable
                    {
                        detail::init_iteration(args, pack, part_index);

                        while (part_steps != 0)
                        {
                            detail::invoke_iteration(args, pack, f, part_begin);

                            detail::next_iteration(args, pack);

                            std::size_t chunk = (std::min)(S(part_steps), stride);

                            // modifies 'chunk'
                            part_begin = parallel::v1::detail::next(
                                part_begin, part_steps, chunk);
                            part_steps -= chunk;
                        }
                    },
                    [=](std::vector<hpx::future<void> > &&) mutable -> void
                    {
                        // make sure live-out variables are properly set on
                        // return
                        detail::exit_iteration(args, pack, size);
                    });
            }
        };

        // reshuffle arguments, last argument is function object, will go first
        template <typename ExPolicy, typename B, typename E, typename S,
            std::size_t... Is, typename... Args>
        typename util::detail::algorithm_result<ExPolicy>::type
        for_loop(ExPolicy && policy, B first, E last, S stride,
            hpx::util::detail::pack_c<std::size_t, Is...>, Args &&... args)
        {
            // stride shall not be zero
            HPX_ASSERT(stride != 0);

            // stride should be negative only if E is an integral type or at
            // least a bidirectional iterator
            if (stride < 0)
            {
                HPX_ASSERT(std::is_integral<E>::value ||
                    hpx::traits::is_bidirectional_iterator<E>::value);
            }

            typedef typename boost::mpl::bool_<
                is_sequential_execution_policy<ExPolicy>::value ||
                hpx::traits::is_input_iterator<B>::value
            >::type is_seq;

            auto && t = hpx::util::forward_as_tuple(std::forward<Args>(args)...);

            return for_loop_n().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last, stride,
                hpx::util::get<sizeof...(Args)-1>(t), hpx::util::get<Is>(t)...);
        }

        /// \endcond
    }

    /// The for_loop implements loop functionality over a range specified by
    /// integral or iterator bounds. For the iterator case, these algorithms
    /// resemble for_each from the Parallelism TS, but leave to the programmer
    /// when and if to dereference the iterator.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam I           The type of the iteration variable. This could be
    ///                     an (input) iterator type or an integral type.
    /// \tparam Args        A parameter pack, it's last element is a function
    ///                     object to be invoked for each iteration, the others
    ///                     have to be either conforming to the induction or
    ///                     reduction concept.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param args         The last element of this parameter pack is the
    ///                     function (object) to invoke, while the remaining
    ///                     elements of the parameter pack are instances of
    ///                     either induction or reduction objects.
    ///                     The function (or function object) which will be
    ///                     invoked for each of the elements in the sequence
    ///                     specified by [first, last) should expose a signature
    ///                     equivalent to:
    ///                     \code
    ///                     <ignored> pred(I const& a, ...);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. It will
    ///                     receive the current value of the iteration variable
    ///                     and one argument for each of the induction or
    ///                     reduction objects passed to the algorithms,
    ///                     representing their current values.
    ///
    /// Requires: \a I shall be an integral type or meet the requirements
    ///           of an input iterator type. The \a args parameter pack shall
    ///           have at least one element, comprising objects returned by
    ///           invocations of \a reduction and/or \a induction function
    ///           templates followed by exactly one element invocable
    ///           element-access function, \a f. \a f shall meet the
    ///           requirements of MoveConstructible.
    ///
    /// Effects:  Applies \a f to each element in the input sequence, with
    ///           additional arguments corresponding to the reductions and
    ///           inductions in the \a args parameter pack. The length of the
    ///           input sequence is last - first.
    ///
    /// The first element in the input sequence is specified by \a first. Each
    /// subsequent element is generated by incrementing the previous element.
    ///
    /// \note As described in the C++ standard, arithmetic on non-random-access
    ///       iterators is performed using advance and distance.
    ///
    /// \note The order of the elements of the input sequence is important for
    ///       determining ordinal position of an application of \a f, even
    ///       though the applications themselves may be unordered.
    ///
    /// Along with an element from the input sequence, for each member of the
    /// \a args parameter pack excluding \a f, an additional argument is passed
    /// to each application of \a f as follows:
    ///
    /// If the pack member is an object returned by a call to a reduction
    /// function listed in section, then the
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
    /// \returns  The \a for_loop algorithm returns a
    ///           \a hpx::future<void> if the execution policy is of
    ///           type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and returns \a void
    ///           otherwise.
    ///
    template <typename ExPolicy, typename I, typename... Args,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        (hpx::traits::is_iterator<I>::value || std::is_integral<I>::value))>
    typename util::detail::algorithm_result<ExPolicy>::type
    for_loop(ExPolicy && policy,
        typename std::decay<I>::type first, I last, Args &&... args)
    {
        static_assert(sizeof...(Args) >= 1,
            "for_loop must be called with at least a function object");

        using hpx::util::detail::make_index_pack;
        return detail::for_loop(
            std::forward<ExPolicy>(policy), first, last, 1,
            typename make_index_pack<sizeof...(Args)-1>::type(),
            std::forward<Args>(args)...);
    }

    /// The for_loop implements loop functionality over a range specified by
    /// integral or iterator bounds. For the iterator case, these algorithms
    /// resemble for_each from the Parallelism TS, but leave to the programmer
    /// when and if to dereference the iterator.
    ///
    /// The execution of for_loop without specifying an execution policy is
    /// equivalent to specifying \a parallel::seq as the execution policy.
    ///
    /// \tparam I           The type of the iteration variable. This could be
    ///                     an (input) iterator type or an integral type.
    /// \tparam Args        A parameter pack, it's last element is a function
    ///                     object to be invoked for each iteration, the others
    ///                     have to be either conforming to the induction or
    ///                     reduction concept.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param args         The last element of this parameter pack is the
    ///                     function (object) to invoke, while the remaining
    ///                     elements of the parameter pack are instances of
    ///                     either induction or reduction objects.
    ///                     The function (or function object) which will be
    ///                     invoked for each of the elements in the sequence
    ///                     specified by [first, last) should expose a signature
    ///                     equivalent to:
    ///                     \code
    ///                     <ignored> pred(I const& a, ...);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. It will
    ///                     receive the current value of the iteration variable
    ///                     and one argument for each of the induction or
    ///                     reduction objects passed to the algorithms,
    ///                     representing their current values.
    ///
    /// Requires: \a I shall be an integral type or meet the requirements
    ///           of an input iterator type. The \a args parameter pack shall
    ///           have at least one element, comprising objects returned by
    ///           invocations of \a reduction and/or \a induction function
    ///           templates followed by exactly one element invocable
    ///           element-access function, \a f. \a f shall meet the
    ///           requirements of MoveConstructible.
    ///
    /// Effects:  Applies \a f to each element in the input sequence, with
    ///           additional arguments corresponding to the reductions and
    ///           inductions in the \a args parameter pack. The length of the
    ///           input sequence is last - first.
    ///
    /// The first element in the input sequence is specified by \a first. Each
    /// subsequent element is generated by incrementing the previous element.
    ///
    /// \note As described in the C++ standard, arithmetic on non-random-access
    ///       iterators is performed using advance and distance.
    ///
    /// \note The order of the elements of the input sequence is important for
    ///       determining ordinal position of an application of \a f, even
    ///       though the applications themselves may be unordered.
    ///
    /// Along with an element from the input sequence, for each member of the
    /// \a args parameter pack excluding \a f, an additional argument is passed
    /// to each application of \a f as follows:
    ///
    /// If the pack member is an object returned by a call to a reduction
    /// function listed in section, then the
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
    template <typename I, typename... Args,
    HPX_CONCEPT_REQUIRES_(
        hpx::traits::is_iterator<I>::value || std::is_integral<I>::value)>
    void for_loop(typename std::decay<I>::type first, I last,
        Args &&... args)
    {
        static_assert(sizeof...(Args) >= 1,
            "for_loop must be called with at least a function object");

        return for_loop(parallel::seq, first, last, std::forward<Args>(args)...);
    }

    /// The for_loop_strided implements loop functionality over a range
    /// specified by integral or iterator bounds. For the iterator case, these
    /// algorithms resemble for_each from the Parallelism TS, but leave to the
    /// programmer when and if to dereference the iterator.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam I           The type of the iteration variable. This could be
    ///                     an (input) iterator type or an integral type.
    /// \tparam S           The type of the stride variable. This should be
    ///                     an integral type.
    /// \tparam Args        A parameter pack, it's last element is a function
    ///                     object to be invoked for each iteration, the others
    ///                     have to be either conforming to the induction or
    ///                     reduction concept.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param stride       Refers to the stride of the iteration steps. This
    ///                     shall have non-zero value and shall be negative
    ///                     only if I has integral type or meets the requirements
    ///                     of a bidirectional iterator.
    /// \param args         The last element of this parameter pack is the
    ///                     function (object) to invoke, while the remaining
    ///                     elements of the parameter pack are instances of
    ///                     either induction or reduction objects.
    ///                     The function (or function object) which will be
    ///                     invoked for each of the elements in the sequence
    ///                     specified by [first, last) should expose a signature
    ///                     equivalent to:
    ///                     \code
    ///                     <ignored> pred(I const& a, ...);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. It will
    ///                     receive the current value of the iteration variable
    ///                     and one argument for each of the induction or
    ///                     reduction objects passed to the algorithms,
    ///                     representing their current values.
    ///
    /// Requires: \a I shall be an integral type or meet the requirements
    ///           of an input iterator type. The \a args parameter pack shall
    ///           have at least one element, comprising objects returned by
    ///           invocations of \a reduction and/or \a induction function
    ///           templates followed by exactly one element invocable
    ///           element-access function, \a f. \a f shall meet the
    ///           requirements of MoveConstructible.
    ///
    /// Effects:  Applies \a f to each element in the input sequence, with
    ///           additional arguments corresponding to the reductions and
    ///           inductions in the \a args parameter pack. The length of the
    ///           input sequence is last - first.
    ///
    /// The first element in the input sequence is specified by \a first. Each
    /// subsequent element is generated by incrementing the previous element.
    ///
    /// \note As described in the C++ standard, arithmetic on non-random-access
    ///       iterators is performed using advance and distance.
    ///
    /// \note The order of the elements of the input sequence is important for
    ///       determining ordinal position of an application of \a f, even
    ///       though the applications themselves may be unordered.
    ///
    /// Along with an element from the input sequence, for each member of the
    /// \a args parameter pack excluding \a f, an additional argument is passed
    /// to each application of \a f as follows:
    ///
    /// If the pack member is an object returned by a call to a reduction
    /// function listed in section, then the
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
    /// \returns  The \a for_loop_strided algorithm returns a
    ///           \a hpx::future<void> if the execution policy is of
    ///           type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and returns \a void
    ///           otherwise.
    ///
    template <typename ExPolicy, typename I, typename S, typename... Args,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        (hpx::traits::is_iterator<I>::value || std::is_integral<I>::value) &&
        std::is_integral<S>::value)>
    typename util::detail::algorithm_result<ExPolicy>::type
    for_loop_strided(ExPolicy && policy,
        typename std::decay<I>::type first, I last, S stride, Args &&... args)
    {
        static_assert(sizeof...(Args) >= 1,
            "for_loop_strided must be called with at least a function object");

        using hpx::util::detail::make_index_pack;
        return detail::for_loop(
            std::forward<ExPolicy>(policy), first, last, stride,
            typename make_index_pack<sizeof...(Args)-1>::type(),
            std::forward<Args>(args)...);
    }

    /// The for_loop_strided implements loop functionality over a range
    /// specified by integral or iterator bounds. For the iterator case, these
    /// algorithms resemble for_each from the Parallelism TS, but leave to the
    /// programmer when and if to dereference the iterator.
    ///
    /// The execution of for_loop without specifying an execution policy is
    /// equivalent to specifying \a parallel::seq as the execution policy.
    ///
    /// \tparam I           The type of the iteration variable. This could be
    ///                     an (input) iterator type or an integral type.
    /// \tparam S           The type of the stride variable. This should be
    ///                     an integral type.
    /// \tparam Args        A parameter pack, it's last element is a function
    ///                     object to be invoked for each iteration, the others
    ///                     have to be either conforming to the induction or
    ///                     reduction concept.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param stride       Refers to the stride of the iteration steps. This
    ///                     shall have non-zero value and shall be negative
    ///                     only if I has integral type or meets the requirements
    ///                     of a bidirectional iterator.
    /// \param args         The last element of this parameter pack is the
    ///                     function (object) to invoke, while the remaining
    ///                     elements of the parameter pack are instances of
    ///                     either induction or reduction objects.
    ///                     The function (or function object) which will be
    ///                     invoked for each of the elements in the sequence
    ///                     specified by [first, last) should expose a signature
    ///                     equivalent to:
    ///                     \code
    ///                     <ignored> pred(I const& a, ...);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. It will
    ///                     receive the current value of the iteration variable
    ///                     and one argument for each of the induction or
    ///                     reduction objects passed to the algorithms,
    ///                     representing their current values.
    ///
    /// Requires: \a I shall be an integral type or meet the requirements
    ///           of an input iterator type. The \a args parameter pack shall
    ///           have at least one element, comprising objects returned by
    ///           invocations of \a reduction and/or \a induction function
    ///           templates followed by exactly one element invocable
    ///           element-access function, \a f. \a f shall meet the
    ///           requirements of MoveConstructible.
    ///
    /// Effects:  Applies \a f to each element in the input sequence, with
    ///           additional arguments corresponding to the reductions and
    ///           inductions in the \a args parameter pack. The length of the
    ///           input sequence is last - first.
    ///
    /// The first element in the input sequence is specified by \a first. Each
    /// subsequent element is generated by incrementing the previous element.
    ///
    /// \note As described in the C++ standard, arithmetic on non-random-access
    ///       iterators is performed using advance and distance.
    ///
    /// \note The order of the elements of the input sequence is important for
    ///       determining ordinal position of an application of \a f, even
    ///       though the applications themselves may be unordered.
    ///
    /// Along with an element from the input sequence, for each member of the
    /// \a args parameter pack excluding \a f, an additional argument is passed
    /// to each application of \a f as follows:
    ///
    /// If the pack member is an object returned by a call to a reduction
    /// function listed in section, then the
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
    template <typename I, typename S, typename... Args,
    HPX_CONCEPT_REQUIRES_(
        (hpx::traits::is_iterator<I>::value || std::is_integral<I>::value) &&
        std::is_integral<S>::value)>
    void for_loop_strided(typename std::decay<I>::type first, I last,
        S stride, Args &&... args)
    {
        static_assert(sizeof...(Args) >= 1,
            "for_loop_strided must be called with at least a function object");

        return for_loop_strided(parallel::seq, first, last, stride,
            std::forward<Args>(args)...);
    }
}}}

#endif

