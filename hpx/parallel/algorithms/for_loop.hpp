//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/for_loop.hpp

#if !defined(HPX_PARALLEL_ALGORITHM_FOR_LOOP_MAR_02_2016_1256PM)
#define HPX_PARALLEL_ALGORITHM_FOR_LOOP_MAR_02_2016_1256PM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#include <hpx/traits/get_function_address.hpp>
#include <hpx/traits/get_function_annotation.hpp>
#endif
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/annotated_function.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/unused.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/algorithms/for_loop_induction.hpp>
#include <hpx/parallel/algorithms/for_loop_reduction.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v2
{
    // for_loop
    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        template <typename ... Ts>
        HPX_HOST_DEVICE
        HPX_FORCEINLINE void init_iteration(std::size_t part_index, Ts&... args)
        {
            int const _sequencer[] =
            {
                0, (args.init_iteration(part_index), 0)...
            };
            (void)_sequencer;
        }

        template <typename ... Ts>
        HPX_HOST_DEVICE
        HPX_FORCEINLINE void next_iteration(std::size_t part_index, Ts&... args)
        {
            int const _sequencer[] =
            {
                0, (args.next_iteration(part_index), 0)...
            };
            (void)_sequencer;
        }

        template <typename ... Ts>
        HPX_HOST_DEVICE
        HPX_FORCEINLINE void exit_iteration(std::size_t size, Ts&... args)
        {
            int const _sequencer[] =
            {
                0, (args.exit_iteration(size), 0)...
            };
            (void)_sequencer;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename S>
        struct part_iterations
        {
            typedef typename hpx::util::decay<F>::type fun_type;

            fun_type f_;
            S stride_;

            template <typename F_, typename S_>
            part_iterations(F_&& f, S_&& stride)
              : f_(std::forward<F_>(f))
              , stride_(std::forward<S_>(stride))
            {}

            template <typename B, typename...Args>
            HPX_HOST_DEVICE
            void execute(B part_begin, std::size_t part_steps,
                std::size_t part_index, Args &&... args)
            {
                detail::init_iteration(part_index, args...);

                while (part_steps != 0)
                {
                    hpx::util::invoke(
                        f_, part_begin, args.iteration_value()...);

                    // NVCC seems to have a bug with std::min...
                    std::size_t chunk =
                        S(part_steps) < stride_ ? part_steps : stride_;

                    // modifies 'chunk'
                    part_begin = parallel::v1::detail::next(
                        part_begin, part_steps, chunk);
                    part_steps -= chunk;
                    part_index += chunk;

                    detail::next_iteration(part_index, args...);
                }
            }

            template <typename B, typename...Args>
            HPX_HOST_DEVICE
            void operator()(B part_begin, std::size_t part_steps,
                std::size_t part_index, Args &&... args)
            {
                hpx::util::annotate_function annotate(f_);
                execute(part_begin, part_steps, part_index,
                    std::forward<Args>(args)...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        struct for_loop_algo : public v1::detail::algorithm<for_loop_algo>
        {
            for_loop_algo()
              : for_loop_algo::algorithm("for_loop_algo")
            {}

            template <typename ExPolicy, typename InIter, typename Size,
                typename S, typename F, typename... Args>
            HPX_HOST_DEVICE
            static hpx::util::unused_type
            sequential(ExPolicy policy, InIter first, Size size, S stride,
                F && f, Args &&... args)
            {
                std::size_t part_index = 0;
                detail::init_iteration(0, args...);

                std::size_t count = size;
                while (count != 0)
                {
                    hpx::util::invoke(f, first, args.iteration_value()...);

                    // NVCC seems to have a bug with std::min...
                    std::size_t chunk = (S(count) < stride) ? count : stride;

                    // modifies 'chunk'
                    first = parallel::v1::detail::next(first, count, chunk);
                    count -= chunk;
                    part_index += chunk;

                    detail::next_iteration(part_index, args...);
                }

                // make sure live-out variables are properly set on return
                detail::exit_iteration(size, args...);

                return hpx::util::unused_type();
            }

            template <typename ExPolicy, typename B, typename Size, typename S,
                typename F, typename... Ts>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy policy, B first, Size size, S stride, F && f,
                Ts &&... ts)
            {
                if (size == 0)
                    return util::detail::algorithm_result<ExPolicy>::get();

                return util::partitioner<ExPolicy>::call_with_index(policy,
                    first, size, stride,
                    part_iterations<F, S>{std::forward<F>(f), stride},
                    [size](std::vector<hpx::future<void>>&&,
                        typename hpx::util::decay<Ts>::type... ts) -> void
                    {
                        // make sure live-out variables are properly set on
                        // return
                        detail::exit_iteration(size, ts...);
                    },
                    std::forward<Ts>(ts)...);
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

            // the for_loop should be executed sequentially if the
            // execution policy enforces sequential execution or if the
            // loop boundaries are integral types
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
            static_assert(
                (std::is_integral<B>::value ||
                 hpx::traits::is_input_iterator<B>::value),
                "Requires at least input iterator or integral loop boundaries.");

            typedef std::integral_constant<bool,
                    execution::is_sequenced_execution_policy<ExPolicy>::value ||
                    (!std::is_integral<B>::value &&
                     !hpx::traits::is_forward_iterator<B>::value)
                > is_seq;
#else
            static_assert(
                (std::is_integral<B>::value ||
                 hpx::traits::is_forward_iterator<B>::value),
                "Requires at least forward iterator or integral loop boundaries.");

            typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
#endif

            std::size_t size = parallel::v1::detail::distance(first, last);
            auto && t = hpx::util::forward_as_tuple(std::forward<Args>(args)...);

            return for_loop_algo().call(
                std::forward<ExPolicy>(policy), is_seq(), first, size, stride,
                hpx::util::get<sizeof...(Args)-1>(t), hpx::util::get<Is>(t)...);
        }

        // reshuffle arguments, last argument is function object, will go first
        template <typename ExPolicy, typename B, typename Size, typename S,
            std::size_t... Is, typename... Args>
        typename util::detail::algorithm_result<ExPolicy>::type
        for_loop_n(ExPolicy && policy, B first, Size size, S stride,
            hpx::util::detail::pack_c<std::size_t, Is...>, Args &&... args)
        {
            // Size should be non-negative
            HPX_ASSERT(size >= 0);

            // stride shall not be zero
            HPX_ASSERT(stride != 0);

            // stride should be negative only if E is an integral type or at
            // least a bidirectional iterator
            if (stride < 0)
            {
                HPX_ASSERT(std::is_integral<B>::value ||
                    hpx::traits::is_bidirectional_iterator<B>::value);
            }

#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
            // the for_loop_n should be executed sequentially either if the
            // execution policy enforces sequential execution or if the
            // loop boundaries are input or output iterators
            typedef std::integral_constant<bool,
                    execution::is_sequenced_execution_policy<ExPolicy>::value ||
                    (!std::is_integral<B>::value &&
                     !hpx::traits::is_forward_iterator<B>::value)
                > is_seq;
#else
            static_assert(
                (std::is_integral<B>::value ||
                 hpx::traits::is_forward_iterator<B>::value),
                "Requires at least forward iterator or integral loop boundaries.");

            typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
#endif

            auto && t = hpx::util::forward_as_tuple(std::forward<Args>(args)...);

            return for_loop_algo().call(
                std::forward<ExPolicy>(policy), is_seq(), first, size, stride,
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
    ///                     an (forward) iterator type or an integral type.
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
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a void
    ///           otherwise.
    ///
    template <typename ExPolicy, typename I, typename... Args,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
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
    /// equivalent to specifying \a parallel::execution::seq as the execution
    /// policy.
    ///
    /// \tparam I           The type of the iteration variable. This could be
    ///                     an (forward) iterator type or an integral type.
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

        return for_loop(parallel::execution::seq, first, last,
            std::forward<Args>(args)...);
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
    ///                     an (forward) iterator type or an integral type.
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
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a void
    ///           otherwise.
    ///
    template <typename ExPolicy, typename I, typename S, typename... Args,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
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
    /// equivalent to specifying \a parallel::execution::seq as the execution
    /// policy.
    ///
    /// \tparam I           The type of the iteration variable. This could be
    ///                     an (forward) iterator type or an integral type.
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

        return for_loop_strided(parallel::execution::seq, first, last, stride,
            std::forward<Args>(args)...);
    }

    /// The for_loop_n implements loop functionality over a range specified by
    /// integral or iterator bounds. For the iterator case, these algorithms
    /// resemble for_each from the Parallelism TS, but leave to the programmer
    /// when and if to dereference the iterator.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam I           The type of the iteration variable. This could be
    ///                     an (forward) iterator type or an integral type.
    /// \tparam Size        The type of a non-negative integral value specifying
    ///                     the number of items to iterate over.
    /// \tparam Args        A parameter pack, it's last element is a function
    ///                     object to be invoked for each iteration, the others
    ///                     have to be either conforming to the induction or
    ///                     reduction concept.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param size         Refers to the number of items the algorithm will be
    ///                     applied to.
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
    /// \returns  The \a for_loop_n algorithm returns a
    ///           \a hpx::future<void> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a void
    ///           otherwise.
    ///
    template <typename ExPolicy, typename I, typename Size, typename... Args,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        (hpx::traits::is_iterator<I>::value || std::is_integral<I>::value) &&
        std::is_integral<Size>::value)>
    typename util::detail::algorithm_result<ExPolicy>::type
    for_loop_n(ExPolicy && policy, I first, Size size, Args &&... args)
    {
        static_assert(sizeof...(Args) >= 1,
            "for_loop_n must be called with at least a function object");

        using hpx::util::detail::make_index_pack;
        return detail::for_loop_n(
            std::forward<ExPolicy>(policy), first, size, 1,
            typename make_index_pack<sizeof...(Args)-1>::type(),
            std::forward<Args>(args)...);
    }

    /// The for_loop implements loop functionality over a range specified by
    /// integral or iterator bounds. For the iterator case, these algorithms
    /// resemble for_each from the Parallelism TS, but leave to the programmer
    /// when and if to dereference the iterator.
    ///
    /// The execution of for_loop without specifying an execution policy is
    /// equivalent to specifying \a parallel::execution::seq as the execution
    /// policy.
    ///
    /// \tparam I           The type of the iteration variable. This could be
    ///                     an (forward) iterator type or an integral type.
    /// \tparam Size        The type of a non-negative integral value specifying
    ///                     the number of items to iterate over.
    /// \tparam Args        A parameter pack, it's last element is a function
    ///                     object to be invoked for each iteration, the others
    ///                     have to be either conforming to the induction or
    ///                     reduction concept.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param size         Refers to the number of items the algorithm will be
    ///                     applied to.
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
    template <typename I, typename Size, typename... Args,
    HPX_CONCEPT_REQUIRES_(
        (hpx::traits::is_iterator<I>::value || std::is_integral<I>::value) &&
        std::is_integral<Size>::value)>
    void for_loop_n(I first, Size size, Args &&... args)
    {
        static_assert(sizeof...(Args) >= 1,
            "for_loop_n must be called with at least a function object");

        return for_loop_n(parallel::execution::seq, first, size,
            std::forward<Args>(args)...);
    }

    /// The for_loop_n_strided implements loop functionality over a range
    /// specified by integral or iterator bounds. For the iterator case, these
    /// algorithms resemble for_each from the Parallelism TS, but leave to the
    /// programmer when and if to dereference the iterator.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam I           The type of the iteration variable. This could be
    ///                     an (forward) iterator type or an integral type.
    /// \tparam Size        The type of a non-negative integral value specifying
    ///                     the number of items to iterate over.
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
    /// \param size         Refers to the number of items the algorithm will be
    ///                     applied to.
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
    /// \returns  The \a for_loop_n_strided algorithm returns a
    ///           \a hpx::future<void> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a void
    ///           otherwise.
    ///
    template <typename ExPolicy, typename I, typename Size, typename S,
        typename... Args,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        (hpx::traits::is_iterator<I>::value || std::is_integral<I>::value) &&
        std::is_integral<Size>::value &&
        std::is_integral<S>::value)>
    typename util::detail::algorithm_result<ExPolicy>::type
    for_loop_n_strided(ExPolicy && policy, I first, Size size, S stride,
        Args &&... args)
    {
        static_assert(sizeof...(Args) >= 1,
            "for_loop_n_strided must be called with at least a function object");

        using hpx::util::detail::make_index_pack;
        return detail::for_loop_n(
            std::forward<ExPolicy>(policy), first, size, stride,
            typename make_index_pack<sizeof...(Args)-1>::type(),
            std::forward<Args>(args)...);
    }

    /// The for_loop_n_strided implements loop functionality over a range
    /// specified by integral or iterator bounds. For the iterator case, these
    /// algorithms resemble for_each from the Parallelism TS, but leave to the
    /// programmer when and if to dereference the iterator.
    ///
    /// The execution of for_loop without specifying an execution policy is
    /// equivalent to specifying \a parallel::execution::seq as the execution
    /// policy.
    ///
    /// \tparam I           The type of the iteration variable. This could be
    ///                     an (forward) iterator type or an integral type.
    /// \tparam Size        The type of a non-negative integral value specifying
    ///                     the number of items to iterate over.
    /// \tparam S           The type of the stride variable. This should be
    ///                     an integral type.
    /// \tparam Args        A parameter pack, it's last element is a function
    ///                     object to be invoked for each iteration, the others
    ///                     have to be either conforming to the induction or
    ///                     reduction concept.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param size         Refers to the number of items the algorithm will be
    ///                     applied to.
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
    template <typename I, typename Size, typename S, typename... Args,
    HPX_CONCEPT_REQUIRES_(
        (hpx::traits::is_iterator<I>::value || std::is_integral<I>::value) &&
        std::is_integral<Size>::value &&
        std::is_integral<S>::value)>
    void for_loop_n_strided(I first, Size size, S stride, Args &&... args)
    {
        static_assert(sizeof...(Args) >= 1,
            "for_loop_n_strided must be called with at least a function object");
        return for_loop_strided_n(parallel::execution::seq, first, size, stride,
            std::forward<Args>(args)...);
    }
}}}

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
namespace hpx { namespace traits
{
    template <typename F, typename S>
    struct get_function_address<
        parallel::v2::detail::part_iterations<F, S> >
    {
        static std::size_t call(
            parallel::v2::detail::part_iterations<F, S> const& f)
                noexcept
        {
            return get_function_address<
                    typename hpx::util::decay<F>::type
                >::call(f.f_);
        }
    };

    template <typename F, typename S>
    struct get_function_annotation<
        parallel::v2::detail::part_iterations<F, S> >
    {
        static char const* call(
            parallel::v2::detail::part_iterations<F, S> const& f)
                noexcept
        {
            return get_function_annotation<
                    typename hpx::util::decay<F>::type
                >::call(f.f_);
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename F, typename S>
    struct get_function_annotation_itt<
        parallel::v2::detail::part_iterations<F, S> >
    {
        static util::itt::string_handle call(
            parallel::v2::detail::part_iterations<F, S> const& f)
                noexcept
        {
            return get_function_annotation_itt<
                    typename hpx::util::decay<F>::type
                >::call(f.f_);
        }
    };
#endif
}}
#endif

#endif

