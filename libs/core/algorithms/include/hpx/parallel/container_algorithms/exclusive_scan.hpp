//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/exclusive_scan.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(+, init, *first, ...,
    /// *(first + (i - result) - 1))
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a std::plus<T>.
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked without an execution policy object will execute in sequential
    /// order in the calling thread.
    ///
    /// \returns  The \a exclusive_scan algorithm returns \a
    ///           util::in_out_result<InIter, OutIter>.
    ///           The \a exclusive_scan algorithm returns an input iterator to
    ///           the point denoted by the sentinel and an output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aK)
    ///           + GENERALIZED_NONCOMMUTATIVE_SUM(+, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum.
    ///
    template <typename InIter, typename Sent, typename OutIter, typename T>
    exclusive_scan_result<InIter, OutIter> exclusive_scan(
        InIter first, Sent last, OutIter dest, T init);

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(+, init, *first, ...,
    /// *(first + (i - result) - 1))
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a std::plus<T>.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a exclusive_scan algorithm returns a
    ///           \a hpx::future<util::in_out_result<FwdIter1, FwdIter2>> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a util::in_out_result<FwdIter1, FwdIter2> otherwise.
    ///           The \a exclusive_scan algorithm returns an input iterator to
    ///           the point denoted by the sentinel and an output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aK)
    ///           + GENERALIZED_NONCOMMUTATIVE_SUM(+, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent,
        typename FwdIter2, typename T>
    typename util::detail::algorithm_result<ExPolicy,
        exclusive_scan_result<FwdIter1, FwdIter2>>::type
    exclusive_scan(
        ExPolicy&& policy, FwdIter1 first, Sent last, FwdIter2 dest, T init);

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, init, *first, ...,
    /// *(first + (i - result) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a op.
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Op          The type of the binary function object used for
    ///                     the reduction operation.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    /// \param op           Specifies the function (or function object) which
    ///                     will be invoked for each of the values of the input
    ///                     sequence. This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The types \a Type1 and \a Ret must be
    ///                     such that an object of a type as given by the input
    ///                     sequence can be implicitly converted to any
    ///                     of those types.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked without an execution policy object will execute in sequential
    /// order in the calling thread.
    ///
    /// \returns  The \a exclusive_scan algorithm returns \a
    ///           util::in_out_result<InIter, OutIter>.
    ///           The \a exclusive_scan algorithm returns an input iterator to
    ///           the point denoted by the sentinel and an output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN))
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum. If
    /// \a op is not mathematically associative, the behavior of
    /// \a inclusive_scan may be non-deterministic.
    ///
    template <typename InIter, typename Sent, typename OutIter, typename T,
        typename Op>
    exclusive_scan_result<InIter, OutIter> exclusive_scan(
        InIter first, Sent last, OutIter dest, T init, Op&& op);

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, init, *first, ...,
    /// *(first + (i - result) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a op.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter1.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Op          The type of the binary function object used for
    ///                     the reduction operation.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    /// \param op           Specifies the function (or function object) which
    ///                     will be invoked for each of the values of the input
    ///                     sequence. This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The types \a Type1 and \a Ret must be
    ///                     such that an object of a type as given by the input
    ///                     sequence can be implicitly converted to any
    ///                     of those types.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a exclusive_scan algorithm returns a
    ///           \a hpx::future<util::in_out_result<FwdIter1, FwdIter2>> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a util::in_out_result<FwdIter1, FwdIter2> otherwise.
    ///           The \a exclusive_scan algorithm returns an input iterator to
    ///           the point denoted by the sentinel and an output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN))
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum. If
    /// \a op is not mathematically associative, the behavior of
    /// \a inclusive_scan may be non-deterministic.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent,
        typename FwdIter2, typename T, typename Op>
    typename util::detail::algorithm_result<ExPolicy,
        exclusive_scan_result<FwdIter1, FwdIter2>>::type
    exclusive_scan(ExPolicy&& policy, FwdIter1 first, Sent last, FwdIter2 dest,
        T init, Op&& op);

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(+, init, *first, ...,
    /// *(first + (i - result) - 1))
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a std::plus<T>.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked without an execution policy object will execute in sequential
    /// order in the calling thread.
    ///
    /// \returns  The \a exclusive_scan algorithm returns
    ///           \a util::in_out_result<traits::range_iterator_t<Rng>, O>
    ///           The \a exclusive_scan algorithm returns an input iterator to
    ///           the point denoted by the sentinel and an output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aK)
    ///           + GENERALIZED_NONCOMMUTATIVE_SUM(+, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum.
    ///
    template <typename Rng, typename O, typename T>
    exclusive_scan_result<traits::range_iterator_t<Rng>, O> exclusive_scan(
        Rng&& rng, O dest, T init);

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(+, init, *first, ...,
    /// *(first + (i - result) - 1))
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a std::plus<T>.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a exclusive_scan algorithm returns a
    ///           \a hpx::future<util::in_out_result
    ///           <traits::range_iterator_t<Rng>, O>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a util::in_out_result
    ///           <traits::range_iterator_t<Rng>, O>
    ///           otherwise.
    ///           The \a exclusive_scan algorithm returns an input iterator to
    ///           the point denoted by the sentinel and an output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aK)
    ///           + GENERALIZED_NONCOMMUTATIVE_SUM(+, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum.
    ///
    template <typename ExPolicy, typename Rng, typename O, typename T>
    typename util::detail::algorithm_result<ExPolicy,
        exclusive_scan_result<traits::range_iterator_t<Rng>, O>>::type
    exclusive_scan(ExPolicy&& policy, Rng&& rng, O dest, T init);

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(+, init, *first, ...,
    /// *(first + (i - result) - 1))
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a std::plus<T>.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Op          The type of the binary function object used for
    ///                     the reduction operation.
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    /// \param op           Specifies the function (or function object) which
    ///                     will be invoked for each of the values of the input
    ///                     sequence. This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The types \a Type1 and \a Ret must be
    ///                     such that an object of a type as given by the input
    ///                     sequence can be implicitly converted to any
    ///                     of those types.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked without an execution policy object will execute in sequential
    /// order in the calling thread.
    ///
    /// \returns  The \a exclusive_scan algorithm returns
    ///           \a util::in_out_result<traits::range_iterator_t<Rng>, O>
    ///           The \a exclusive_scan algorithm returns an input iterator to
    ///           the point denoted by the sentinel and an output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aK)
    ///           + GENERALIZED_NONCOMMUTATIVE_SUM(+, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum.
    ///
    template <typename Rng, typename O, typename T, typename Op>
    exclusive_scan_result<traits::range_iterator_t<Rng>, O> exclusive_scan(
        Rng&& rng, O dest, T init, Op&& op);

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(+, init, *first, ...,
    /// *(first + (i - result) - 1))
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a std::plus<T>.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Op          The type of the binary function object used for
    ///                     the reduction operation.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    /// \param op           Specifies the function (or function object) which
    ///                     will be invoked for each of the values of the input
    ///                     sequence. This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The types \a Type1 and \a Ret must be
    ///                     such that an object of a type as given by the input
    ///                     sequence can be implicitly converted to any
    ///                     of those types.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a exclusive_scan algorithm returns a
    ///           \a hpx::future<util::in_out_result
    ///           <traits::range_iterator_t<Rng>, O>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a util::in_out_result
    ///           <traits::range_iterator_t<Rng>, O>
    ///           otherwise.
    ///           The \a exclusive_scan algorithm returns an input iterator to
    ///           the point denoted by the sentinel and an output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aK)
    ///           + GENERALIZED_NONCOMMUTATIVE_SUM(+, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum.
    ///
    template <typename ExPolicy, typename Rng, typename O, typename T,
        typename Op>
    typename util::detail::algorithm_result<ExPolicy,
        exclusive_scan_result<traits::range_iterator_t<Rng>, O>>::type
    exclusive_scan(ExPolicy&& policy, Rng&& rng, O dest, T init, Op&& op);
}}    // namespace hpx::ranges
#else

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/algorithms/exclusive_scan.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace ranges {

    template <typename I, typename O>
    using exclusive_scan_result = parallel::util::in_out_result<I, O>;

    inline constexpr struct exclusive_scan_t final
      : hpx::detail::tag_parallel_algorithm<exclusive_scan_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename Sent, typename OutIter,
            typename T = typename std::iterator_traits<InIter>::value_type,
            typename Op = std::plus<T>,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<InIter> &&
                hpx::traits::is_sentinel_for<Sent, InIter>::value &&
                hpx::traits::is_iterator_v<OutIter> &&
                hpx::is_invocable_v<Op,
                    typename std::iterator_traits<InIter>::value_type,
                    typename std::iterator_traits<InIter>::value_type
                >
            )>
        // clang-format on
        friend exclusive_scan_result<InIter, OutIter> tag_fallback_invoke(
            hpx::ranges::exclusive_scan_t, InIter first, Sent last,
            OutIter dest, T init, Op&& op = Op())
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter>,
                "Requires at least input iterator.");
            static_assert(hpx::traits::is_output_iterator_v<OutIter>,
                "Requires at least output iterator.");

            using result_type = exclusive_scan_result<InIter, OutIter>;

            return hpx::parallel::v1::detail::exclusive_scan<result_type>()
                .call(hpx::execution::seq, first, last, dest, HPX_MOVE(init),
                    HPX_FORWARD(Op, op));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent,
            typename FwdIter2,
            typename T = typename std::iterator_traits<FwdIter1>::value_type,
            typename Op = std::plus<T>,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator_v<FwdIter1> &&
                hpx::traits::is_sentinel_for<Sent, FwdIter1>::value &&
                hpx::traits::is_iterator_v<FwdIter2> &&
                hpx::is_invocable_v<Op,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter1>::value_type
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            exclusive_scan_result<FwdIter1, FwdIter2>>::type
        tag_fallback_invoke(hpx::ranges::exclusive_scan_t, ExPolicy&& policy,
            FwdIter1 first, Sent last, FwdIter2 dest, T init, Op&& op = Op())
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter1>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter2>,
                "Requires at least forward iterator.");

            using result_type = exclusive_scan_result<FwdIter1, FwdIter2>;

            return hpx::parallel::v1::detail::exclusive_scan<result_type>()
                .call(HPX_FORWARD(ExPolicy, policy), first, last, dest,
                    HPX_MOVE(init), HPX_FORWARD(Op, op));
        }

        // clang-format off
        template <typename Rng, typename O,
            typename T = typename std::iterator_traits<
                hpx::traits::range_iterator_t<Rng>>::value_type,
            typename Op = std::plus<T>,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::is_invocable_v<Op,
                    typename hpx::traits::range_traits<Rng>::value_type,
                    typename hpx::traits::range_traits<Rng>::value_type
                >
            )>
        // clang-format on
        friend exclusive_scan_result<traits::range_iterator_t<Rng>, O>
        tag_fallback_invoke(hpx::ranges::exclusive_scan_t, Rng&& rng, O dest,
            T init, Op&& op = Op())
        {
            static_assert(hpx::traits::is_input_iterator<
                              traits::range_iterator_t<Rng>>::value,
                "Requires at least input iterator.");

            using result_type =
                exclusive_scan_result<traits::range_iterator_t<Rng>, O>;

            return hpx::parallel::v1::detail::exclusive_scan<result_type>()
                .call(hpx::execution::seq, std::begin(rng), std::end(rng), dest,
                    HPX_MOVE(init), HPX_FORWARD(Op, op));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,  typename O,
            typename T = typename std::iterator_traits<
                hpx::traits::range_iterator_t<Rng>>::value_type,
            typename Op = std::plus<T>,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::is_invocable_v<Op,
                    typename hpx::traits::range_traits<Rng>::value_type,
                    typename hpx::traits::range_traits<Rng>::value_type
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            exclusive_scan_result<traits::range_iterator_t<Rng>, O>>::type
        tag_fallback_invoke(hpx::ranges::exclusive_scan_t, ExPolicy&& policy,
            Rng&& rng, O dest, T init, Op&& op = Op())
        {
            static_assert(hpx::traits::is_forward_iterator<
                              traits::range_iterator_t<Rng>>::value,
                "Requires at least forward iterator.");

            using result_type =
                exclusive_scan_result<traits::range_iterator_t<Rng>, O>;

            return hpx::parallel::v1::detail::exclusive_scan<result_type>()
                .call(HPX_FORWARD(ExPolicy, policy), std::begin(rng),
                    std::end(rng), dest, HPX_MOVE(init), HPX_FORWARD(Op, op));
        }
    } exclusive_scan{};
}}    // namespace hpx::ranges

#endif
