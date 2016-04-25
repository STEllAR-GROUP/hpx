//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/reduce_by_key.hpp

#if !defined(HPX_PARALLEL_ALGORITHM_REDUCE_BY_KEY_DEC_2015)
#define HPX_PARALLEL_ALGORITHM_REDUCE_BY_KEY_DEC_2015
//
#include <hpx/parallel/executors.hpp>
//
#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/parallel/algorithms/inclusive_scan.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/util/transform_iterator.hpp>
#include <hpx/util/tuple.hpp>
//
#include <vector>
//
#ifdef EXTRA_DEBUG
# include <iostream>
# define debug_reduce_by_key(a) std::cout << a
#else
# define debug_reduce_by_key(a)
#endif

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // reduce_by_key
    namespace detail {
        /// \cond NOINTERNAL

        // -------------------------------------------------------------------
        // simple iterator helper object for access to prev/next items
        // -------------------------------------------------------------------
        struct reduce_stencil_transformer
        {
            // declare result type as a template
            template<typename T>
            struct result;

            // specialize result for iterator type
            template<typename This, typename Iterator>
            struct result<This(Iterator)>
            {
                typedef typename std::iterator_traits<
                    Iterator
                >::reference element_type;
                typedef hpx::util::tuple<
                    element_type, element_type, element_type
                > type;
            };

            // call operator for stencil transform
            // it will dereference tuple(it-1, it, it+1)
            template<typename Iterator>
            typename result<reduce_stencil_transformer(Iterator)>::type
            operator()(Iterator const &it) const
            {
                typedef typename result<
                    reduce_stencil_transformer(Iterator)
                >::type type;
                return type(*std::prev(it), *it, *std::next(it));
            }
        };

        // -------------------------------------------------------------------
        // transform iterator using reduce_stencil_transformer helper
        // -------------------------------------------------------------------
        template<
            typename Iterator,
            typename Transformer = detail::reduce_stencil_transformer
        >
        class reduce_stencil_iterator : public hpx::util::transform_iterator<
            Iterator, Transformer
        >
        {
        private:
            typedef hpx::util::transform_iterator<
                Iterator, Transformer
            > base_type;

        public:
            reduce_stencil_iterator() { }

            explicit reduce_stencil_iterator(Iterator const &it) : base_type(it,
                Transformer()) { }

            reduce_stencil_iterator(Iterator const &it, Transformer const &t)
                : base_type(it, t) { }
        };

        template<typename Iterator, typename Transformer>
        inline reduce_stencil_iterator<
            Iterator, Transformer
        > make_reduce_stencil_iterator(Iterator const &it, Transformer const &t)
        {
            return reduce_stencil_iterator<Iterator, Transformer>(it, t);
        }

        // -------------------------------------------------------------------
        // state of a reduce by key step
        // -------------------------------------------------------------------
        struct reduce_key_series_states
        {
            bool start;    // START of a segment
            bool end;      // END of a segment
            reduce_key_series_states(bool s = false, bool e = false) :
                start(s), end(e) { }
        };

        // -------------------------------------------------------------------
        // functor that actually computes the state using the stencil iterator
        // -------------------------------------------------------------------
        template<
            typename Transformer, typename StencilIterType,
            typename KeyStateIterType, typename Compare
        >
        struct reduce_stencil_generate
        {
            typedef typename Transformer::template result<
                Transformer(StencilIterType)
            >::element_type element_type;
            typedef typename Transformer::template result<
                Transformer(StencilIterType)
            >::type tuple_type;
            typedef typename std::iterator_traits<
                KeyStateIterType
            >::reference KeyStateType;

            reduce_stencil_generate() { }

            void operator()(const tuple_type &value, KeyStateType &kiter,
                const Compare &comp) const
            {
                // resolves to a tuple of values for *(it-1), *it, *(it+1)

                element_type left = hpx::util::get<0>(value);
                element_type mid = hpx::util::get<1>(value);
                element_type right = hpx::util::get<2>(value);

                // we need to determine which of three states this
                // index is. It can be:
                // 1. Middle of a set of equivalent keys.
                // 2. Start of a set of equivalent keys.
                // 3. End of a set of equivalent keys.
                // 4. Both the start and end of a set of keys

                {
                    const bool leftMatches(comp(left, mid));
                    const bool rightMatches(comp(mid, right));
                    kiter = reduce_key_series_states(!leftMatches, !rightMatches);
                }
            }
        };

        // -------------------------------------------------------------------
        // helper that extracts the final output iterators from copy_if
        // -------------------------------------------------------------------
        // Zip iterator has 3 iterators inside
        // Iter1, key type : Iter2, value type : Iter3, state type
        template<typename ZIter, typename iKey, typename iVal>
        std::pair<iKey, iVal> make_pair_result(ZIter zipiter, iKey key_start,
            iVal val_start)
        {
            // the iterator we want is 'second' part of tagged_pair type (from copy_if)
            auto const &t = zipiter.second.get_iterator_tuple();
            iKey key_end = hpx::util::get<0>(t);
            return std::make_pair(key_end,
                std::next(val_start, std::distance(key_start, key_end)));
        }

        // async version that returns future<pair> from future<zip_iterator<blah>>
        template<typename ZIter, typename iKey, typename iVal>
        hpx::future<std::pair<iKey, iVal> > make_pair_result(
            hpx::future<ZIter> &&ziter, iKey key_start, iVal val_start)
        {
            typedef std::pair<iKey, iVal> result_type;

            return lcos::make_future<result_type>(std::move(ziter),
                [=](ZIter zipiter)
                {
                    auto const &t = zipiter.second.get_iterator_tuple();
                    iKey key_end = hpx::util::get<0>(t);
                    return std::make_pair(key_end,
                        std::next(val_start, std::distance(key_start, key_end)));
                });
        }

        // -------------------------------------------------------------------
        // when we are being run with an asynchronous policy, we do not want to
        // pass the policy directly to other algorithms we are using - as we
        // would have to wait internally on them before proceeding.
        // Instead create a new policy from the old one which removes the async/future
        // -------------------------------------------------------------------
        template<typename ExPolicy>
        struct remove_asynchronous
        {
            typedef ExPolicy type;
        };

        template<>
        struct remove_asynchronous<
            hpx::parallel::parallel_vector_execution_policy
        >
        {
            typedef hpx::parallel::parallel_execution_policy type;
        };

        template<>
        struct remove_asynchronous<
            hpx::parallel::sequential_task_execution_policy
        >
        {
            typedef hpx::parallel::sequential_execution_policy type;
        };

        template<>
        struct remove_asynchronous<hpx::parallel::parallel_task_execution_policy>
        {
            typedef hpx::parallel::parallel_execution_policy type;
        };

        // -------------------------------------------------------------------
        // The main algorithm is implemented here, it replaces any async
        // execution policy with a non async one so that no waits are
        // necessry on the internal algorithms. Async execution is handled
        // by the wrapper layer that calls this.
        // -------------------------------------------------------------------
        template<
            typename ExPolicy,
            typename RanIter, typename RanIter2, typename OutIter, typename OutIter2,
            typename Compare
            >
        static std::pair<OutIter, OutIter2>
        reduce_by_key_impl(ExPolicy &&policy, RanIter key_first, RanIter key_last,
            RanIter2 values_first, OutIter keys_output, OutIter2 values_output,
            Compare &&comp)
        {
            using namespace hpx::parallel::v1::detail;
            using namespace hpx::util;

            typedef typename detail::remove_asynchronous<
                typename std::decay<ExPolicy>::type>::type sync_policy_type;

            sync_policy_type sync_policy = sync_policy_type().on(policy.executor())
                .with(policy.parameters());

            // we need to determine based on the keys what is the keystate for
            // each key. The states are start, middle, end of a series and the special
            // state start and end of the sequence
            std::vector<reduce_key_series_states> key_state;
            typedef std::vector<reduce_key_series_states>::iterator keystate_iter_type;
            typedef detail::reduce_stencil_iterator<RanIter, reduce_stencil_transformer>
                reducebykey_iter;
            typedef typename std::iterator_traits<RanIter>::reference element_type;
            typedef typename zip_iterator<reducebykey_iter, keystate_iter_type>
              ::reference zip_ref;
            //
            const uint64_t number_of_keys = std::distance(key_first, key_last);
            //
            key_state.assign(number_of_keys, reduce_key_series_states());
            {
                reduce_stencil_transformer r_s_t;
                reducebykey_iter reduce_begin =
                    make_reduce_stencil_iterator(key_first, r_s_t);
                reducebykey_iter reduce_end =
                    make_reduce_stencil_iterator(key_last, r_s_t);

                if (number_of_keys == 2) {
                    // for two entries, one is a start, the other an end,
                    // if they are different, then they are both start/end
                    element_type left = *key_first;
                    element_type right = *std::next(key_first);
                    key_state[0] = reduce_key_series_states(true, !comp(left, right));
                    key_state[1] = reduce_key_series_states(!comp(left, right), true);
                } else {
                    // do the first and last elements by hand to simplify the iterator
                    // traversal as there is no prev/next for first/last
                    element_type elem0 = *key_first;
                    element_type elem1 = *std::next(key_first);
                    key_state[0] = reduce_key_series_states(true, elem0 != elem1);
                    // middle elements
                    reduce_stencil_generate <reduce_stencil_transformer, RanIter,
                    keystate_iter_type, Compare> kernel;
                    hpx::parallel::for_each(sync_policy,
                        make_zip_iterator(reduce_begin + 1, key_state.begin() + 1),
                        make_zip_iterator(reduce_end - 1, key_state.end() - 1),
                        [&kernel, &comp](zip_ref ref)
                        {
                            kernel.operator()(get<0>(ref), get<1>(ref), comp);
                        });
                    // Last element
                    element_type elemN = *std::prev(key_last);
                    element_type elemn = *std::prev(std::prev(key_last));
                    key_state.back() = reduce_key_series_states(elemN != elemn, true);
                }
            }
            {
                typedef zip_iterator<
                    RanIter2, std::vector<reduce_key_series_states>::iterator
                > zip_iterator_in;
                typedef typename zip_iterator_in::value_type zip_type_in;

                typedef zip_iterator<
                    OutIter2, std::vector<reduce_key_series_states>::iterator
                > zip_iterator_vout;
                typedef typename zip_iterator_vout::value_type zip_type_out;

                typedef typename std::iterator_traits<RanIter2>::value_type value_type;

                zip_iterator_in states_begin = make_zip_iterator(values_first,
                    std::begin(key_state));
                zip_iterator_in states_end = make_zip_iterator(
                    values_first + number_of_keys, std::end(key_state));
                zip_iterator_vout states_out_begin = make_zip_iterator(values_output,
                    std::begin(key_state));
                //
                zip_type_in initial = tuple<float, reduce_key_series_states>(0.0,
                    reduce_key_series_states(true, false));
                //
                hpx::parallel::inclusive_scan(sync_policy, states_begin,
                    states_end, states_out_begin, initial,
                    // B is the current entry, A is the one passed in from 'previous'
                    [](zip_type_in a, zip_type_in b)
                    {
                        value_type a_val = get<0>(a);
                        reduce_key_series_states a_state = get<1>(a);
                        value_type b_val = get<0>(b);
                        reduce_key_series_states b_state = get<1>(b);
                        debug_reduce_by_key(
                            "{ " << a_val << "+" << b_val << " },\t" << a_state <<
                            b_state);
                        // if carrying a start flag, then copy - don't add
                        if (b_state.start) {
                            debug_reduce_by_key(" = " << b_val << std::endl);
                            return make_tuple(b_val, reduce_key_series_states(
                                    a_state.start || b_state.start, b_state.end));
                        }
                            // normal add of previous + this
                        else {
                            debug_reduce_by_key(" = " << a_val + b_val << std::endl);
                            return make_tuple(a_val + b_val, reduce_key_series_states(
                                    a_state.start || b_state.start, b_state.end));
                        }
                    });

                // now copy the values and keys for each element that
                // is marked by an 'END' state to the final output
                typedef typename hpx::util::zip_iterator<
                    RanIter, OutIter2, std::vector<reduce_key_series_states>::iterator
                >::reference zip2_ref;

                return make_pair_result(std::move(
                    hpx::parallel::copy_if(sync_policy,
                        make_zip_iterator(key_first, values_output,
                            std::begin(key_state)),
                        make_zip_iterator(key_last, values_output + number_of_keys,
                            std::end(key_state)),
                        make_zip_iterator(keys_output, values_output,
                            std::begin(key_state)),
                        // copies to dest only when 'end' state is true
                        [](zip2_ref it)
                        {
                            return get<2>(it).end;
                        })), keys_output, values_output);
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // reduce_by_key wrapper struct
        template<typename OutIter, typename OutIter2>
        struct reduce_by_key : public detail::algorithm<
            reduce_by_key<OutIter, OutIter2>,
            std::pair<OutIter, OutIter2>>
            {
            reduce_by_key()
                  : reduce_by_key::algorithm("reduce_by_key")
            {}

            template <
                typename ExPolicy, typename RanIter, typename RanIter2,
                typename Compare>
            static std::pair<OutIter, OutIter2>
            sequential(ExPolicy &&policy, RanIter key_first, RanIter key_last,
                RanIter2 values_first, OutIter keys_output, OutIter2 values_output,
                Compare && comp)
                {
                return reduce_by_key_impl(
                    policy, key_first, key_last,
                    values_first, keys_output, values_output,
                    std::forward<Compare>(comp));
                }

            template <
                typename ExPolicy, typename RanIter, typename RanIter2,
                typename Compare>
            static typename
                util::detail::algorithm_result<ExPolicy,
                std::pair<OutIter, OutIter2>>::type
            parallel(ExPolicy &&policy, RanIter key_first, RanIter key_last,
                RanIter2 values_first, OutIter keys_output, OutIter2 values_output,
                Compare && comp)
            {
                typedef typename hpx::util::decay<ExPolicy>::type::executor_type
                    executor_type;

                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;

                return util::detail::algorithm_result<ExPolicy,
                    std::pair<OutIter, OutIter2>>::get(
                    executor_traits::async_execute(
                        policy.executor(),
                        hpx::util::deferred_call(
                            &hpx::parallel::v1::detail::reduce_by_key_impl<
                                ExPolicy&, RanIter, RanIter2,
                                OutIter, OutIter2, Compare&>,
                            std::ref(policy), key_first, key_last,
                            values_first, keys_output, values_output,
                            std::ref(comp))));
            }
        };
        /// \endcond
    }

#ifdef EXTRA_DEBUG
    std::ostream &operator<<(std::ostream &os,
        const detail::reduce_key_series_states &rs)
    {
        os << "{ start=" << rs.start << ",end=" << rs.end << "} ";
        return os;
    }
#endif

    //-----------------------------------------------------------------------------
    /// Reduce by Key performs an inclusive scan reduction operation on elements
    /// supplied in key/value pairs. The algorithm produces a single output
    /// value for each set of equal consecutive keys in [key_first, key_last).
    /// the value being the
    /// GENERALIZED_NONCOMMUTATIVE_SUM(op, init, *first, ..., *(first + (i - result))).
    /// for the run of consecutive matching keys.
    /// The number of keys supplied must match the number of values.
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a op.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam RanIter     The type of the key iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam RanIter2    The type of the value iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination key range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam OutIter2    The type of the iterator representing the
    ///                     destination value range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Compare     The type of the optional function/function object to use
    ///                     to compare keys (deduced).
    ///                     Assumed to be std::equal_to otherwise.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param key_first    Refers to the beginning of the sequence of key elements
    ///                     the algorithm will be applied to.
    /// \param key_last     Refers to the end of the sequence of key elements the
    ///                     algorithm will be applied to.
    /// \param value_first  Refers to the beginning of the sequence of value elements
    ///                     the algorithm will be applied to.
    /// \param keys_output  Refers to the start output location for the keys
    ///                     produced by the algorithm.
    /// \param values_output Refers to the start output location for the values
    ///                     produced by the algorithm.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    ///
    /// \a comp has to induce a strict weak ordering on the values.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequential_execution_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_execution_policy or \a parallel_task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a reduce_by_key algorithm returns a
    ///           \a hpx::future<pair<Iter1,Iter2>> if the execution policy is of
    ///           type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and returns \a pair<Iter1,Iter2>
    ///           otherwise.
    //-----------------------------------------------------------------------------

    template<
        typename ExPolicy,
        typename RanIter, typename RanIter2, typename OutIter, typename OutIter2,
        typename Compare =
            std::equal_to<typename std::iterator_traits<RanIter>::value_type>,
        HPX_CONCEPT_REQUIRES_(is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<RanIter>::value &&
            hpx::traits::is_iterator<RanIter2>::value &&
            hpx::traits::is_iterator<OutIter>::value &&
            hpx::traits::is_iterator<OutIter2>::value
        )
    >

    typename util::detail::algorithm_result<ExPolicy, std::pair<OutIter, OutIter2>>::type
        reduce_by_key(ExPolicy &&policy, RanIter key_first, RanIter key_last,
            RanIter2 values_first, OutIter keys_output, OutIter2 values_output,
            Compare &&comp = Compare())
    {
        typedef util::detail::algorithm_result<
            ExPolicy, std::pair<OutIter, OutIter2>> result;

        static_assert(
            (hpx::traits::is_random_access_iterator<RanIter>::value) &&
            (hpx::traits::is_random_access_iterator<RanIter2>::value) &&
            (hpx::traits::is_output_iterator<OutIter>::value ||
                hpx::traits::is_forward_iterator<OutIter>::value) &&
            (hpx::traits::is_output_iterator<OutIter>::value ||
                hpx::traits::is_forward_iterator<OutIter>::value),
            "iterators : Random_access for inputs and Output for outputs.");

        const uint64_t number_of_keys = std::distance(key_first, key_last);

        if (number_of_keys <= 1)
        { // we only have a single key/value so that is our output
            *keys_output = *key_first;
            *values_output = *values_first;
            return result::get(std::make_pair(keys_output, values_output));
        }

        typedef is_sequential_execution_policy<ExPolicy> is_seq;

        return detail::reduce_by_key<OutIter,OutIter2>().call(
            std::forward<ExPolicy>(policy), is_seq(), key_first, key_last,
            values_first, keys_output, values_output,
            std::forward<Compare>(comp));
    }

}}}

#endif
