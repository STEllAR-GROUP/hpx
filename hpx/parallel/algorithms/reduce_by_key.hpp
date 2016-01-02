//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_REDUCE_BY_KEY_DEC_2015)
#define HPX_PARALLEL_ALGORITHM_REDUCE_BY_KEY_DEC_2015

#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/parallel/algorithms/detail/tuple_iterator.hpp>
#include <hpx/parallel/algorithms/prefix_scan.hpp>
#include <hpx/parallel/algorithms/inclusive_scan.hpp>
#include <hpx/parallel/algorithms/exclusive_scan.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/util/transform_iterator.hpp>
#include <hpx/util/tuple.hpp>
//
#include <boost/thread/locks.hpp>
//
#define VTKM_CONT_EXPORT
#define VTKM_EXEC_EXPORT

#ifdef EXTRA_DEBUG
# define debug_reduce_by_key(a) std::cout << a
#else
# define debug_reduce_by_key(a)
#endif

typedef hpx::lcos::local::shared_mutex mutex_type;

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // reduce_by_key
    namespace detail
    {
        /// \cond NOINTERNAL

        // -------------------------------------------------------------------
        // simple iterator helper object for access to prev/next items
        // -------------------------------------------------------------------
        struct reduce_stencil_transformer
        {
            // declare result type as a template
            template <typename T> struct result;

            // specialize result for iterator type
            template <typename This, typename Iterator>
            struct result<This(Iterator)>
            {
                typedef typename std::iterator_traits<Iterator>::reference element_type;
                typedef hpx::util::tuple<element_type, element_type, element_type> type;
            };

            // call operator for stencil transform
            // it will dereference tuple(it-1, it, it+1)
            template <typename Iterator>
            typename result<reduce_stencil_transformer(Iterator)>::type
            operator()(Iterator const& it) const
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
        template <typename Iterator,
                typename Transformer = detail::reduce_stencil_transformer>
        class reduce_stencil_iterator
                : public hpx::util::transform_iterator<Iterator, Transformer>
        {
        private:
            typedef hpx::util::transform_iterator<Iterator, Transformer> base_type;

        public:

            reduce_stencil_iterator() {}

            explicit reduce_stencil_iterator(Iterator const& it)
                    : base_type(it, Transformer())
            {}

            reduce_stencil_iterator(Iterator const& it, Transformer const& t)
                    : base_type(it, t)
            {}
        };

        template <typename Iterator, typename Transformer>
        inline reduce_stencil_iterator<Iterator, Transformer>
        make_reduce_stencil_iterator(Iterator const& it, Transformer const& t)
        {
            return reduce_stencil_iterator<Iterator, Transformer>(it, t);
        }

        // -------------------------------------------------------------------
        // state of a reduce by key step
        // -------------------------------------------------------------------
        struct ReduceKeySeriesStates
        {
            bool fStart;    // START of a segment
            bool fEnd;      // END of a segment
            ReduceKeySeriesStates(bool start=false, bool end=false) :
                fStart(start), fEnd(end) {}
        };

        template<typename Transformer, typename StencilIterType, typename KeyStateIterType >
        struct ReduceStencilGeneration
        {
            //
            typedef typename Transformer::template result<Transformer(StencilIterType)>::element_type element_type;
            typedef typename Transformer::template result<Transformer(StencilIterType)>::type tuple_type;
            typedef typename std::iterator_traits<KeyStateIterType>::reference KeyStateType;

            VTKM_CONT_EXPORT
            ReduceStencilGeneration() {}

            VTKM_EXEC_EXPORT
            void operator()(const tuple_type &value, KeyStateType &kiter) const
            {
                // resolves to a tuple of values for *(it-1), *it, *(it+1)

                element_type left  = hpx::util::get<0>(value);
                element_type mid   = hpx::util::get<1>(value);
                element_type right = hpx::util::get<2>(value);

                // we need to determine which of three states this
                // index is. It can be:
                // 1. Middle of a set of equivalent keys.
                // 2. Start of a set of equivalent keys.
                // 3. End of a set of equivalent keys.
                // 4. Both the start and end of a set of keys

                {
                    const bool leftMatches(left == mid);
                    const bool rightMatches(right == mid);
                    kiter = ReduceKeySeriesStates(!leftMatches, !rightMatches);
                }
            }
        };

        // Zip iterator has 3 iterators inside
        // Iter1, key type : Iter2, value type : Iter3, state type
        template <typename ZIter, typename iKey, typename iVal>
        std::pair< iKey, iVal>
        make_pair_result(ZIter zipiter, iKey key_start, iVal val_start)
        {
            // tagged_pair type (from copy_if) has second item the iterator we want
            auto const& t = zipiter.second.get_iterator_tuple();
            iKey key_end = hpx::util::get<0>(t);
            return std::make_pair(key_end,
                                  std::next(val_start, std::distance(key_start, key_end)));
        }
/*
        template <typename Iter1, typename Iter2>
        hpx::future< std::pair< Iter1, Iter2> >
        make_pair_result(hpx::future<Iter1> && iter1, Iter1 ibegin, Iter2 iter2)
        {
            typedef std::pair< Iter1, Iter2 > result_type;

            return lcos::make_future<result_type>(
                    std::move(iter1),
                    [=](Iter1 iter1){
                        return std::make_pair(iter1,std::next(iter2,std::distance(ibegin, iter1)));
                    });
        }
*/
        /// \endcond
    }

    std::ostream& operator << (std::ostream &os, const detail::ReduceKeySeriesStates &rs) {
        os << "{ start=" << rs.fStart << ",end=" << rs.fEnd << "} ";
        return os;
    }

    //-----------------------------------------------------------------------------
    /// Sorts the elements in the range [first, last) in ascending order. The
    /// order of equal elements is not guaranteed to be preserved. The function
    /// uses the given comparison function object comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity: O(Nlog(N)), where N = std::distance(first, last)
    ///                     comparisons.
    ///
    /// A sequence is sorted with respect to a comparator \a comp and a
    /// projection \a proj if for every iterator i pointing to the sequence and
    /// every non-negative integer n such that i + n is a valid iterator
    /// pointing to an element of the sequence, and
    /// INVOKE(comp, INVOKE(proj, *(i + n)), INVOKE(proj, *i)) == false.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam Iter        The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
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
    /// \returns  The \a sort algorithm returns a
    ///           \a hpx::future<Iter> if the execution policy is of
    ///           type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and returns \a Iter
    ///           otherwise.
    ///           It returns \a last.
    //-----------------------------------------------------------------------------

    template <typename Proj = util::projection_identity,
        typename ExPolicy, typename RanIter, typename RanIter2, typename OutIter, typename OutIter2,
        typename Compare = std::less<
            typename std::remove_reference<
                typename traits::projected_result_of<Proj, RanIter>::type
            >::type
        >,
        HPX_CONCEPT_REQUIRES_(
            is_execution_policy<ExPolicy>::value &&
            traits::detail::is_iterator<RanIter>::value &&
            traits::is_projected<Proj, RanIter>::value &&
            traits::is_indirect_callable<
                Compare,
                traits::projected<Proj, RanIter>,
                traits::projected<Proj, RanIter>
            >::value)>

    typename util::detail::algorithm_result<ExPolicy, std::pair<OutIter,OutIter2> >::type
    reduce_by_key(
        ExPolicy && policy,
        RanIter key_first,
        RanIter key_last,
        RanIter2 values_first,
        OutIter keys_output,
        OutIter2 values_output,
        Compare && comp = Compare(), Proj && proj = Proj())
    {
        typedef typename std::iterator_traits<RanIter>::iterator_category
                iterator_category1;
        typedef typename std::iterator_traits<RanIter2>::iterator_category
                iterator_category2;
        typedef typename std::iterator_traits<OutIter>::iterator_category
                iterator_category3;
        typedef typename std::iterator_traits<OutIter2>::iterator_category
                iterator_category4;

        static_assert(
            (boost::is_base_of<
                std::random_access_iterator_tag, iterator_category1>::value) ||
            (boost::is_base_of<
                std::random_access_iterator_tag, iterator_category2>::value) ||
            (boost::is_base_of<
                std::output_iterator_tag, iterator_category3>::value) ||
            (boost::is_base_of<
                std::output_iterator_tag, iterator_category4>::value),
            "iterators : Random_access for inputs and Output for outputs.");

        typedef is_sequential_execution_policy<ExPolicy> is_seq;

        const uint64_t numberOfKeys = std::distance(key_first, key_last);

        if(numberOfKeys <= 1)
        { // we only have a single key/value so that is our output
            *keys_output = *key_first;
            *values_output = *values_first;
            return std::make_pair(keys_output,values_output);
        }

        using namespace hpx::parallel::detail;

        //we need to determine based on the keys what is the keystate for
        //each key. The states are start, middle, end of a series and the special
        //state start and end of a series
        std::vector< ReduceKeySeriesStates > keystate;
        typedef std::vector< ReduceKeySeriesStates >::iterator KeyStateIterType;
        typedef typename detail::reduce_stencil_iterator<RanIter, detail::reduce_stencil_transformer> reducebykey_iter;
        typedef typename std::iterator_traits<RanIter>::reference element_type;
        typedef typename hpx::util::zip_iterator<reducebykey_iter, KeyStateIterType>::reference zip_ref;
        keystate.assign(numberOfKeys, ReduceKeySeriesStates());
        {
            detail::reduce_stencil_transformer r_s_t;
            reducebykey_iter reduce_begin = make_reduce_stencil_iterator(key_first, r_s_t);
            reducebykey_iter reduce_end   = make_reduce_stencil_iterator(key_last, r_s_t);


            if (numberOfKeys==2) {
                // for two entries, one is a start, the other an end,
                // if they are different, then they are both start/end
                element_type left  = *key_first;
                element_type right = *std::next(key_first);
                keystate[0] = ReduceKeySeriesStates(true, left!=right);
                keystate[1] = ReduceKeySeriesStates(left!=right, true);
            }
            else {
                // do the first element and last one by hand to simplify the iterator
                // traversal as there is no prev/next for first/last
                element_type elem0 = *key_first;
                element_type elem1 = *std::next(key_first);
                keystate[0] = ReduceKeySeriesStates(true, elem0!=elem1);
                //
                ReduceStencilGeneration <detail::reduce_stencil_transformer, RanIter, KeyStateIterType> kernel;
                hpx::parallel::for_each(
                        policy,
                        hpx::util::make_zip_iterator(reduce_begin + 1, keystate.begin() + 1),
                        hpx::util::make_zip_iterator(reduce_end - 1, keystate.end() - 1),
                        [&kernel](zip_ref ref) {
                            kernel.operator()(hpx::util::get<0>(ref), hpx::util::get<1>(ref));
                        }
                );
                //
                element_type elemN = *std::prev(key_last);
                element_type elemn = *std::prev(std::prev(key_last));
                keystate.back() = ReduceKeySeriesStates(elemN!=elemn, true);
            }
        }
        {
            typedef hpx::util::zip_iterator<RanIter2, std::vector< ReduceKeySeriesStates >::iterator> zip_iterator;
            typedef std::vector< ReduceKeySeriesStates >::iterator rki;
            typedef typename zip_iterator::value_type zip_type;
            typedef typename std::iterator_traits<RanIter2>::value_type value_type;

            zip_iterator states_begin = hpx::util::make_zip_iterator(
                    values_first, std::begin(keystate));
            zip_iterator states_end = hpx::util::make_zip_iterator(
                    values_first + numberOfKeys, std::end(keystate));
            zip_iterator states_out_begin = hpx::util::make_zip_iterator(
                    values_output, std::begin(keystate));

            zip_type initial = hpx::util::tuple<float, ReduceKeySeriesStates>(0.0, ReduceKeySeriesStates(true, false));
            // caution : if the current value is a start value, then it should be used
            // otherwise we add the incoming B to our A.
            // BUT, the partitions are updated with values carried from if the when summing we can add values if the current value is not a start
            hpx::parallel::prefix_scan_inclusive(
                    policy,
                    states_begin,
                    states_end,
                    states_out_begin,
                    initial,
                    // B is the current entry, A is the one passed in from 'previous'
                    [](zip_type a, zip_type b) {
                        value_type            a_val   = hpx::util::get<0>(a);
                        ReduceKeySeriesStates a_state = hpx::util::get<1>(a);
                        value_type            b_val   = hpx::util::get<0>(b);
                        ReduceKeySeriesStates b_state = hpx::util::get<1>(b);
                        debug_reduce_by_key(
                                "{ " << a_val << "+" << b_val << " },\t" << a_state << b_state);
                        // if carrying a start flag, then copy - don't add
                        if (b_state.fStart) {
                            debug_reduce_by_key(" = " << b_val << std::endl);
                            return hpx::util::make_tuple(
                                     b_val,
                                    ReduceKeySeriesStates(a_state.fStart || b_state.fStart, b_state.fEnd));
                        }
                        // normal add of previous + this
                        else {
                            debug_reduce_by_key(" = " << a_val + b_val << std::endl);
                            return hpx::util::make_tuple(
                                     a_val + b_val,
                                    ReduceKeySeriesStates(a_state.fStart || b_state.fStart, b_state.fEnd));
                        }
                    }
            );

            // now copy the values and keys for each element that
            // is marked by an 'END' state to the final output
            typedef hpx::util::zip_iterator<
                    RanIter,
                    OutIter2,
                    std::vector< ReduceKeySeriesStates >::iterator> zip_iterator2;
            typedef typename zip_iterator2::reference zip2_ref;
            typedef typename zip_iterator2::value_type zip2_val;

            zip_iterator2 compact_begin = hpx::util::make_zip_iterator(
                    key_first,
                    values_output,
                    std::begin(keystate));
            zip_iterator2 compact_end = hpx::util::make_zip_iterator(
                    key_last,
                    values_output + numberOfKeys,
                    std::end(keystate));

            // auto result = std::make_pair(key_last, values_output + numberOfKeys);

            // @TODO : fix this to write keys to output array instead of input
            auto result = detail::make_pair_result(
                hpx::parallel::copy_if(
                    policy,
                    compact_begin,
                    compact_end,
                    compact_begin,
                    // return true if this is an end state
                    [](zip2_ref it) {
                        return hpx::util::get< 2 >(it).fEnd;
                    }
                ), key_first, values_output);

            return result;
        }
    }
        }
}}

#endif
