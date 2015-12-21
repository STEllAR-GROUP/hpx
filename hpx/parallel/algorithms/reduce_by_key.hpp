//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_REDUCE_BY_KEY_DEC_2015)
#define HPX_PARALLEL_ALGORITHM_REDUCE_BY_KEY_DEC_2015

#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/parallel/algorithms/detail/tuple_iterator.hpp>

#define VTKM_CONT_EXPORT
#define VTKM_EXEC_EXPORT
namespace vtkm {
    typedef uint64_t Id;

    template<typename T1, typename T2> using Pair = std::pair<T1, T2>;
}

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // sort
    namespace detail
    {
        /// \cond NOINTERNAL

        struct ReduceKeySeriesStates
        {
            bool fStart;    // START of a segment
            bool fEnd;      // END of a segment
            ReduceKeySeriesStates(bool start=false, bool end=false) : fStart(start), fEnd(end) {}
        };

        template<typename InputIterType, typename KeyStateIterType >
        struct ReduceStencilGeneration
        {
            InputIterType keys_last;
//            KeyStateIterType KeyState;

            VTKM_CONT_EXPORT
            ReduceStencilGeneration(const InputIterType &key_first, const InputIterType &key_last)
                    : keys_last(key_last)
            {  }

            VTKM_EXEC_EXPORT
            void operator()(const InputIterType centerIter, KeyStateIterType kiter) const
            {
                // typedef typename InputIterType::value_type ValueType;
                // typedef typename KeyStateIterType::value_type KeyStateType;

                typedef typename std::iterator_traits<InputIterType>::value_type ValueType;
                typedef typename std::iterator_traits<KeyStateIterType>::value_type KeyStateType;

                const InputIterType leftIter = std::prev(centerIter);
                const InputIterType rightIter = std::next(centerIter);

                //we need to determine which of three states this
                //index is. It can be:
                // 1. Middle of a set of equivalent keys.
                // 2. Start of a set of equivalent keys.
                // 3. End of a set of equivalent keys.
                // 4. Both the start and end of a set of keys

                //we don't have to worry about an array of length 1, as
                //the calling code handles that use case

                if(centerIter == 0)
                {
                    //this means we are at the start of the array
                    //means we are automatically START
                    //just need to check if we are END
                    const ValueType centerValue = *centerIter;
                    const ValueType rightValue = *rightIter;
                    const KeyStateType state = ReduceKeySeriesStates(true, rightValue != centerValue);
                    *kiter = state;
                }
                else if(rightIter == keys_last)
                {
                    //this means we are at the end, so we are at least END
                    //just need to check if we are START
                    const ValueType centerValue = *centerIter;
                    const ValueType leftValue = *leftIter;
                    const KeyStateType state = ReduceKeySeriesStates(leftValue != centerValue, true);
                    *kiter = state;
                }
                else
                {
                    const ValueType centerValue = *centerIter;
                    const bool leftMatches(*leftIter == centerValue);
                    const bool rightMatches(*rightIter == centerValue);

                    //assume it is the middle, and check for the other use-case
                    KeyStateType state = ReduceKeySeriesStates(!leftMatches, !rightMatches);
                    *kiter = state;
                }
            }
        };

        template<typename BinaryFunctor>
        struct ReduceByKeyAdd
        {
            BinaryFunctor BinaryOperator;

            ReduceByKeyAdd(BinaryFunctor binary_functor):
                    BinaryOperator( binary_functor )
            { }

            template<typename T>
            vtkm::Pair<T, ReduceKeySeriesStates> operator()(const vtkm::Pair<T, ReduceKeySeriesStates>& a,
                                                            const vtkm::Pair<T, ReduceKeySeriesStates>& b) const
            {
                typedef vtkm::Pair<T, ReduceKeySeriesStates> ReturnType;
                //need too handle how we are going to add two numbers together
                //based on the keyStates that they have

                // Make it work for parallel inclusive scan.  Will end up with all start bits = 1
                // the following logic should change if you use a different parallel scan algorithm.
                if (!b.second.fStart) {
                    // if b is not START, then it's safe to sum a & b.
                    // Propagate a's start flag to b
                    // so that later when b's START bit is set, it means there must exists a START between a and b
                    return ReturnType(this->BinaryOperator(a.first , b.first),
                                      ReduceKeySeriesStates(a.second.fStart, b.second.fEnd));
                }
                return b;
            }
        };

        struct ReduceByKeyUnaryStencilOp
        {
            bool operator()(ReduceKeySeriesStates keySeriesState) const
            {
                return keySeriesState.fEnd;
            }

        };

/*
        template <typename... T>
        auto sort_zip(T&... containers)
        -> boost::iterator_range<decltype(hpx::iterators::makeTupleIterator(std::begin(containers)...))> {
            return boost::make_iterator_range(hpx::iterators::makeTupleIterator(std::begin(containers)...),
                                              hpx::iterators::makeTupleIterator(std::end(containers)...));
        }
*/
        /// \endcond
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

    typename util::detail::algorithm_result<ExPolicy, void>::type
    reduce_by_key(
        ExPolicy && policy,
        RanIter key_first, RanIter key_last,
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
            "Requires a random access iterator.");

        typedef is_sequential_execution_policy<ExPolicy> is_seq;

        const uint64_t numberOfKeys = std::distance(key_first, key_last);

        if(numberOfKeys <= 1)
        { // we only have a single key/value so that is our output
            *keys_output = *key_first;
            *values_output = *values_first;
            return;
        }

        using namespace hpx::parallel::detail;

        //we need to determine based on the keys what is the keystate for
        //each key. The states are start, middle, end of a series and the special
        //state start and end of a series
        std::vector< ReduceKeySeriesStates > keystate;
        typedef std::vector< ReduceKeySeriesStates >::iterator KeyStateIterType;
        keystate.assign(numberOfKeys, ReduceKeySeriesStates());
        {
            typedef typename hpx::util::zip_iterator<RanIter, KeyStateIterType>::reference zip_ref;
            ReduceStencilGeneration<RanIter, KeyStateIterType> kernel(key_first, key_last);
            hpx::parallel::for_each(
                    policy,
                    hpx::util::make_zip_iterator(key_first, keystate.begin()),
                    key_last,
                    [&kernel](zip_ref ref) {
                        kernel.operator()(hpx::util::get<0>(ref), hpx::util::get<1>(ref));
                    }
            );
        }
/*
            //next step is we need to reduce the values for each key. This is done
            //by running an inclusive scan over the values array using the stencil.
            //
            // this inclusive scan will write out two values, the first being
            // the value summed currently, the second being 0 or 1, with 1 being used
            // when this is a value of a key we need to write ( END or START_AND_END)
            {
                hpx::parallel::inclusive_scan(
                        policy,
                        std::begin(keystate), std::end(keystate),

                );

                typedef vtkm::cont::ArrayHandle<U,VIn> ValueInHandleType;
                typedef vtkm::cont::ArrayHandle<U,VOut> ValueOutHandleType;
                typedef vtkm::cont::ArrayHandle< ReduceKeySeriesStates> StencilHandleType;
                typedef vtkm::cont::ArrayHandleZip<ValueInHandleType,
                    StencilHandleType> ZipInHandleType;
                typedef vtkm::cont::ArrayHandleZip<ValueOutHandleType,
                    StencilHandleType> ZipOutHandleType;

                StencilHandleType stencil;
                ValueOutHandleType reducedValues;

                ZipInHandleType scanInput( values, keystate);
                ZipOutHandleType scanOutput( reducedValues, stencil);

                DerivedAlgorithm::ScanInclusive(scanInput,
                                                scanOutput,
                                                ReduceByKeyAdd<BinaryFunctor>(binary_functor) );

                //at this point we are done with keystate, so free the memory
                keystate.ReleaseResources();

                // all we need know is an efficient way of doing the write back to the
                // reduced global memory. this is done by using StreamCompact with the
                // stencil and values we just created with the inclusive scan
                DerivedAlgorithm::StreamCompact( reducedValues,
                                                 stencil,
                                                 values_output,
                                                 ReduceByKeyUnaryStencilOp());

            } //release all temporary memory


            //find all the unique keys
            DerivedAlgorithm::Copy(keys,keys_output);
            DerivedAlgorithm::Unique(keys_output);
            */
        }
    }
}}

#endif
