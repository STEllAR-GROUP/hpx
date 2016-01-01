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
#include <hpx/util/transform_iterator.hpp>
#include <hpx/util/tuple.hpp>
//
#include <boost/thread/locks.hpp>
//
#define VTKM_CONT_EXPORT
#define VTKM_EXEC_EXPORT

#define boolvals(a,b) \
  (a?1:0) + (b?2:0)

typedef hpx::lcos::local::shared_mutex mutex_type;

namespace vtkm {
    typedef uint64_t Id;
    template<typename T1, typename T2> using Pair = hpx::util::tuple<T1, T2>;
}

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
            bool fNull;
            ReduceKeySeriesStates(bool start=false, bool end=false, bool null=true) :
                    fStart(start), fEnd(end), fNull(null) {}
/*
            ReduceKeySeriesStates(const ReduceKeySeriesStates &other)
                    : fStart(other.fStart), fEnd(other.fEnd) {}
            ReduceKeySeriesStates(const ReduceKeySeriesStates *other)
                    : fStart(other->fStart), fEnd(other->fEnd) {}
            ReduceKeySeriesStates operator=(const ReduceKeySeriesStates &other) {
                this->fStart = other.fStart;
                this->fEnd = other.fEnd;
                return *this;
            }
*/
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
                    kiter = ReduceKeySeriesStates(!leftMatches, !rightMatches, false);
                }
            }
        };
/*
        struct ReduceByKeyUnaryStencilOp
        {
            bool operator()(ReduceKeySeriesStates keySeriesState) const
            {
                return keySeriesState.fEnd;
            }

        };
*/
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

            mutex_type mtx_;

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
                keystate[0] = ReduceKeySeriesStates(true, left!=right, false);
                keystate[1] = ReduceKeySeriesStates(left!=right, true, false);
            }
            else {
                // do the first element and last one by hand to simplify the iterator
                // traversal as there is no prev/next for first/last
                element_type elem0 = *key_first;
                element_type elem1 = *std::next(key_first);
                keystate[0] = ReduceKeySeriesStates(true, elem0!=elem1, false);
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
                keystate.back() = ReduceKeySeriesStates(elemN!=elemn, true, false);
            }
        }
        {
            typedef hpx::util::zip_iterator<RanIter2, std::vector< ReduceKeySeriesStates >::iterator> zip_iterator;
            typedef std::vector< ReduceKeySeriesStates >::iterator rki;
            typedef typename zip_iterator::value_type zip_type;
            typedef typename std::iterator_traits<RanIter2>::value_type value_type;

            zip_iterator states_begin = hpx::util::make_zip_iterator(values_first, std::begin(keystate));
            zip_iterator states_end = hpx::util::make_zip_iterator(
                    values_first + std::distance(std::begin(keystate), std::end(keystate)),
                    std::end(keystate));
            zip_iterator states_out_begin = hpx::util::make_zip_iterator(values_output, std::begin(keystate));

            zip_type initial = hpx::util::tuple<float, ReduceKeySeriesStates>(0.0, ReduceKeySeriesStates(true, false, true));
            // caution : if the current value is a start value, then it should be used
            // otherwise we add the incoming B to our A.
            // BUT, the partitions are updated with values carried from if the when summing we can add values if the current value is not a start
            hpx::parallel::prefix_scan(
                    policy,
                    states_begin, states_end,
                    states_out_begin,
                    initial,
                    // B is the current entry, A is the one passed in from 'previous'
                    [](zip_type a, zip_type b) {
                        value_type            a_val   = hpx::util::get<0>(a);
                        ReduceKeySeriesStates a_state = hpx::util::get<1>(a);
                        value_type            b_val   = hpx::util::get<0>(b);
                        ReduceKeySeriesStates b_state = hpx::util::get<1>(b);
                        static value_type x = 1000;
                        static value_type y = 0;
                        // if carrying a start flag from both previous ops
                        // then do not add (used in higher level upsweep)
                        boost::lock_guard<mutex_type> m(mtx_);
                        std::cout << "{ " << a_val << "+" << b_val << " },\t" << a_state << b_state ;
                        if (b_state.fNull) {
                            throw std::string("help");
                            std::cout << " * " << a_val << std::endl;
                            return hpx::util::make_tuple(
                                    //boolvals(a_state.fStart, b_state.fStart) ,
                                    a_val,
                                    ReduceKeySeriesStates(a_state.fStart, b_state.fEnd, false));
                        }
                        else if (/*a_state.fStart && */b_state.fStart) {
                            std::cout << " = " << b_val << std::endl;
                            return hpx::util::make_tuple(
                                    //boolvals(a_state.fStart, b_state.fStart) ,
                                     b_val,
                                    ReduceKeySeriesStates(a_state.fStart || b_state.fStart, b_state.fEnd, false));
                        }
                        // if b is a start then reset sequence, just use b value
                        else if (b_state.fStart) {
                            std::cout << " = " << b_val << std::endl;
                            return hpx::util::make_tuple(
                                    //boolvals(a_state.fStart, b_state.fStart) ,
                                     b_val,
                                    ReduceKeySeriesStates(a_state.fStart || b_state.fStart, b_state.fEnd, false));
                        }
                        // normal add of previous + this
                        else if (a_state.fStart) {
                            std::cout << " = " << a_val + b_val << std::endl;
                            return hpx::util::make_tuple(
                                    //boolvals(a_state.fStart, b_state.fStart) ,
                                     a_val + b_val,
                                    ReduceKeySeriesStates(a_state.fStart || b_state.fStart, b_state.fEnd, false));
                        }
                        // should not ever be called
                        else {
                            std::cout << " = " << a_val + b_val << std::endl;
                            return hpx::util::make_tuple(
                                    //boolvals(a_state.fStart, b_state.fStart) ,
                                     a_val + b_val,
                                    ReduceKeySeriesStates(a_state.fStart || b_state.fStart, b_state.fEnd, false));
                        }
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
