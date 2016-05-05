//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_PREFIX_UNIQUE_MAY_2016)
#define HPX_PARALLEL_ALGORITHM_PREFIX_UNIQUE_MAY_2016
//
#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/tagged_pair.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/util/transform_iterator.hpp>
//
#include <hpx/parallel/algorithms/reduce_by_key.hpp>
#include <hpx/parallel/algorithms/prefix_scan.hpp>
#include <hpx/parallel/algorithms/prefix_copy_if.hpp>
#include <hpx/util/tuple.hpp>
//

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // reduce_by_key
    namespace detail
    {
        /// \cond NOINTERNAL
        // -------------------------------------------------------------------
        // simple iterator helper object for access to prev/this items
        // -------------------------------------------------------------------
        struct unique_stencil_transformer
        {
            // declare result type as a template
            template<typename T>
            struct result;

            // specialize result for iterator type
            template<typename This, typename Iterator>
            struct result<This(Iterator)>
            {
                typedef typename std::iterator_traits<Iterator>::reference element_type;
                typedef hpx::util::tuple<element_type, element_type> type;
            };

            // call operator for stencil transform
            // it will dereference tuple(it-1, it)
            template<typename Iterator>
            typename result<unique_stencil_transformer(Iterator)>::type
            operator()(Iterator const &it) const
            {
                typedef typename result<
                    unique_stencil_transformer(Iterator)
                >::type type;
                return type(*std::prev(it), *it);
            }
        };

        // -------------------------------------------------------------------
        // transform iterator using unique_stencil_transformer helper
        // -------------------------------------------------------------------
        template<
            typename Iterator,
            typename Transformer = unique_stencil_transformer
        >
        class unique_stencil_iterator :
            public hpx::util::transform_iterator<Iterator, Transformer>
        {
        private:
            typedef hpx::util::transform_iterator<Iterator, Transformer> base_type;

        public:
            unique_stencil_iterator() { }

            explicit unique_stencil_iterator(Iterator const &it) : base_type(it,
                Transformer()) { }

            unique_stencil_iterator(Iterator const &it, Transformer const &t)
                : base_type(it, t) { }
        };

        template<typename Iterator, typename Transformer>
        inline unique_stencil_iterator<Iterator, Transformer>
        make_unique_stencil_iterator(Iterator const &it, Transformer const &t)
        {
            return unique_stencil_iterator<Iterator, Transformer>(it, t);
        }

        // -------------------------------------------------------------------
        // functor that actually computes the state using the stencil iterator
        // -------------------------------------------------------------------
        template<
            typename Transformer, typename StencilIterType,
            typename StateIterType, typename Compare
        >
        struct unique_stencil_generate
        {
            typedef typename Transformer::template result<
                Transformer(StencilIterType)
            >::element_type element_type;
            typedef typename Transformer::template result<
                Transformer(StencilIterType)
            >::type tuple_type;
            typedef typename std::iterator_traits<
                StateIterType
            >::reference KeyStateType;

            unique_stencil_generate() { }

            void operator()(const tuple_type &value, unsigned char &kiter,
                const Compare &comp) const
            {
                // resolves to a tuple of values for *(it-1), *it
                element_type left = hpx::util::get<0>(value);
                element_type mid = hpx::util::get<1>(value);
                kiter = !comp(left, mid);
            }
        };


        // -------------------------------------------------------------------
        // The main algorithm is implemented here, it replaces any async
        // execution policy with a non async one so that no waits are
        // necessary on the internal algorithms. Async execution is handled
        // by the wrapper layer that calls this.
        // -------------------------------------------------------------------
        template<
            typename ExPolicy,
            typename RanIter,
            typename Compare
            >
        static RanIter
        unique_impl(ExPolicy &&policy, RanIter first, RanIter last,
            Compare &&comp)
        {
            using namespace hpx::parallel::v1::detail;
            using namespace hpx::util;

            typedef typename detail::remove_asynchronous<
                typename std::decay<ExPolicy>::type>::type sync_policy_type;

            sync_policy_type sync_policy = sync_policy_type().on(policy.executor())
                .with(policy.parameters());

            // we need to determine the state for a given element
            std::vector<unsigned char> key_state;
            typedef std::vector<unsigned char>::iterator state_iter_type;
            typedef detail::unique_stencil_iterator<RanIter, unique_stencil_transformer>
                uniquebykey_iter;

            typedef typename std::iterator_traits<RanIter>::value_type value_type;
            typedef typename std::iterator_traits<RanIter>::reference element_type;
            typedef typename zip_iterator<uniquebykey_iter, state_iter_type>
              ::reference zip_ref;
            //
            const uint64_t N = std::distance(first, last);
            //
            std::vector<value_type> temp_buffer_to_be_removed(N);
            key_state.assign(N, 0);
            {
                unique_stencil_transformer u_s_t;
                uniquebykey_iter unique_begin =
                    make_unique_stencil_iterator(first, u_s_t);
                uniquebykey_iter unique_end =
                    make_unique_stencil_iterator(last, u_s_t);

                key_state[0] = 1;
                if (N == 2) {
                    element_type left = *first;
                    element_type right = *std::next(first);
                    key_state[1] = !comp(left, right);
                } else {
                    // skip first element to make iterator traversal easier
                    unique_stencil_generate <unique_stencil_transformer, RanIter,
                        state_iter_type, Compare> kernel;

                    hpx::parallel::for_each(sync_policy,
                        make_zip_iterator(std::next(unique_begin,1), &key_state[1]),
                        make_zip_iterator(unique_end, &key_state[N]),
                        [&kernel, &comp](zip_ref ref)
                        {
                            kernel.operator()(get<0>(ref), get<1>(ref), comp);
                        });

                }
            }

            auto new_end = hpx::parallel::prefix_copy_if_stencil(
                sync_policy,
                first, last, key_state.begin(),
                temp_buffer_to_be_removed.begin());

            return hpx::util::get<1>(hpx::parallel::copy(sync_policy,
                temp_buffer_to_be_removed.begin(), new_end, first));
        }
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


    template <typename ExPolicy, typename InIter, typename Compare,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<InIter>::value)
        >
    typename util::detail::algorithm_result<
        ExPolicy, InIter
    >::type
    prefix_unique(ExPolicy&& policy, InIter first, InIter last, Compare && comp,
        Proj && proj = Proj())
    {
        typedef std::integral_constant<bool,
                is_sequential_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<InIter>::value ||
               !hpx::traits::is_forward_iterator<InIter>::value
            > is_seq;

        typedef typename std::iterator_traits<InIter>::value_type value_type;

        typedef typename detail::remove_asynchronous<
                    typename std::decay< ExPolicy >::type >::type sync_policy_type;

        sync_policy_type sync_policy =
            sync_policy_type().on(policy.executor()).with(policy.parameters());

        return detail::unique_impl(sync_policy, first, last, comp);

/*
        typedef detail::unique_stencil_iterator<InIter, detail::unique_stencil_transformer>
            uniquebykey_iter;
        typedef typename std::iterator_traits<InIter>::value_type value_type;
        typedef typename std::iterator_traits<InIter>::reference element_type;
        typedef typename hpx::util::zip_iterator<uniquebykey_iter, bool*>
          ::reference zip_ref;

        typedef hpx::util::zip_iterator<InIter, bool*> zip_iterator;
        std::size_t N = std::distance(first,last);
        boost::shared_array<bool> flags(new bool[N]);
        std::size_t init = 0;
        //
        {
            detail::unique_stencil_transformer u_s_t;
            uniquebykey_iter unique_begin =
                make_unique_stencil_iterator(first, u_s_t);
            uniquebykey_iter unique_end =
                make_unique_stencil_iterator(last, u_s_t);

            // do first element to make iterator traversal simple
            flags[0] = true;
            if (N == 2) {
                element_type left = *first;
                element_type right = *std::next(first);
                flags[1] = !comp(left, right);
            } else {
                detail::unique_stencil_generate
                    <detail::unique_stencil_transformer, InIter, bool*, Compare> kernel;

                // first element skipped already
                zip_iterator s_begin = hpx::util::make_zip_iterator(unique_begin + 1, flags.get() + 1);
                zip_iterator s_end   = hpx::util::make_zip_iterator(unique_end, flags.get()+N);
                std::size_t total = 0;

                auto result = detail::parallel_scan_struct_lambda< std::size_t, detail::exclusive_scan_tag>().call(
                    sync_policy,
                    is_seq(),
                    s_begin,
                    s_end,
                    total,
                    init,
                    // stage 1 : initial pass of each section of the input
                    [&kernel, &comp](zip_ref ref, std::size_t count, value_type init) {
                        std::size_t offset = 0;
                        for (; count-- != 0; ++first) {
                            kernel.operator()(get<0>(ref), get<1>(ref), comp);
                            // assign bool to final stencil, if true increment count
                            if (get<1>(ref)) offset++;
                        }
                        return offset;
                    },
                    // stage 2 operator to use to combine intermediate results
                    std::plus<std::size_t>(),
                    // stage 3 lambda to apply results to each section
                    [out_iter](zip_iterator first, std::size_t count, OutIter dest, std::size_t offset) mutable {
                        std::advance(out_iter, offset);
                        for (; count-- != 0; ++first) {
                            if (hpx::util::get<1>(*first)) {
                                *out_iter++ = hpx::util::get<0>(*first);
        //                        std::cout << "writing " << hpx::util::get<0>(*first) << "\n";
                            }
                        }
                        return out_iter;
                    },
                    // stage 4 : generate a return value
                    [last](OutIter dest) mutable ->  std::pair<InIter, OutIter> {
                        //std::advance(out_iter, offset);
                        return std::make_pair(last, dest);
                    }
                );

            }

        }
        //
*/
    }

}}}

#endif
