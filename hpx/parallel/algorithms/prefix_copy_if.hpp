//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_ALGORITHM_PREFIX_COPY_IF_JAN_2016)
#define HPX_PARALLEL_ALGORITHM_PREFIX_COPY_IF_JAN_2016
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
#include <hpx/util/tuple.hpp>
//

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // reduce_by_key
    namespace detail
    {
        /// \cond NOINTERNAL

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


    template <typename ExPolicy, typename InIter, typename OutIter, typename F,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<InIter>::value &&
            hpx::traits::is_iterator<OutIter>::value)
        >
    typename util::detail::algorithm_result<
        ExPolicy, OutIter
    >::type
    prefix_copy_if(ExPolicy&& policy, InIter first, InIter last, OutIter dest, F && op,
        Proj && proj = Proj())
    {
        typedef std::integral_constant<bool,
                is_sequential_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<InIter>::value ||
               !hpx::traits::is_forward_iterator<OutIter>::value ||
               !hpx::traits::is_output_iterator<OutIter>::value
            > is_seq;

        typedef typename std::iterator_traits<InIter>::value_type value_type;

        typedef typename detail::remove_asynchronous<
                    typename std::decay< ExPolicy >::type >::type sync_policy_type;

        sync_policy_type sync_policy = sync_policy_type().on(policy.executor()).with(policy.parameters());

        typedef hpx::util::zip_iterator<InIter, bool*> zip_iterator;
        std::size_t N = std::distance(first,last);
        boost::shared_array<bool> flags(new bool[N]);
        std::size_t init = 0;
        //
        zip_iterator s_begin = hpx::util::make_zip_iterator(first, flags.get());
        zip_iterator s_end   = hpx::util::make_zip_iterator(last,  flags.get()+N);
        OutIter out_iter = dest;
        //
        auto result = detail::parallel_scan_struct_lambda< OutIter, detail::exclusive_scan_tag>().call(
            std::forward < ExPolicy > (policy),
            is_seq(),
            s_begin,
            s_end,
            dest,
            init,
            // stage 1 : initial pass of each section of the input
            [&op](zip_iterator first, std::size_t count, value_type init) {
                std::size_t offset = 0;
                for (/* */; count-- != 0; ++first) {
                    bool temp = op(hpx::util::get<0>(*first));
                    // assign bool to final stencil, if true increment count
                    if ((hpx::util::get<1>(*first) = temp)) offset++;
                }
                return offset;
            },
            // stage 2 operator to use to combine intermediate results
            std::plus<std::size_t>(),
            // stage 3 lambda to apply results to each section
            [out_iter](zip_iterator first, std::size_t count, OutIter dest, std::size_t offset) mutable {
                std::advance(out_iter, offset);
                for (/* */; count-- != 0; ++first) {
                    if (hpx::util::get<1>(*first)) {
                        *out_iter++ = hpx::util::get<0>(*first);
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

        return result;
    }

    template <typename ExPolicy, typename InIter, typename StencilIter,
        typename OutIter,
        typename StencilUnary = util::projection_identity,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<InIter>::value &&
            hpx::traits::is_iterator<OutIter>::value)
        >
    typename util::detail::algorithm_result<
        ExPolicy, OutIter
    >::type
    prefix_copy_if_stencil(ExPolicy&& policy, InIter first, InIter last,
        StencilIter stencil, OutIter dest,
        StencilUnary && unary = StencilUnary(),
        Proj && proj = Proj())
    {
        typedef std::integral_constant<bool,
                is_sequential_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<InIter>::value ||
               !hpx::traits::is_forward_iterator<OutIter>::value ||
               !hpx::traits::is_forward_iterator<StencilIter>::value ||
               !hpx::traits::is_output_iterator<OutIter>::value
            > is_seq;

        typedef typename std::iterator_traits<InIter>::value_type value_type;

        typedef typename detail::remove_asynchronous<
                    typename std::decay< ExPolicy >::type >::type sync_policy_type;

        sync_policy_type sync_policy =
            sync_policy_type().on(policy.executor()).with(policy.parameters());

        typedef hpx::util::zip_iterator<InIter, StencilIter> zip_iterator;
        std::size_t N = std::distance(first,last);
        std::size_t init = 0;
        //
        zip_iterator s_begin = hpx::util::make_zip_iterator(first, stencil);
        zip_iterator s_end   = hpx::util::make_zip_iterator(last,  stencil+N);
        OutIter out_iter = dest;
        //
        auto result = detail::parallel_scan_struct_lambda< OutIter,
          detail::exclusive_scan_tag>().call(
            sync_policy,
            std::false_type(), // is_seq(),
            s_begin,
            s_end,
            dest,
            init,

            // f1 : initial pass of each section of the input
            [&unary](zip_iterator first, std::size_t count, std::size_t init) {
                std::size_t offset = 0;
                for (/* */; count-- != 0; ++first) {
                    // if stencil true increment count
                    if (unary(hpx::util::get<1>(*first))) offset++;
                }
                return offset;
            },

            // operator to use to combine intermediate results
            std::plus<std::size_t>(),

            // f2 lambda to apply results to each section
            [&unary,out_iter](zip_iterator first, std::size_t count, OutIter dest, std::size_t offset) mutable {
                //std::cout << "Offset at start is " << offset <<"\n";
                std::advance(out_iter, offset);
                for (/* */; count-- != 0; ++first) {
                    if (unary(hpx::util::get<1>(*first)))  {
                        *out_iter++ = hpx::util::get<0>(*first);
                    }
                }
                return out_iter;
            },

            // f3 : generate a return value
            [](OutIter dest) {
                //std::advance(out_iter, offset);
                return dest;
            }
        );

        return result;
    }

}}}

#endif
