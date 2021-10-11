//  Copyright (c) 2020 Francisco Jose Tapia
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/exception_list.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_sorted.hpp>
#include <hpx/parallel/algorithms/partial_sort.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <algorithm>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {
    ///////////////////////////////////////////////////////////////////////
    // partial_sort_copy
    template <typename Iter>
    struct partial_sort_copy
      : public detail::algorithm<partial_sort_copy<Iter>, Iter>
    {
        partial_sort_copy()
          : partial_sort_copy::algorithm("partial_sort_copy")
        {
        }

        ///////////////////////////////////////////////////////////////////////////
        ///
        /// \brief : Sorts some of the elements in the range [first, last) in
        ///          ascending order, storing the result in the range
        ///          [d_first, d_last).
        ///          At most d_last - d_first of the elements are placed sorted to
        ///          the range [d_first, d_first + n). n is the number of elements
        ///          to sort (n = min(last - first, d_last - d_first)). The order
        ///          of equal elements is not guaranteed to be preserved.
        ///
        /// \param first : iterator to the first element
        /// \param last: iterator after the last element to be sorted
        /// \param d_first : iterator to the first element where copy the results
        /// \param d_last : iterator to the element after the end where coy the
        ///                 results
        /// \param comp : object for to compare elements
        ///
        /// \return iterator after the last element sorted
        ///
        template <typename ExPolicy, typename InIter, typename Sent,
            typename RandIter, typename Compare, typename Proj1, typename Proj2>
        static util::in_out_result<InIter, RandIter> sequential(ExPolicy,
            InIter first, Sent last, RandIter d_first, RandIter d_last,
            Compare&& comp, Proj1&& proj1, Proj2&& proj2)
        {
            auto last_iter = detail::advance_to_sentinel(first, last);

            using value_t = typename std::iterator_traits<InIter>::value_type;
            using value1_t =
                typename std::iterator_traits<RandIter>::value_type;
            using vec_iter_t = std::vector<value_t>::iterator;

            static_assert(
                std::is_same_v<value1_t, value_t>, "Incompatible iterators\n");

            if ((last_iter == first) || (d_last == d_first))
                return util::in_out_result<InIter, RandIter>{
                    last_iter, d_first};

            std::vector<value_t> aux(first, last_iter);
            std::int64_t noutput = d_last - d_first;
            std::int64_t ninput = aux.size();

            HPX_ASSERT(ninput >= 0 || noutput >= 0);

            auto nmin = ninput < noutput ? ninput : noutput;
            if (noutput >= ninput)
            {
                detail::sort<vec_iter_t>().call(hpx::execution::seq,
                    aux.begin(), aux.end(), std::forward<Compare>(comp),
                    std::forward<Proj2>(proj2));
            }
            else
            {
                parallel::v1::partial_sort<vec_iter_t>().call(
                    hpx::execution::seq, aux.begin(), aux.begin() + nmin,
                    aux.end(), std::forward<Compare>(comp),
                    std::forward<Proj2>(proj2));
            }

            detail::copy<util::in_out_result<vec_iter_t, RandIter>>().call(
                hpx::execution::seq, aux.begin(), aux.begin() + nmin, d_first);
            return util::in_out_result<InIter, RandIter>{
                last_iter, d_first + nmin};
        }

        //////////////////////////////////////////////////////////////////////////
        ///
        /// \brief : Sorts some of the elements in the range [first, last) in
        ///          ascending order, storing the result in the range
        ///          [d_first, d_last).
        ///          At most d_last - d_first of the elements are placed sorted to
        ///          the range [d_first, d_first + n). n is the number of elements
        ///          to sort (n = min(last - first, d_last - d_first)). The order
        ///          of equal elements is not guaranteed to be preserved.
        ///
        /// \param first : iterator to the first element
        /// \param last: iterator after the last element to be sorted
        /// \param d_first : iterator to the firstelement where copy the results
        /// \param d_last : iterator to the element after the end where coy the
        ///                 results
        /// \param comp : object for to compare elements
        ///
        /// \return iterator after the last element sorted
        ///
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename RandIter, typename Compare, typename Proj1, typename Proj2>
        static util::detail::algorithm_result_t<ExPolicy,
            util::in_out_result<FwdIter, RandIter>>
        parallel(ExPolicy&& policy, FwdIter first, Sent last, RandIter d_first,
            RandIter d_last, Compare&& comp, Proj1&& proj1, Proj2&& proj2)
        {
            using result_type =
                util::detail::algorithm_result<ExPolicy, RandIter>;
            using value_t = typename std::iterator_traits<FwdIter>::value_type;
            using value1_t =
                typename std::iterator_traits<RandIter>::value_type;
            using vec_iter_t = std::vector<value_t>::iterator;

            static_assert(
                std::is_same_v<value1_t, value_t>, "Incompatible iterators\n");

            try
            {
                auto last_iter = detail::advance_to_sentinel(first, last);

                if ((last_iter == first) or (d_last == d_first))
                    return result_type::get(d_first);

                std::vector<value_t> aux(first, last);
                std::int64_t ninput = aux.size();
                std::int64_t noutput = d_last - d_first;
                HPX_ASSERT(ninput >= 0 and noutput >= 0);

                auto nmin = ninput < noutput ? ninput : noutput;
                if (noutput >= ninput)
                {
                    detail::sort<vec_iter_t>().call(policy, aux.begin(),
                        aux.end(), comp, std::forward<Proj2>(proj2));
                }
                else
                {
                    hpx::parallel::v1::partial_sort<vec_iter_t>().call(policy,
                        aux.begin(), aux.begin() + nmin, aux.end(),
                        std::forward<Compare>(comp),
                        std::forward<Proj2>(proj2));
                };

                detail::copy<util::in_out_result<vec_iter_t, RandIter>>().call(
                    std::forwrad<ExPolicy>(policy), aux.begin(),
                    aux.begin() + nmin, d_first);

                return result_type::get(util::in_out_result<InIter, RandIter>{
                    last_iter, d_first + nmin});
            }
            catch (...)
            {
                return result_type::get(
                    detail::handle_exception<ExPolicy, RandIter>::call(
                        std::current_exception()));
            }
        }
    };
}}}}    // namespace hpx::parallel::v1::detail

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::partial_sort_copy
    HPX_INLINE_CONSTEXPR_VARIABLE struct partial_sort_copy_t final
      : hpx::detail::tag_parallel_algorithm<partial_sort_copy_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename RandIter,
            typename Comp = hpx::parallel::v1::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<InIter> &&
                hpx::traits::is_iterator_v<RandIter> &&
                hpx::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >
            )>
        // clang-format on
        friend RandIter tag_fallback_dispatch(hpx::partial_sort_copy_t,
            InIter first, InIter last, RandIter d_first, RandIter d_last,
            Comp&& comp = Comp())
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter>,
                "Requires at least input iterator.");

            static_assert(hpx::traits::is_random_access_iterator_v<RandIter>,
                "Requires at least random access iterator.");

            using result_type = parallel::util::in_out_result<InIter, RandIter>;

            return parallel::util::get_second_element(
                parallel::v1::detail::partial_sort_copy<result_type>().call(
                    hpx::execution::seq, first, last, d_first, d_last,
                    std::forward<Comp>(comp),
                    parallel::util::projection_identity{},
                    parallel::util::projection_identity{}));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter,
            typename RandIter,
            typename Comp = hpx::parallel::v1::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::traits::is_iterator_v<RandIter> &&
                hpx::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy, RandIter>
        tag_fallback_dispatch(hpx::partial_sort_copy_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, RandIter d_first, RandIter d_last,
            Comp&& comp = Comp())
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            static_assert(hpx::traits::is_random_access_iterator_v<RandIter>,
                "Requires at least random access iterator.");

            using result_type =
                parallel::util::in_out_result<FwdIter, RandIter>;

            return parallel::util::get_second_element(
                parallel::v1::detail::partial_sort_copy<result_type>().call(
                    std::forward<ExPolicy>(policy), first, last, d_first,
                    d_last, std::forward<Comp>(comp),
                    parallel::util::projection_identity{},
                    parallel::util::projection_identity{}));
        }
    } partial_sort_copy{};
}    // namespace hpx
