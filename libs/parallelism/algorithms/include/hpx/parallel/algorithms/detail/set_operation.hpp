//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/invoke.hpp>

#include <hpx/execution/executors/execution_information.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/algorithms/detail/upper_lower_bound.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/foreach_partitioner.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/type_support/unused.hpp>

#if !defined(HPX_HAVE_CXX17_SHARED_PTR_ARRAY)
#include <boost/shared_array.hpp>
#else
#include <memory>
#endif

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    template <typename FwdIter>
    struct set_operations_buffer
    {
        template <typename T>
        class rewritable_ref
        {
        public:
            constexpr rewritable_ref()
              : item_(0)
            {
            }
            constexpr rewritable_ref(T const& item)
              : item_(item)
            {
            }

            rewritable_ref& operator=(T const& item)
            {
                item_ = &item;
                return *this;
            }

            // different versions of clang-format produce different results
            // clang-format off
            operator T const&() const
            {
                HPX_ASSERT(item_ != 0);
                return *item_;
            }
            // clang-format on

        private:
            T const* item_;
        };

        typedef typename std::iterator_traits<FwdIter>::value_type value_type;
        typedef typename std::conditional<std::is_scalar<value_type>::value,
            value_type, rewritable_ref<value_type>>::type type;
    };

    struct set_chunk_data
    {
        std::size_t start = std::size_t(-1);
        std::size_t len = std::size_t(-1);
        std::size_t start_index = std::size_t(-1);
        std::size_t first1 = std::size_t(-1);
        std::size_t first2 = std::size_t(-1);
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter1, typename Sent1, typename Iter2,
        typename Sent2, typename Iter3, typename F, typename Proj1,
        typename Proj2, typename Combiner, typename SetOp>
    typename util::detail::algorithm_result<ExPolicy,
        util::in_in_out_result<Iter1, Iter2, Iter3>>::type
    set_operation(ExPolicy&& policy, Iter1 first1, Sent1 last1, Iter2 first2,
        Sent2 last2, Iter3 dest, F&& f, Proj1&& proj1, Proj2&& proj2,
        Combiner&& combiner, SetOp&& setop)
    {
        using result_type = util::in_in_out_result<Iter1, Iter2, Iter3>;

        using difference_type1 =
            typename std::iterator_traits<Iter1>::difference_type;
        using difference_type2 =
            typename std::iterator_traits<Iter2>::difference_type;

        // allocate intermediate buffers
        difference_type1 len1 = detail::distance(first1, last1);
        difference_type2 len2 = detail::distance(first2, last2);

        using buffer_type = typename set_operations_buffer<Iter3>::type;

        std::size_t cores = execution::processing_units_count(
            policy.parameters(), policy.executor());

        std::size_t step = (len1 + cores - 1) / cores;

#if defined(HPX_HAVE_CXX17_SHARED_PTR_ARRAY)
        std::shared_ptr<buffer_type[]> buffer(
            new buffer_type[combiner(len1, len2)]);
        std::shared_ptr<set_chunk_data[]> chunks(new set_chunk_data[cores]);
#else
        boost::shared_array<buffer_type> buffer(
            new buffer_type[combiner(len1, len2)]);
        boost::shared_array<set_chunk_data> chunks(new set_chunk_data[cores]);
#endif

        // first step, is applied to all partitions
        auto f1 = [=](set_chunk_data* curr_chunk,
                      std::size_t part_size) -> void {
            HPX_ASSERT(part_size == 1);
            HPX_UNUSED(part_size);

            // find start in sequence 1
            std::size_t start1 = (curr_chunk - chunks.get()) * step;
            std::size_t end1 = (std::min)(start1 + step, std::size_t(len1));

            if (start1 >= end1)
            {
                return;
            }

            bool first_partition = (start1 == 0);
            bool last_partition = (end1 == std::size_t(len1));

            auto start_value = hpx::util::invoke(proj1, first1[start1]);
            auto end_value = hpx::util::invoke(proj1, first1[end1]);

            // all but the last chunk require special handling
            if (!last_partition)
            {
                // this chunk will be handled by the next one if all
                // elements of this partition are equal
                if (!hpx::util::invoke(f, start_value, end_value))
                {
                    return;
                }

                // move backwards to find earliest element which is equal to
                // the last element of the current chunk
                if (end1 != 0)
                {
                    auto end_value1 =
                        hpx::util::invoke(proj1, first1[end1 - 1]);

                    while (!hpx::util::invoke(f, end_value1, end_value) &&
                        --end1 != 0)
                    {
                        end_value = std::move(end_value1);
                        end_value1 = hpx::util::invoke(proj1, first1[end1 - 1]);
                    }
                }
            }

            // move backwards to find earliest element which is equal to
            // the first element of the current chunk
            if (start1 != 0)
            {
                auto start_value1 =
                    hpx::util::invoke(proj1, first1[start1 - 1]);

                while (!hpx::util::invoke(f, start_value1, start_value) &&
                    --start1 != 0)
                {
                    start_value = std::move(start_value1);
                    start_value1 = hpx::util::invoke(proj1, first1[start1 - 1]);
                }
            }

            // find start and end in sequence 2
            std::size_t start2 = 0;
            if (!first_partition)
            {
                start2 = detail::lower_bound(
                             first2, first2 + len2, start_value, f, proj2) -
                    first2;
            }

            std::size_t end2 = len2;
            if (!last_partition)
            {
                end2 = detail::lower_bound(first2 + start2, first2 + len2,
                           end_value, f, proj2) -
                    first2;
            }

            // perform requested set-operation into the proper place of the
            // intermediate buffer
            curr_chunk->start = combiner(start1, start2);
            auto buffer_dest = buffer.get() + curr_chunk->start;
            auto op_result = setop(first1 + start1, first1 + end1,
                first2 + start2, first2 + end2, buffer_dest, f);
            curr_chunk->first1 = op_result.in1 - first1;
            curr_chunk->first2 = op_result.in2 - first2;
            curr_chunk->len = op_result.out - buffer_dest;
        };

        // second step, is executed after all partitions are done running

        // different versions of clang-format produce different formatting
        // clang-format off
        auto f2 = [buffer, chunks, cores, first1, first2, dest](
                      std::vector<future<void>>&& data) -> result_type {
            // clang-format on

            // make sure iterators embedded in function object that is
            // attached to futures are invalidated
            data.clear();

            // accumulate real length and rightmost positions in input sequences
            std::size_t first1_pos = 0;
            std::size_t first2_pos = 0;

            set_chunk_data* chunk = chunks.get();
            chunk->start_index = 0;
            for (size_t i = 1; i != cores; ++i)
            {
                set_chunk_data* curr_chunk = chunk++;
                chunk->start_index = curr_chunk->start_index + curr_chunk->len;
                if (curr_chunk->first1 != std::size_t(-1))
                {
                    first1_pos = (std::max)(first1_pos, curr_chunk->first1);
                }
                if (curr_chunk->first2 != std::size_t(-1))
                {
                    first2_pos = (std::max)(first2_pos, curr_chunk->first2);
                }
            }

            // finally, copy data to destination
            parallel::util::
                foreach_partitioner<hpx::execution::parallel_policy>::call(
                    hpx::execution::par, chunks.get(), cores,
                    [buffer, dest](
                        set_chunk_data* chunk, std::size_t, std::size_t) {
                        if (chunk->start == std::size_t(-1) ||
                            chunk->start_index == std::size_t(-1) ||
                            chunk->len == std::size_t(-1))
                        {
                            return;
                        }
                        std::copy(buffer.get() + chunk->start,
                            buffer.get() + chunk->start + chunk->len,
                            dest + chunk->start_index);
                    },
                    [](set_chunk_data* last) -> set_chunk_data* {
                        return last;
                    });

            return {std::next(first1, first1_pos),
                std::next(first2, first2_pos),
                std::next(dest, chunk->start_index + chunk->len)};
        };

        // fill the buffer piecewise
        return parallel::util::partitioner<ExPolicy, result_type, void>::call(
            policy, chunks.get(), cores, std::move(f1), std::move(f2));
    }

    /// \endcond
}}}}    // namespace hpx::parallel::v1::detail
