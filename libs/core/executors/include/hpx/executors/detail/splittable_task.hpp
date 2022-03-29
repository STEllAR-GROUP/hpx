//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2021 Shahrzad Shirzad
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/detail/post_policy_dispatch.hpp>
#include <hpx/modules/async_base.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/modules/topology.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace hpx::threads {

    HPX_CORE_EXPORT mask_type get_idle_core_mask();
}

namespace hpx::parallel::execution::detail {

    ///////////////////////////////////////////////////////////////////////////
    enum class splittable_mode
    {
        guided = 0,
        adaptive = 1,
    };

    inline constexpr char const* get_splittable_mode_name(splittable_mode mode)
    {
        constexpr char const* splittable_mode_names[] = {"guided", "adaptive"};
        return splittable_mode_names[static_cast<std::size_t>(mode)];
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Policy, typename F, typename Iterator>
    struct splittable_task
    {
        template <typename F_>
        splittable_task(hpx::util::thread_description const& desc,
            threads::thread_pool_base* pool, Policy policy, F_&& f, Iterator it,
            std::size_t size, std::size_t hierarchical_threshold)
          : desc_(desc)
          , pool_(pool)
          , policy_(policy)
          , f_(std::forward<F_>(f))
          , it_(it)
          , size_(size)
          , hierarchical_threshold_(hierarchical_threshold)
        {
        }

        template <typename... Ts>
        void operator()(hpx::lcos::local::latch* outer_latch, Ts&&... ts)
        {
            // execute all tasks as needed
            call(HPX_FORWARD(Ts, ts)...);

            // notify outer waiting task
            if (outer_latch != nullptr)
            {
                outer_latch->count_down(1);
            }
        }

    private:
        template <typename... Ts>
        void call(Ts&&... ts)
        {
            auto mask = hpx::threads::get_idle_core_mask();
            std::size_t num_free = hpx::threads::count(mask);

            hpx::lcos::local::latch l(num_free + 1);

            std::size_t task_size = std::ceil(double(size_) / (num_free + 1));
            std::size_t remainder = size_ - num_free * task_size;

            if (num_free != 0 && task_size > hierarchical_threshold_)
            {
                // split the current task into num_free parts based on the number of
                // idle cores at the time of split. if task_size >
                // hierarchical_threshold_, task_size iterations are executed by
                // the current thread and the remainder iterations are assigned
                // to the first identified idle core as a splittable task.
                for (std::size_t i = 0, t = 0; t != num_free; ++t)
                {
                    if (hpx::threads::test(mask, t))
                    {
                        // pass schedule hint to place new task on an idle core
                        auto post_policy = policy_;
                        hpx::execution::experimental::with_hint(post_policy,
                            threads::thread_schedule_hint{std::int16_t(t)});

                        hpx::detail::post_policy_dispatch<Policy>::call(
                            post_policy, desc_, pool_,
                            splittable_task(desc_, pool_, policy_, f_, it_,
                                task_size, hierarchical_threshold_),
                            &l, ts...);

                        std::advance(it_, task_size);
                        ++i;
                    }
                }
            }
            else
            {
                l.count_down(num_free);
            }

            // run the remaining tasks directly
            for (std::size_t i = 0; i != remainder; (void) ++it_, ++i)
            {
                f_(*it_, ts...);
            }

            // wait for task scheduled above
            l.arrive_and_wait(1);
        }

    private:
        hpx::util::thread_description const& desc_;
        hpx::threads::thread_pool_base* pool_;
        Policy policy_;
        F f_;
        Iterator it_;
        std::size_t size_;
        std::size_t hierarchical_threshold_;
    };

    template <typename Policy, typename F, typename Iterator>
    splittable_task(hpx::util::thread_description const&,
        threads::thread_pool_base*, Policy, F&&, Iterator, std::size_t,
        std::size_t) -> splittable_task<Policy, std::decay_t<F>, Iterator>;

}    // namespace hpx::parallel::execution::detail
