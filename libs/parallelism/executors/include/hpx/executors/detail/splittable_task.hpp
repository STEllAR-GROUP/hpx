//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2021 Shahrzad Shirzad
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/execution/detail/post_policy_dispatch.hpp>
#include <hpx/modules/datastructures.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/threading_base/thread_description.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace execution {

    ///////////////////////////////////////////////////////////////////////////
    enum class splittable_mode
    {
        guided = 0,
        adaptive = 1,
    };

    inline char const* get_splittable_mode_name(splittable_mode mode)
    {
        static constexpr char const* const splittable_mode_names[] = {
            "guided", "adaptive"};
        return splittable_mode_names[static_cast<std::size_t>(mode)];
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor, typename F>
    struct splittable_task
    {
        template <typename F_, typename Shape>
        splittable_task(Executor& exec, F_&& f, Shape const& elem,
            std::size_t num_free, splittable_mode split_type,
            std::size_t min_task_size)
          : start_(hpx::get<0>(elem))
          , stop_(hpx::get<1>(elem))
          , index_(hpx::get<2>(elem))
          , num_free_(num_free)
          , f_(std::forward<F_>(f))
          , exec_(exec)
          , split_type_(split_type)
          , min_task_size_(min_task_size)
        {
        }

        void operator()(hpx::latch* outer_latch = nullptr)
        {
            if (split_type_ == splittable_mode::adaptive)
            {
                call_adaptive();
            }
            else
            {
                call_guided();
            }

            // notify outer waiting task
            if (outer_latch != nullptr)
            {
                outer_latch->count_down(1);
            }
        }

    private:
        void call_adaptive()
        {
            auto mask = hpx::threads::get_idle_core_mask();
            num_free_ = hpx::threads::count(mask);

            hpx::latch l(2);

            std::size_t task_size =
                std::ceil(float(stop_ - start_) / (num_free_ + 1));
            std::size_t remainder = stop_ - start_ - task_size;

            hpx::util::thread_description desc(f_);

            if ((num_free_ != 0 && task_size > min_task_size_) && remainder > 0)
            {
                // split the current task into two parts based on the number
                // of idle cores at the time of split.
                // if task_size > min_task_size_, task_size iterations are
                // executed by the current thread and the remainder iterations
                // are assigned to the first identified idle core as a splittable task.

                for (std::size_t i = 0, j = 0; i != 1; ++j)
                {
                    if (hpx::threads::test(mask, j))
                    {
                        // pass schedule hint to place new task on an idle core
                        using policy = hpx::launch::async_policy;
                        detail::post_policy_dispatch<policy>::call(policy{},
                            desc, exec_.get_priority(), exec_.get_stacksize(),
                            hpx::threads::thread_schedule_hint(std::int16_t(j)),
                            splittable_task(exec_, f_,
                                hpx::make_tuple(
                                    stop_ - remainder, stop_, index_ + i + 1),
                                num_free_, split_type_, min_task_size_),
                            &l);

                        stop_ = stop_ - remainder;
                        ++i;
                    }
                }
            }
            else
            {
                l.count_down(1);
            }

            f_(hpx::make_tuple(start_, stop_ - start_, index_));

            // wait for task scheduled above
            l.arrive_and_wait(1);
        }

        void call_guided()
        {
            // this mode basically splits the work equally among all the
            // cores by splitting the current task into two parts
            // if task_size > min_task_size_.
            // The smaller part is executed by the current thread and the
            // larger part would be scheduled to be executed as a splittable
            // task.

            hpx::latch l(2);

            std::size_t task_size =
                std::ceil(float(stop_ - start_) / (num_free_ + 1));
            std::size_t remainder = stop_ - start_ - task_size;

            if ((num_free_ != 0 && task_size > min_task_size_) && remainder > 0)
            {
                // split the current task into two parts.
                exec_.post(
                    splittable_task(exec_, f_,
                        hpx::make_tuple(stop_ - remainder, stop_, index_ + 1),
                        num_free_ - 1, split_type_, min_task_size_),
                    &l);
                stop_ = stop_ - remainder;
            }
            else
            {
                l.count_down(1);
            }

            f_(hpx::make_tuple(start_, stop_ - start_, index_));

            // wait for task scheduled above
            l.arrive_and_wait(1);
        }

    private:
        std::size_t start_;
        std::size_t stop_;
        std::size_t index_;
        std::size_t num_free_;
        F f_;
        Executor& exec_;
        splittable_mode split_type_;
        std::size_t min_task_size_;
        char const* desc_;
    };

    template <typename Executor, typename F, typename Shape>
    splittable_task<Executor, typename std::decay<F>::type>
    make_splittable_task(Executor& exec, F&& f, Shape const& s,
        splittable_mode split_type, std::size_t min_task_size)
    {
        std::size_t num_free = hpx::get_os_thread_count() - 1;
        return splittable_task<Executor, typename std::decay<F>::type>(
            exec, std::forward<F>(f), s, num_free, split_type, min_task_size);
    }

}}}    // namespace hpx::parallel::execution
