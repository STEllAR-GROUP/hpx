//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2020 Shahrzad Shirzad
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SPLITTABLE_TASK_HPP
#define HPX_SPLITTABLE_TASK_HPP

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
        all = 0,
        idle = 1,
        idle_mask = 2,
        all_multiple_tasks = 3
    };

    inline char const* const get_splittable_mode_name(splittable_mode mode)
    {
        static constexpr char const* const splittable_mode_names[] = {
            "all", "idle", "idle_mask", "all_multiple_tasks"};
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
          : start_(hpx::util::get<0>(elem))
          , stop_(hpx::util::get<1>(elem))
          , index_(hpx::util::get<2>(elem))
          , num_free_(num_free)
          , f_(std::forward<F_>(f))
          , exec_(exec)
          , split_type_(split_type)
          , min_task_size_(min_task_size)
        {
        }

        void operator()(hpx::latch* outer_latch = nullptr)
        {
            if (split_type_ == splittable_mode::idle_mask)
            {
                call_idle_mask();
            }
            else if (split_type_ == splittable_mode::all_multiple_tasks)
            {
                call_all_multiple_tasks();
            }
            else if (split_type_ == splittable_mode::idle)
            {
                call_idle();
            }
            else
            {
                call();
            }

            // notify outer waiting task
            if (outer_latch != nullptr)
            {
                outer_latch->count_down(1);
            }
        }

    private:
        void call_idle()
        {
            auto mask = hpx::threads::get_idle_core_mask();
            num_free_ = hpx::threads::count(mask);

            hpx::latch l(2);

            std::size_t task_size =
                std::ceil((stop_ - start_) / (num_free_ + 1));
            std::size_t remainder = stop_ - start_ - task_size;

            hpx::util::thread_description desc(f_);

            if ((num_free_ != 0 && task_size > min_task_size_) && remainder > 1)
            {
                // split the current task, create one for each idle core
                for (std::size_t i = 0, j = 0; i != 1; ++j)
                {
                    if (hpx::threads::test(mask, j))
                    {
                        // pass schedule hint to place new task on empty core
                        using policy = hpx::launch::async_policy;
                        detail::post_policy_dispatch<policy>::call(policy{},
                            desc, exec_.get_priority(), exec_.get_stacksize(),
                            hpx::threads::thread_schedule_hint(std::int16_t(j)),
                            splittable_task(exec_, f_,
                                hpx::util::make_tuple(
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

            f_(hpx::util::make_tuple(start_, stop_ - start_, index_));

            // wait for task scheduled above
            l.arrive_and_wait(1);
        }

        void call_idle_mask()
        {
            auto mask = hpx::threads::get_idle_core_mask();
            num_free_ = hpx::threads::count(mask);

            hpx::latch l(num_free_ + 1);

            std::size_t task_size =
                std::ceil((stop_ - start_) / (num_free_ + 1));
            std::size_t remainder = stop_ - start_ - task_size;

            if ((num_free_ != 0 && task_size > min_task_size_) && remainder > 1)
            {
                hpx::util::thread_description desc(f_);

                // split the current task, create one for each idle core
                for (std::size_t i = 0, j = 0; i != num_free_; ++j)
                {
                    if (hpx::threads::test(mask, j))
                    {
                        // pass schedule hint to place new task on empty core
                        using policy = hpx::launch::async_policy;
                        detail::post_policy_dispatch<policy>::call(policy{},
                            desc, exec_.get_priority(), exec_.get_stacksize(),
                            hpx::threads::thread_schedule_hint(std::int16_t(j)),
                            splittable_task(exec_, f_,
                                hpx::util::make_tuple(
                                    stop_ - task_size, stop_, index_ + i + 1),
                                num_free_, split_type_, min_task_size_),
                            &l);

                        stop_ = stop_ - task_size;
                        ++i;
                    }
                }
            }
            else
            {
                l.count_down(num_free_);
            }

            f_(hpx::util::make_tuple(start_, stop_ - start_, index_));

            // wait for task scheduled above
            l.arrive_and_wait(1);
        }

        void call()
        {
            hpx::latch l(2);

            std::size_t task_size =
                std::ceil((stop_ - start_) / (num_free_ + 1));
            std::size_t remainder = stop_ - start_ - task_size;

            if ((num_free_ != 0 && task_size > min_task_size_) && remainder > 1)
            {
                // split the current task
                exec_.post(splittable_task(exec_, f_,
                               hpx::util::make_tuple(
                                   stop_ - remainder, stop_, index_ + 1),
                               num_free_ - 1, split_type_, min_task_size_),
                    &l);
                stop_ = stop_ - remainder;
            }
            else
            {
                l.count_down(1);
            }

            f_(hpx::util::make_tuple(start_, stop_ - start_, index_));

            // wait for task scheduled above
            l.arrive_and_wait(1);
        }

        void call_all_multiple_tasks()
        {
            hpx::latch l(num_free_ + 1);

            std::size_t task_size =
                std::ceil((stop_ - start_) / (num_free_ + 1));
            std::size_t remainder = stop_ - start_ - task_size;

            if ((num_free_ != 0 && task_size > min_task_size_) && remainder > 1)
            {
                hpx::util::thread_description desc(f_);

                // split the current task, create one for each of num_free_ cores
                for (std::size_t i = 0; i != num_free_; ++i)
                {
                    using policy = hpx::launch::async_policy;
                    detail::post_policy_dispatch<policy>::call(policy{}, desc,
                        exec_.get_priority(), exec_.get_stacksize(),
                        hpx::threads::thread_schedule_hint(),
                        splittable_task(exec_, f_,
                            hpx::util::make_tuple(
                                stop_ - task_size, stop_, index_ + i + 1),
                            num_free_ - 1, split_type_, min_task_size_),
                        &l);

                    stop_ = stop_ - task_size;
                }
            }
            else
            {
                l.count_down(num_free_);
            }

            f_(hpx::util::make_tuple(start_, stop_ - start_, index_));

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

#endif    // HPX_SPLITTABLE_TASK_HPP