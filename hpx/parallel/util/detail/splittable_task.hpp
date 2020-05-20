//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2020 Shahrzad Shirzad
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SPLITTABLE_TASK_HPP
#define HPX_SPLITTABLE_TASK_HPP

#include <type_traits>
#include <utility>

template <typename F>
struct splittable_task
{
    std::size_t start_;
    std::size_t stop_;
    std::size_t index_;
    std::size_t num_free_;
    F f_;
    static std::string split_type;

    template <typename F_, typename Shape>
    splittable_task(F_&& f, Shape const& elem, std::size_t cores)
      : start_(hpx::util::get<0>(elem))
      , stop_(hpx::util::get<1>(elem))
      , index_(hpx::util::get<2>(elem))
      , f_(std::forward<F_>(f))
    {
        if (split_type == "all")
            num_free_ = cores;
        else
            num_free_ = hpx::threads::get_idle_core_count() + 1;
    }

    void operator()()
    {
        hpx::future<void> result;
        std::size_t remainder =
            (stop_ - start_) * float(num_free_ - 1) / float(num_free_);
        std::cout << "number of "<<split_type << " cores: " <<num_free_ - 1 << "remaining number of iterations:" << remainder << std::endl;
        if ((num_free_ > 1) &&
            (remainder > 1))    //split the current task among the idle cores
        {
            num_free_ -= 1;
            result = hpx::async(splittable_task(f_,
                hpx::util::make_tuple(start_, start_ + remainder, index_ + 1),
                num_free_));
            start_ = start_ + remainder;
        }

	std::cout << " task index: " << index_ << " from: " << start_<< " to: " << stop_ << std::endl;

        f_(hpx::util::make_tuple(start_, stop_ - start_, index_));

        if (result.valid())
        {
            result.get();
        }
    }
};

template <typename F>
std::string splittable_task<F>::split_type = "all";

template <typename F, typename Shape>
splittable_task<typename std::decay<F>::type> make_splittable_task(
    F&& f, Shape const& s, std::string split_type)
{
    splittable_task<F>::split_type = split_type;
    std::size_t cores = hpx::get_os_thread_count();
    return splittable_task<typename std::decay<F>::type>(
        std::forward<F>(f), s, cores);
}

#endif    //HPX_SPLITTABLE_TASK_HPP
