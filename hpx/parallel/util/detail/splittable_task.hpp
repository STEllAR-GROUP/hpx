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

    template <typename F_, typename Shape>
    splittable_task(F_&& f, Shape const& elem, std::size_t cores)
      : start_(hpx::util::get<0>(elem))
      , stop_(hpx::util::get<1>(elem))
      , index_(hpx::util::get<2>(elem))
      , num_free_(cores)
      , f_(std::forward<F_>(f))

    {
    }

    void operator()()
    {
        std::vector<hpx::future<void>> futures;
	std::size_t remainder = (stop_ - start_) * float(num_free_ - 1) / float(num_free_);
        //std::cout<<"remainder:"<<remainder<<std::endl;
        if ((num_free_ > 1) &
            (remainder > 1))    //split the current task among the idle cores
        {
            num_free_ -= 1;
            futures.push_back(hpx::async(splittable_task(f_,
                hpx::util::make_tuple(
                    start_, start_ + remainder, index_ + 1),
                num_free_)));
            start_ = start_ + remainder;
        }

        //std::cout << "task " << num_free_ << " from: " << start_
        //          << " to: " << stop_ << std::endl;

        f_(hpx::util::make_tuple(start_, stop_, 0));
        //num_free_ -= 1;

        if (!futures.empty())
        {
            wait_all(futures);
        }
    }
};

template <typename F, typename Shape>
splittable_task<typename std::decay<F>::type> make_splittable_task(
    F&& f, Shape const& s, std::size_t cores)
{
    return splittable_task<typename std::decay<F>::type>(
        std::forward<F>(f), s, cores);
}

#endif    //HPX_SPLITTABLE_TASK_HPP

