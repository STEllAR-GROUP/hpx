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
#include <hpx/include/iostreams.hpp>

template <typename F>
struct splittable_task
{
    std::size_t start_;
    std::size_t stop_;
    std::size_t index_;
    std::size_t num_free_;
    F f_;

    template <typename F_, typename Shape>
    splittable_task(
        F_&& f, Shape const& elem, std::size_t cores, std::string split_type)
      : start_(hpx::util::get<0>(elem))
      , stop_(hpx::util::get<1>(elem))
      , index_(hpx::util::get<2>(elem))
      , f_(std::forward<F_>(f))
      , split_type_(split_type)
    {
         num_free_ = cores;

	//if (split_type_ == "all")
        //    num_free_ = cores;
        //else
        //    num_free_ = hpx::threads::get_idle_core_count() + 1;
    }

    void operator()()
    {
        hpx::future<void> result;
	if (split_type_ == "idle")
		num_free_ = hpx::threads::get_idle_core_count() + 1;

	std::size_t remainder = static_cast<std::size_t>(std::floor((stop_ - start_) * double(num_free_ - 1) / double(num_free_)));

	if ((num_free_ > 1) &&
            (remainder > 0))    //split the current task
        {
            num_free_ -= 1;
            result = hpx::async(splittable_task(f_,
                hpx::util::make_tuple(stop_ - remainder, stop_, index_ + 1),
                num_free_, split_type_));
            stop_ = stop_ - remainder;
        }

        f_(hpx::util::make_tuple(start_, stop_ - start_, index_));

        if (result.valid())
        {
            result.get();
        }
    }

private:
    std::string split_type_;
};

template <typename F, typename Shape>
splittable_task<typename std::decay<F>::type> make_splittable_task(
    F&& f, Shape const& s, std::string split_type)
{
    std::size_t cores = hpx::get_os_thread_count();
    return splittable_task<typename std::decay<F>::type>(
        std::forward<F>(f), s, cores, split_type);
}

#endif    //HPX_SPLITTABLE_TASK_HPP
