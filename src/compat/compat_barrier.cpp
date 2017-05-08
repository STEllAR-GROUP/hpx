//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/compat/barrier.hpp>

#include <cstddef>

namespace hpx { namespace compat
{
    barrier::barrier(std::size_t number_of_threads)
      : number_of_threads_(number_of_threads),
        total_(barrier_flag),
        mtx_(),
        cond_()
    {}

    barrier::~barrier()
    {
        std::unique_lock<mutex_type> l(mtx_);

        while (total_ > barrier_flag)
        {
            // Wait until everyone exits the barrier
            cond_.wait(l);
        }
    }

    void barrier::wait()
    {
        std::unique_lock<mutex_type> l(mtx_);

        while (total_ > barrier_flag)
        {
            // wait until everyone exits the barrier
            cond_.wait(l);
        }

        // Are we the first to enter?
        if (total_ == barrier_flag)
            total_ = 0;

        ++total_;

        if (total_ == number_of_threads_)
        {
            total_ += barrier_flag - 1;
            cond_.notify_all();
        }
        else
        {
            while (total_ < barrier_flag)
            {
                // wait until enough threads enter the barrier
                cond_.wait(l);
            }
            --total_;

            // get entering threads to wake up
            if (total_ == barrier_flag)
            {
                cond_.notify_all();
            }
        }
    }
}}
