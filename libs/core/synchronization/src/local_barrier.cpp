//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/synchronization/barrier.hpp>

#include <cstddef>
#include <mutex>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::detail {

    void intrusive_ptr_add_ref(barrier_data* p) noexcept
    {
        ++p->count_;
    }

    void intrusive_ptr_release(barrier_data* p) noexcept
    {
        if (0 == --p->count_)
        {
            delete p;
        }
    }
}    // namespace hpx::detail

///////////////////////////////////////////////////////////////////////////////
namespace hpx::lcos::local {

    barrier::barrier(std::size_t expected)
      : number_of_threads_(expected)
      , total_(barrier_flag)
      , mtx_()
      , cond_()
    {
    }

    barrier::~barrier()
    {
        std::unique_lock<mutex_type> l(mtx_);

        while (total_ > barrier_flag)    //-V776
        {
            // Wait until everyone exits the barrier
            cond_.wait(l, "barrier::~barrier");
        }
    }

    void barrier::wait()
    {
        std::unique_lock<mutex_type> l(mtx_);

        while (total_ > barrier_flag)    //-V776
        {
            // wait until everyone exits the barrier
            cond_.wait(l, "barrier::wait");
        }

        // Are we the first to enter?
        if (total_ == barrier_flag)
            total_ = 0;

        ++total_;

        if (total_ == number_of_threads_)
        {
            total_ += barrier_flag - 1;
            cond_.notify_all(HPX_MOVE(l));
        }
        else
        {
            while (total_ < barrier_flag)    //-V776
            {
                // wait until enough threads enter the barrier
                cond_.wait(l, "barrier::wait");
            }
            --total_;

            // get entering threads to wake up
            if (total_ == barrier_flag)    //-V547
            {
                cond_.notify_all(HPX_MOVE(l));
            }
        }
    }

    void barrier::count_up()
    {
        std::unique_lock<mutex_type> l(mtx_);
        ++number_of_threads_;
    }

    void barrier::reset(std::size_t number_of_threads)
    {
        std::unique_lock<mutex_type> l(mtx_);
        this->number_of_threads_ = number_of_threads;
    }

}    // namespace hpx::lcos::local
