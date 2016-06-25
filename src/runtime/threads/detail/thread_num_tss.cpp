//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/threads/detail/thread_num_tss.hpp>
#include <hpx/util/assert.hpp>

#include <utility>

namespace hpx { namespace threads { namespace detail
{
    // the TSS holds the number associated with a given OS thread
    thread_num_tss thread_num_tss_;

    ///////////////////////////////////////////////////////////////////////////
    hpx::util::thread_specific_ptr<
            std::size_t, thread_num_tss::tls_tag
        > thread_num_tss::thread_num_;

    void thread_num_tss::init_tss(std::size_t num)
    {
        // shouldn't be initialized yet
        if (nullptr == thread_num_tss::thread_num_.get())
        {
            thread_num_tss::thread_num_.reset(new std::size_t);
            *thread_num_tss::thread_num_.get() = num;
        }
    }

    void thread_num_tss::deinit_tss()
    {
        thread_num_tss::thread_num_.reset();
    }

    std::size_t thread_num_tss::set_tss_threadnum(std::size_t num)
    {
        // should have been initialized
        HPX_ASSERT(nullptr != thread_num_tss::thread_num_.get());

        std::swap(*thread_num_tss::thread_num_.get(), num);
        return num;
    }

    std::size_t thread_num_tss::get_worker_thread_num() const
    {
        if (nullptr != thread_num_tss::thread_num_.get())
            return *thread_num_tss::thread_num_;

        // some OS threads are not managed by the thread-manager
        return std::size_t(-1);
    }
}}}
