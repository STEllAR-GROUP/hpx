//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_THREAD_NUM_TSS_JUL_17_2015_0811PM)
#define HPX_RUNTIME_THREADS_DETAIL_THREAD_NUM_TSS_JUL_17_2015_0811PM

#include <hpx/config.hpp>
#include <hpx/util/thread_specific_ptr.hpp>

#include <cstdarg>

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    class thread_num_tss
    {
    public:
        std::size_t set_tss_threadnum(std::size_t num);
        void init_tss(std::size_t num);
        void deinit_tss();

        std::size_t get_worker_thread_num() const;

    private:
        // the TSS holds the number associated with a given OS thread
        struct tls_tag {};
        static hpx::util::thread_specific_ptr<std::size_t, tls_tag> thread_num_;
    };

    // the TSS holds the number associated with a given OS thread
    extern thread_num_tss thread_num_tss_;

    ///////////////////////////////////////////////////////////////////////////
    struct reset_tss_helper
    {
        reset_tss_helper(std::size_t thread_num)
          : thread_num_(thread_num_tss_.set_tss_threadnum(thread_num))
        {}

        ~reset_tss_helper()
        {
            thread_num_tss_.set_tss_threadnum(thread_num_);
        }

        std::size_t previous_thread_num() const
        {
            return thread_num_;
        }

    private:
        std::size_t thread_num_;
    };
}}}

#endif
