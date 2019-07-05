//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_THREAD_NUM_TSS_JUL_17_2015_0811PM)
#define HPX_RUNTIME_THREADS_DETAIL_THREAD_NUM_TSS_JUL_17_2015_0811PM

#include <hpx/config.hpp>

#include <cstddef>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace detail
{
    HPX_EXPORT std::size_t set_thread_num_tss(std::size_t num);
    HPX_EXPORT std::size_t get_thread_num_tss();

    ///////////////////////////////////////////////////////////////////////////
    struct reset_tss_helper
    {
        reset_tss_helper(std::size_t thread_num)
          : thread_num_(set_thread_num_tss(thread_num))
        {}

        ~reset_tss_helper()
        {
            set_thread_num_tss(thread_num_);
        }

        std::size_t previous_thread_num() const
        {
            return thread_num_;
        }

    private:
        std::size_t thread_num_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
