//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedname:boost::barrier

#pragma once

#include <hpx/config.hpp>

#include <climits>
#include <condition_variable>
#include <cstddef>
#include <mutex>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util {
    class HPX_CORE_EXPORT barrier
    {
    private:
        typedef std::mutex mutex_type;

        HPX_STATIC_CONSTEXPR std::size_t barrier_flag =
            static_cast<std::size_t>(1) << (CHAR_BIT * sizeof(std::size_t) - 1);

    public:
        barrier(std::size_t number_of_threads);
        ~barrier();

        void wait();

    private:
        std::size_t const number_of_threads_;
        std::size_t total_;

        mutable mutex_type mtx_;
        std::condition_variable cond_;
    };
}}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>
