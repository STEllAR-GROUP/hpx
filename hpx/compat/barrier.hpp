//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedname:boost::barrier

#ifndef HPX_COMPAT_BARRIER_HPP
#define HPX_COMPAT_BARRIER_HPP

#include <hpx/config.hpp>

#include <hpx/compat/condition_variable.hpp>
#include <hpx/compat/mutex.hpp>

#include <climits>
#include <cstddef>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace compat
{
    class HPX_EXPORT barrier
    {
    private:
        typedef compat::mutex mutex_type;

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
        compat::condition_variable cond_;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
