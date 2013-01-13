//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/executors/default_executor.hpp>
#include <hpx/lcos/local/once.hpp>
#include <hpx/util/reinitializable_static.hpp>

namespace hpx { namespace threads
{
    executor& executor::default_executor()
    {
        typedef util::reinitializable_static<
            executors::default_executor, tag, 1, lcos::local::once_flag
        > static_type;

        static_type instance;
        return instance.get();
    }
}}

