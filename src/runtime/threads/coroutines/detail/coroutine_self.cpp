//  Copyright (c) 2008-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/threads/coroutines/detail/coroutine_self.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/thread_specific_ptr.hpp>

#include <cstddef>

namespace hpx { namespace threads { namespace coroutines { namespace detail
{
    struct tls_tag {};

    static util::thread_specific_ptr<coroutine_self*, tls_tag> self_;

    void coroutine_self::set_self(coroutine_self* self)
    {
        HPX_ASSERT(NULL != self_.get());
        *self_ = self;
    }

    coroutine_self* coroutine_self::get_self()
    {
        return (NULL == self_.get()) ? NULL : *self_;
    }

    void coroutine_self::init_self()
    {
        HPX_ASSERT(NULL == self_.get());
        self_.reset(new coroutine_self*(NULL));
    }

    void coroutine_self::reset_self()
    {
        self_.reset(NULL);
    }
}}}}
