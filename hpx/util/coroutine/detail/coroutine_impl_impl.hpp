//  Copyright (c) 2008-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COROUTINE_COROUTINE_IMPL_IMPL_HPP_20081127
#define HPX_COROUTINE_COROUTINE_IMPL_IMPL_HPP_20081127

#include <hpx/assert.hpp>
#include <hpx/util/coroutine/detail/coroutine_impl.hpp>

namespace hpx { namespace util { namespace coroutines { namespace detail
{
    template<typename CoroutineType, typename ContextImpl,
        template <typename> class Heap>
    void coroutine_impl<CoroutineType, ContextImpl, Heap>::set_self(self_type* self)
    {
        HPX_ASSERT(NULL != self_.get());
        *self_ = self;
    }

    template<typename CoroutineType, typename ContextImpl,
        template <typename> class Heap>
    typename coroutine_impl<CoroutineType, ContextImpl, Heap>::self_type*
    coroutine_impl<CoroutineType, ContextImpl, Heap>::get_self()
    {
        return (NULL == self_.get()) ? NULL : *self_;
    }

    template<typename CoroutineType, typename ContextImpl,
        template <typename> class Heap>
    void coroutine_impl<CoroutineType, ContextImpl, Heap>::init_self()
    {
        HPX_ASSERT(NULL == self_.get());
        self_.reset(new self_type* (NULL));
    }

    template<typename CoroutineType, typename ContextImpl,
        template <typename> class Heap>
    void coroutine_impl<CoroutineType, ContextImpl, Heap>::reset_self()
    {
        self_.reset(NULL);
    }
}}}}

#endif
