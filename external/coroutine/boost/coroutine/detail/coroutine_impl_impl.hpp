//  Copyright (c) 2008-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_COROUTINE_COROUTINE_IMPL_IMPL_HPP_20081127
#define BOOST_COROUTINE_COROUTINE_IMPL_IMPL_HPP_20081127

#include <boost/assert.hpp>
#include <boost/coroutine/detail/coroutine_impl.hpp>

namespace boost { namespace coroutines { namespace detail 
{
    template<typename CoroutineType, typename ContextImpl,
        template <typename> class Heap>
    void coroutine_impl<CoroutineType, ContextImpl, Heap>::set_self(self_type* self)
    {
        BOOST_ASSERT(NULL != self_.get());
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
        BOOST_ASSERT(NULL == self_.get());
        self_.reset(new self_type* (NULL));
    }

    template<typename CoroutineType, typename ContextImpl,
        template <typename> class Heap>
    void coroutine_impl<CoroutineType, ContextImpl, Heap>::reset_self()
    {
        self_.reset(NULL);
    }
}}}

#endif
