//  Copyright (c) 2006, Giovanni P. Deretta
//
//  This code may be used under either of the following two licences:
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE. OF SUCH DAMAGE.
//
//  Or:
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COROUTINE_DEFAULT_CONTEXT_IMPL_HPP_20060601
#define HPX_COROUTINE_DEFAULT_CONTEXT_IMPL_HPP_20060601

#include <boost/config.hpp>
#include <boost/version.hpp>

/*
   ContextImpl concept documentation.
   A ContextImpl holds a context plus its stack.
   - ContextImpl must have constructor with the following signature:

     template<typename Functor>
     ContextImpl(Functor f, std::ptrdiff_t stack_size);

     Preconditions:
     f is a generic function object (support for function pointers
     is not required.
     stack_size is the size of the stack allocated
     for the context. The stack_size is only an hint. If stack_size is -1
     it should default to a sensible value.
     Postconditions:
     f is bound to this context in its own stack.
     When the context is activated with swap_context for the first time
     f is entered.

   - ContextImpl destructor must properly dispose of the stack and perform
     any other clean up action required.

   - ContextImpl is not required to be DefaultConstructible nor Copyable.

   - ContextImpl must expose the following typedef:

     typedef unspecified-type context_impl_base;

     ContextImpl must be convertible to context_impl_base.
     context_impl_base must have conform to the ContextImplBase concept:

     - DefaultConstructible. A default constructed ContextImplBase is in
       an initialized state.
       A ContextImpl is an initialized ContextImplBase.

     - ContextImplBase is Copyable. A copy of a ContextImplBase holds
       The same informations as the original. Once a ContextImplBase
       is used as an argument to swap_context, all its copies become stale.
       (that is, only one copy of ContextImplBase can be used).
       A ContextImpl cannot be sliced by copying it to a ContextImplBase.

     - void swap_context(ContextImplBase& from,
                         const ContextImplBase& to)

       This free function is called unqualified (via ADL).
       Preconditions:
       Its 'to' argument must be an initialized ContextImplBase.
       Its 'from' argument may be an uninitialized ContextImplBase that
       will be initialized by a swap_context.
       Postconditions:
       The current context is saved in the 'from' context, and the
       'to' context is restored.
       It is undefined behavior if the 'to' argument is an invalid
       (uninitialized) swap context.

     A ContextImplBase is meant to be used when an empty temporary
     context is needed to store the current context before
     restoring a ContextImpl and no current context is
     available. It could be possible to simply have ContextImpl
     default constructible, but on destruction it would need to
     check if a stack has been allocated and would slow down the
     fast invocation path. Also a stack-full context could not be
     make copyable.
*/

#if HPX_COROUTINE_USE_GENERIC_CONTEXT != 0

#if BOOST_VERSION >= 105100

#include <hpx/util/coroutine/detail/context_generic_context.hpp>
namespace hpx { namespace util { namespace coroutines { namespace detail 
{
    typedef generic_context::context_impl default_context_impl;
}}}}

#else
#error Boost.Context is available only with Boost V1.51 or later
#endif

#elif defined(__linux) || defined(linux) || defined(__linux__)

#include <hpx/util/coroutine/detail/context_linux_x86.hpp>
namespace hpx { namespace util { namespace coroutines { namespace detail 
{
    typedef lx::context_impl default_context_impl;
}}}}

#elif defined(_POSIX_VERSION)

#include <hpx/util/coroutine/detail/context_posix.hpp>
namespace hpx { namespace util { namespace coroutines { namespace detail 
{
    typedef posix::context_impl default_context_impl;
}}}}

#elif defined(BOOST_WINDOWS)

#include <hpx/util/coroutine/detail/context_windows_fibers.hpp>
namespace hpx { namespace util { namespace coroutines { namespace detail 
{
    typedef windows::context_impl default_context_impl;
}}}}

#else

#error No default_context_impl available for this system

#endif // HPX_COROUTINE_USE_GENERIC_CONTEXT

namespace hpx { namespace util { namespace coroutines
{
    // functions to be called for each thread after it started running
    // and before it exits
    inline void thread_startup(char const* thread_type)
    {
        detail::default_context_impl::thread_startup(thread_type);
    }

    inline void thread_shutdown()
    {
        detail::default_context_impl::thread_shutdown();
    }
}}}

#endif
