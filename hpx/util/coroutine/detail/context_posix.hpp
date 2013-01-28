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

#ifndef HPX_COROUTINE_CONTEXT_POSIX_HPP_20060601
#define HPX_COROUTINE_CONTEXT_POSIX_HPP_20060601

// NOTE (per http://lists.apple.com/archives/darwin-dev/2008/Jan/msg00232.html):
// > Why the bus error? What am I doing wrong?
// This is a known issue where getcontext(3) is writing past the end of the
// ucontext_t struct when _XOPEN_SOURCE is not defined (rdar://problem/5578699 ).
// As a workaround, define _XOPEN_SOURCE before including ucontext.h.
#if defined(__APPLE__) && ! defined(_XOPEN_SOURCE)
#define _XOPEN_SOURCE
// However, the above #define will only affect <ucontext.h> if it has not yet
// been #included by something else!
#if defined(_STRUCT_UCONTEXT)
#error You must #include coroutine headers before anything that #includes <ucontext.h>
#endif
#endif

#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/detail/atomic_count.hpp>

#if defined(__FreeBSD__) || (defined(_XOPEN_UNIX) && defined(_XOPEN_VERSION) && _XOPEN_VERSION >= 500)

// OS X 10.4 -- despite passing the test above -- doesn't support
// swapcontext() et al. Use GNU Pth workalike functions.
#if defined(__APPLE__) && (__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 1050)

#include "pth/pth.h"
#include <cerrno>

namespace hpx { namespace util { namespace coroutines { namespace detail 
{
  namespace posix { namespace pth {

    inline int check(int rc)
    {
        // The makecontext() functions return zero for success, nonzero for
        // error. The Pth library returns TRUE for success, FALSE for error,
        // with errno set to the nonzero error in the latter case. Map the Pth
        // returns to ucontext style.
        return rc? 0 : errno;
    }

}}}}}}

#define HPX_COROUTINE_POSIX_IMPL "Pth implementation"
#define HPX_COROUTINE_DECLARE_CONTEXT(name) pth_uctx_t name
#define HPX_COROUTINE_CREATE_CONTEXT(ctx)                                     \
    hpx::util::coroutines::detail::posix::pth::check(pth_uctx_create(&(ctx)))
#define HPX_COROUTINE_MAKE_CONTEXT(ctx, stack, size, startfunc, startarg, exitto) \
    /* const sigset_t* sigmask = NULL: we don't expect per-context signal masks */ \
    hpx::util::coroutines::detail::posix::pth::check(                         \
        pth_uctx_make(*(ctx), static_cast<char*>(stack), (size), NULL,        \
        (startfunc), (startarg), (exitto)))
#define HPX_COROUTINE_SWAP_CONTEXT(from, to)                                  \
    hpx::util::coroutines::detail::posix::pth::check(pth_uctx_switch(*(from), *(to)))
#define HPX_COROUTINE_DESTROY_CONTEXT(ctx)                                    \
    hpx::util::coroutines::detail::posix::pth::check(pth_uctx_destroy(ctx))

#else // generic Posix platform (e.g. OS X >= 10.5)

/*
 * makecontext based context implementation. Should be available on all
 * SuSv2 compliant UNIX systems.
 * NOTE: this implementation is not
 * optimal as the makecontext API saves and restore the signal mask.
 * This requires a system call for every context switch that really kills
 * performance. Still is very portable and guaranteed to work.
 * NOTE2: makecontext and friends are declared obsolescent in SuSv3, but
 * it is unlikely that they will be removed any time soon.
 */
#include <ucontext.h>
#include <cstddef>                  // ptrdiff_t

namespace hpx { namespace util { namespace coroutines { namespace detail 
{
  namespace posix { namespace ucontext {

    inline int make_context(::ucontext_t* ctx, void* stack, std::ptrdiff_t size,
                            void (*startfunc)(void*), void* startarg,
                            ::ucontext_t* exitto = NULL)
    {
        int error = ::getcontext(ctx);
        if (error)
            return error;

        ctx->uc_stack.ss_sp = (char*)stack;
        ctx->uc_stack.ss_size = size;
        ctx->uc_link = exitto;

        typedef void (*ctx_main)();
        //makecontext can't fail.
        ::makecontext(ctx,
                      (ctx_main)(startfunc),
                      1,
                      startarg);
        return 0;
    }

}}}}}}

#define HPX_COROUTINE_POSIX_IMPL "ucontext implementation"
#define HPX_COROUTINE_DECLARE_CONTEXT(name) ::ucontext_t name
#define HPX_COROUTINE_CREATE_CONTEXT(ctx) /* nop */
#define HPX_COROUTINE_MAKE_CONTEXT(ctx, stack, size, startfunc, startarg, exitto) \
    hpx::util::coroutines::detail::posix::ucontext::make_context(              \
        ctx, stack, size, startfunc, startarg, exitto)
#define HPX_COROUTINE_SWAP_CONTEXT(pfrom, pto) ::swapcontext((pfrom), (pto))
#define HPX_COROUTINE_DESTROY_CONTEXT(ctx) /* nop */

#endif // generic Posix platform

#include <signal.h>                 // SIGSTKSZ
#include <boost/noncopyable.hpp>
#include <hpx/util/coroutine/exception.hpp>
#include <hpx/util/coroutine/detail/posix_utility.hpp>
#include <hpx/util/coroutine/detail/swap_context.hpp>

namespace hpx { namespace util { namespace coroutines {

  // some platforms need special preparation of the main thread
  struct prepare_main_thread
  {
      prepare_main_thread() {}
      ~prepare_main_thread() {}
  };

  namespace detail { namespace posix
  {
    /*
     * Posix implementation for the context_impl_base class.
     * @note context_impl is not required to be consistent
     * If not initialized it can only be swapped out, not in
     * (at that point it will be initialized).
     *
     */
    class ucontext_context_impl_base : detail::context_impl_base
    {
    public:
      ucontext_context_impl_base()
      {
          HPX_COROUTINE_CREATE_CONTEXT(m_ctx);
      }
      ~ucontext_context_impl_base()
      {
          HPX_COROUTINE_DESTROY_CONTEXT(m_ctx);
      }

    private:
      /*
       * Free function. Saves the current context in @p from
       * and restores the context in @p to.
       */
      friend void
      swap_context(ucontext_context_impl_base& from,
                   const ucontext_context_impl_base& to,
                   default_hint)
      {
          int  error = HPX_COROUTINE_SWAP_CONTEXT(&from.m_ctx, &to.m_ctx);
          (void)error;
          BOOST_ASSERT(error == 0);
      }

    protected:
      HPX_COROUTINE_DECLARE_CONTEXT(m_ctx);
    };

    class ucontext_context_impl
      : public ucontext_context_impl_base,
        private boost::noncopyable
    {
    public:
        typedef ucontext_context_impl_base context_impl_base;

        enum { default_stack_size = SIGSTKSZ };

        /**
         * Create a context that on restore invokes Functor on
         *  a new stack. The stack size can be optionally specified.
         */
        template<typename Functor>
        explicit ucontext_context_impl(Functor& cb, std::ptrdiff_t stack_size)
          : m_stack_size(stack_size == -1? default_stack_size: stack_size),
            m_stack(alloc_stack(m_stack_size))
        {
            BOOST_ASSERT(m_stack);
            typedef void cb_type(Functor*);
            cb_type * cb_ptr = &trampoline<Functor>;
            int error = HPX_COROUTINE_MAKE_CONTEXT(
                &m_ctx, m_stack, m_stack_size, (void (*)(void*))(cb_ptr), &cb, NULL);
            (void)error;
            BOOST_ASSERT(error == 0);
        }

        ~ucontext_context_impl()
        {
            if(m_stack)
                free_stack(m_stack, m_stack_size);
        }

        // Return the size of the reserved stack address space.
        std::ptrdiff_t get_stacksize() const
        {
            return m_stack_size;
        }

        // global functions to be called for each OS-thread after it started
        // running and before it exits
        static void thread_startup(char const* thread_type) {}
        static void thread_shutdown() {}

        void reset_stack() {}
        void rebind_stack() 
        {
            if (m_stack) 
                increment_stack_recycle_count();
        }

        typedef boost::detail::atomic_count counter_type;

        static boost::uint64_t get_stack_unbind_count()
        {
            return 0;
        }

        static counter_type& get_stack_recycle_counter()
        {
            static counter_type counter(0);
            return counter;
        }
        static boost::uint64_t get_stack_recycle_count()
        {
            return get_stack_recycle_counter();
        }
        static boost::uint64_t increment_stack_recycle_count()
        {
            return ++get_stack_recycle_counter();
        }

    private:
        // declare m_stack_size first so we can use it to initialize m_stack
        std::ptrdiff_t m_stack_size;
        void * m_stack;
    };

    typedef ucontext_context_impl context_impl;
  }
}}}}

#else

/**
 * This #else clause is essentially unchanged from the original Google Summer
 * of Code version of Boost.Coroutine, which comments:
 * "Context swapping can be implemented on most posix systems lacking *context
 * using the sigaltstack+longjmp trick."
 * This is in fact what the (highly portable) Pth library does, so if you
 * encounter such a system, perhaps the best approach would be to twiddle the
 * #if logic in this header to use the pth.h implementation above.
 */
#error No context implementation for this POSIX system.

#endif
#endif
