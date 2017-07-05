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

#ifndef HPX_RUNTIME_THREADS_COROUTINES_DETAIL_CONTEXT_POSIX_HPP
#define HPX_RUNTIME_THREADS_COROUTINES_DETAIL_CONTEXT_POSIX_HPP

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

#include <hpx/util/assert.hpp>
#include <hpx/util/get_and_reset_value.hpp>
#include <hpx/util/unused.hpp>

#include <boost/atomic.hpp>

#if defined(__FreeBSD__) || (defined(_XOPEN_UNIX) && defined(_XOPEN_VERSION) \
            && _XOPEN_VERSION >= 500)

// OS X 10.4 -- despite passing the test above -- doesn't support
// swapcontext() et al. Use GNU Pth workalike functions.
#if defined(__APPLE__) && (__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 1050)

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "pth/pth.h"

namespace hpx { namespace threads { namespace coroutines { namespace detail {
namespace posix { namespace pth
{
    inline int check(int rc)
    {
        // The makecontext() functions return zero for success, nonzero for
        // error. The Pth library returns TRUE for success, FALSE for error,
        // with errno set to the nonzero error in the latter case. Map the Pth
        // returns to ucontext style.
        return rc? 0 : errno;
    }
}}
}}}}

#define HPX_COROUTINE_POSIX_IMPL "Pth implementation"
#define HPX_COROUTINE_DECLARE_CONTEXT(name) pth_uctx_t name
#define HPX_COROUTINE_CREATE_CONTEXT(ctx)                                     \
    hpx::threads::coroutines::detail::posix::pth::check(pth_uctx_create(&(ctx)))
#define HPX_COROUTINE_MAKE_CONTEXT(ctx, stack, size, startfunc, startarg, exitto) \
    /* const sigset_t* sigmask = nullptr: we don't expect per-context signal masks */ \
    hpx::threads::coroutines::detail::posix::pth::check(                      \
        pth_uctx_make(*(ctx), static_cast<char*>(stack), (size), nullptr,        \
        (startfunc), (startarg), (exitto)))
#define HPX_COROUTINE_SWAP_CONTEXT(from, to)                                  \
    hpx::threads::coroutines::detail::posix::pth::check(pth_uctx_switch(*(from), *(to)))
#define HPX_COROUTINE_DESTROY_CONTEXT(ctx)                                    \
    hpx::threads::coroutines::detail::posix::pth::check(pth_uctx_destroy(ctx))

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
#include <cstddef>                  // ptrdiff_t
#include <ucontext.h>

#include <signal.h>
#include <stdlib.h>
#include <strings.h>

#ifndef SEGV_STACK_SIZE
  #define SEGV_STACK_SIZE MINSIGSTKSZ+4096
#endif

#include <iostream>
#include <iomanip>

namespace hpx { namespace threads { namespace coroutines { namespace detail {
namespace posix { namespace ucontext
{
    inline int make_context(::ucontext_t* ctx, void* stack, std::ptrdiff_t size,
                            void (*startfunc)(void*), void* startarg,
                            ::ucontext_t* exitto = nullptr)
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
}}
}}}}

#define HPX_COROUTINE_POSIX_IMPL "ucontext implementation"
#define HPX_COROUTINE_DECLARE_CONTEXT(name) ::ucontext_t name
#define HPX_COROUTINE_CREATE_CONTEXT(ctx) /* nop */
#define HPX_COROUTINE_MAKE_CONTEXT(ctx, stack, size, startfunc, startarg, exitto) \
    hpx::threads::coroutines::detail::posix::ucontext::make_context(          \
        ctx, stack, size, startfunc, startarg, exitto)
#define HPX_COROUTINE_SWAP_CONTEXT(pfrom, pto) ::swapcontext((pfrom), (pto))
#define HPX_COROUTINE_DESTROY_CONTEXT(ctx) /* nop */

#endif // generic Posix platform

#include <hpx/runtime/threads/coroutines/detail/get_stack_pointer.hpp>
#include <hpx/runtime/threads/coroutines/detail/posix_utility.hpp>
#include <hpx/runtime/threads/coroutines/detail/swap_context.hpp>
#include <hpx/runtime/threads/coroutines/exception.hpp>
#include <signal.h>                 // SIGSTKSZ

namespace hpx { namespace threads { namespace coroutines
{
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
            friend void swap_context(ucontext_context_impl_base& from,
                const ucontext_context_impl_base& to, default_hint)
            {
                int  error = HPX_COROUTINE_SWAP_CONTEXT(&from.m_ctx, &to.m_ctx);
                HPX_UNUSED(error);
                HPX_ASSERT(error == 0);
            }

        protected:
            HPX_COROUTINE_DECLARE_CONTEXT(m_ctx);
        };

        class ucontext_context_impl
          : public ucontext_context_impl_base
        {
        public:
            HPX_NON_COPYABLE(ucontext_context_impl);

        public:
            typedef ucontext_context_impl_base context_impl_base;

            enum { default_stack_size = SIGSTKSZ };

            /**
             * Create a context that on restore invokes Functor on
             *  a new stack. The stack size can be optionally specified.
             */
            template<typename Functor>
            explicit ucontext_context_impl(Functor & cb, std::ptrdiff_t stack_size)
              : m_stack_size(stack_size == -1 ? (std::ptrdiff_t)default_stack_size
                    : stack_size),
                m_stack(alloc_stack(m_stack_size)),
                cb_(&cb)
            {
                HPX_ASSERT(m_stack);
                funp_ = &trampoline<Functor>;
                int error = HPX_COROUTINE_MAKE_CONTEXT(
                    &m_ctx, m_stack, m_stack_size, funp_, cb_, nullptr);
                HPX_UNUSED(error);
                HPX_ASSERT(error == 0);

                segv_stack.ss_sp = valloc(SEGV_STACK_SIZE);
                segv_stack.ss_flags = 0;
                segv_stack.ss_size = SEGV_STACK_SIZE;

                bzero(&action, sizeof(action));
                action.sa_flags = SA_SIGINFO|SA_ONSTACK; //SA_STACK
                action.sa_sigaction = &sigsegv_handler;

                sigaltstack(&segv_stack, NULL);
                sigfillset(&action.sa_mask);
                sigaction(SIGSEGV, &action, NULL);
            }

            static void sigsegv_handler(int signum, siginfo_t *info,
                void *data)
            {
                void *addr = info->si_addr;

                std::cerr << "Stack overflow in coroutine at address "
                    << std::internal << std::hex << std::setw(sizeof(addr)*2+2)
                    << std::setfill('0') << static_cast<int*>(addr) << "." std::endl
                    << "Configure the hpx runtime to allocate a larger "
                    << "coroutine stack size." << std::endl
                    << "Use the hpx.stacks.small_size, hpx.stacks.medium_size, " << std::endl;
                    << "hpx.stacks.large_size, or hpx.stacks.huge_size runtime " << std::endl;
                    << "flags to configure coroutine heap sizes." << std::endl;

                std::exit(EXIT_FAILURE);
            }

            ~ucontext_context_impl()
            {
                if (m_stack)
                    free_stack(m_stack, m_stack_size);
            }

            // Return the size of the reserved stack address space.
            std::ptrdiff_t get_stacksize() const
            {
                return m_stack_size;
            }

            std::ptrdiff_t get_available_stack_space()
            {
#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
                return get_stack_ptr() - reinterpret_cast<std::size_t>(m_stack);
#else
                return (std::numeric_limits<std::ptrdiff_t>::max)();
#endif
            }

            // global functions to be called for each OS-thread after it started
            // running and before it exits
            static void thread_startup(char const* thread_type) {}
            static void thread_shutdown() {}

            void reset_stack()
            {
                if (m_stack)
                {
                    if (posix::reset_stack(
                        m_stack, static_cast<std::size_t>(m_stack_size)))
                        increment_stack_unbind_count();
                }
            }

            void rebind_stack()
            {
                if (m_stack)
                {
                    // just reset the context stack pointer to its initial value at
                    // the stack start
                    increment_stack_recycle_count();
                    int error = HPX_COROUTINE_MAKE_CONTEXT(
                        &m_ctx, m_stack, m_stack_size, funp_, cb_, nullptr);
                    HPX_UNUSED(error);
                    HPX_ASSERT(error == 0);
                }
            }


            typedef boost::atomic<std::int64_t> counter_type;

            static counter_type& get_stack_unbind_counter()
            {
                static counter_type counter(0);
                return counter;
            }

            static std::uint64_t get_stack_unbind_count(bool reset)
            {
                return util::get_and_reset_value(get_stack_unbind_counter(), reset);
            }

            static std::uint64_t increment_stack_unbind_count()
            {
                return ++get_stack_unbind_counter();
            }

            static counter_type& get_stack_recycle_counter()
            {
                static counter_type counter(0);
                return counter;
            }

            static std::uint64_t get_stack_recycle_count(bool reset)
            {
                return util::get_and_reset_value(get_stack_recycle_counter(), reset);
            }

            static std::uint64_t increment_stack_recycle_count()
            {
                return ++get_stack_recycle_counter();
            }

        private:
            // declare m_stack_size first so we can use it to initialize m_stack
            std::ptrdiff_t m_stack_size;
            void * m_stack;
            void * cb_;
            void(*funp_)(void*);

            struct sigaction action;
            stack_t segv_stack;
        };

        typedef ucontext_context_impl context_impl;
    }}
}}}

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

#endif /*HPX_RUNTIME_THREADS_COROUTINES_DETAIL_CONTEXT_POSIX_HPP*/
