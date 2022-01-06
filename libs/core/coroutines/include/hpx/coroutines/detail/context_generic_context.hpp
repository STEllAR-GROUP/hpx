//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2012 Hartmut Kaiser
//  Copyright (c) 2009 Oliver Kowalke
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/coroutines/detail/get_stack_pointer.hpp>
#include <hpx/coroutines/detail/swap_context.hpp>
#include <hpx/util/get_and_reset_value.hpp>

// include unist.d conditionally to check for POSIX version. Not all OSs have the
// unistd header...
#if defined(HPX_HAVE_UNISTD_H)
#include <unistd.h>
#endif

#if defined(_POSIX_VERSION)
#include <hpx/coroutines/detail/posix_utility.hpp>
#endif

#include <boost/context/detail/fcontext.hpp>
#include <boost/version.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_GENERIC_CONTEXT_USE_SEGMENTED_STACKS)

#define HPX_COROUTINES_SEGMENTS 10

extern "C" {
void* __splitstack_makecontext(
    std::size_t, void* [HPX_COROUTINES_SEGMENTS], std::size_t*);
void __splitstack_releasecontext(void* [HPX_COROUTINES_SEGMENTS]);
void __splitstack_resetcontext(void* [HPX_COROUTINES_SEGMENTS]);
void __splitstack_block_signals_context(
    void* [HPX_COROUTINES_SEGMENTS], int* new_value, int* old_value);
void __splitstack_getcontext(void* [HPX_COROUTINES_SEGMENTS]);
void __splitstack_setcontext(void* [HPX_COROUTINES_SEGMENTS]);
}

#if !defined(SIGSTKSZ)
#define SIGSTKSZ (8 * 1024)
#define UDEF_SIGSTKSZ
#endif

#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace coroutines {
    // some platforms need special preparation of the main thread
    struct prepare_main_thread
    {
        constexpr prepare_main_thread() {}
    };

    namespace detail { namespace generic_context {
        ///////////////////////////////////////////////////////////////////////
        // This is taken directly from one of the Boost.Context examples
#if !defined(HPX_GENERIC_CONTEXT_USE_SEGMENTED_STACKS)
        struct stack_allocator
        {
            static std::size_t maximum_stacksize()
            {
                return HPX_HUGE_STACK_SIZE;
            }

            static std::size_t default_stacksize()
            {
                return HPX_MEDIUM_STACK_SIZE;
            }

            static std::size_t minimum_stacksize()
            {
                return HPX_SMALL_STACK_SIZE;
            }

            void* allocate(std::size_t size) const
            {
#if defined(_POSIX_VERSION)
                void* limit = posix::alloc_stack(size);
                posix::watermark_stack(limit, size);
#else
                void* limit = std::calloc(size, sizeof(char));
                if (!limit)
                    throw std::bad_alloc();
#endif
                return static_cast<char*>(limit) + size;
            }

            void deallocate(void* vp, std::size_t size) const
            {
                HPX_ASSERT(vp);
                void* limit = static_cast<char*>(vp) - size;
#if defined(_POSIX_VERSION)
                posix::free_stack(limit, size);
#else
                std::free(limit);
#endif
            }
        };
#else
        struct stack_allocator
        {
            typedef void* segments_context[HPX_COROUTINES_SEGMENTS];

            static std::size_t maximum_stacksize()
            {
                HPX_ASSERT_MSG(false, "segmented stack is unbound");
                return 0;
            }

            static std::size_t default_stacksize()
            {
                return minimum_stacksize();
            }

            static std::size_t minimum_stacksize()
            {
                return SIGSTKSZ + sizeof(boost::context::detail::fcontext_t) +
                    15;
            }

            void* allocate(std::size_t size) const
            {
                HPX_ASSERT(default_stacksize() <= size);

                void* limit =
                    __splitstack_makecontext(size, segments_ctx_, &size);
                if (!limit)
                    throw std::bad_alloc();

                int off = 0;
                __splitstack_block_signals_context(segments_ctx_, &off, 0);

                return static_cast<char*>(limit) + size;
            }

            void deallocate(void* vp, std::size_t size) const
            {
                __splitstack_releasecontext(segments_ctx_);
            }

            segments_context segments_ctx_;
        };
#endif

        // Generic implementation for the context_impl_base class based on
        // Boost.Context.
        template <typename T>
        HPX_FORCEINLINE void trampoline(boost::context::detail::transfer_t tr)
        {
            auto arg = reinterpret_cast<
                std::pair<void*, boost::context::detail::fcontext_t*>*>(
                tr.data);

            HPX_ASSERT(arg->second);
            *arg->second = tr.fctx;

            T* fun = reinterpret_cast<T*>(arg->first);
            HPX_ASSERT(fun);
            (*fun)();

            std::abort();
        }

        template <typename CoroutineImpl>
        class fcontext_context_impl
        {
        public:
            HPX_NON_COPYABLE(fcontext_context_impl);

        public:
            typedef fcontext_context_impl context_impl_base;

            // Create a context that on restore invokes Functor on
            // a new stack. The stack size can be optionally specified.
            explicit fcontext_context_impl(std::ptrdiff_t stack_size = -1)
              : cb_(std::make_pair(reinterpret_cast<void*>(this), nullptr))
              , funp_(&trampoline<CoroutineImpl>)
              , ctx_(0)
              , alloc_()
              , stack_size_((stack_size == -1) ? alloc_.minimum_stacksize() :
                                                 std::size_t(stack_size))
              , stack_pointer_(nullptr)
            {
            }

            void init()
            {
                if (stack_pointer_ != nullptr)
                    return;

                stack_pointer_ = alloc_.allocate(stack_size_);
                if (stack_pointer_ == nullptr)
                {
                    throw std::runtime_error(
                        "could not allocate memory for stack");
                }
                ctx_ = boost::context::detail::make_fcontext(
                    stack_pointer_, stack_size_, funp_);
            }

            ~fcontext_context_impl()
            {
                if (ctx_ && stack_pointer_)
                {
                    alloc_.deallocate(stack_pointer_, stack_size_);
                }
            }

            // Return the size of the reserved stack address space.
            std::ptrdiff_t get_stacksize() const
            {
                return stack_size_;
            }

            std::ptrdiff_t get_available_stack_space()
            {
#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
                return stack_size_ -
                    (reinterpret_cast<std::size_t>(stack_pointer_) -
                        get_stack_ptr());
#else
                return (std::numeric_limits<std::ptrdiff_t>::max)();
#endif
            }

            void reset_stack()
            {
                if (ctx_)
                {
#if defined(_POSIX_VERSION)
                    void* limit =
                        static_cast<char*>(stack_pointer_) - stack_size_;
                    if (posix::reset_stack(limit, stack_size_))
                    {
#if defined(HPX_HAVE_COROUTINE_COUNTERS)
                        increment_stack_unbind_count();
#endif
                    }
#else
                    // nothing we can do here ...
#endif
                }
            }

            void rebind_stack()
            {
                if (ctx_)
                {
#if defined(HPX_HAVE_COROUTINE_COUNTERS)
                    increment_stack_recycle_count();
#endif
                    ctx_ = boost::context::detail::make_fcontext(
                        stack_pointer_, stack_size_, funp_);
                }
            }

#if defined(HPX_HAVE_COROUTINE_COUNTERS)
            typedef std::atomic<std::int64_t> counter_type;

        private:
            static counter_type& get_stack_unbind_counter()
            {
                static counter_type counter(0);
                return counter;
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

            static std::uint64_t increment_stack_recycle_count()
            {
                return ++get_stack_recycle_counter();
            }

        public:
            static std::uint64_t get_stack_unbind_count(bool reset)
            {
                return util::get_and_reset_value(
                    get_stack_unbind_counter(), reset);
            }

            static std::uint64_t get_stack_recycle_count(bool reset)
            {
                return util::get_and_reset_value(
                    get_stack_recycle_counter(), reset);
            }
#endif

        private:
            friend void swap_context(fcontext_context_impl& from,
                fcontext_context_impl& to, detail::default_hint)
            {
#if defined(HPX_GENERIC_CONTEXT_USE_SEGMENTED_STACKS)
                __splitstack_getcontext(from.alloc_.segments_ctx_);
                __splitstack_setcontext(to.alloc_.segments_ctx);
#endif
                // switch to other coroutine context
                to.cb_.second = &from.ctx_;
                auto transfer = boost::context::detail::jump_fcontext(
                    to.ctx_, reinterpret_cast<void*>(&to.cb_));
                to.ctx_ = transfer.fctx;

#if defined(HPX_GENERIC_CONTEXT_USE_SEGMENTED_STACKS)
                __splitstack_setcontext(from.alloc_.segments_ctx);
#endif
            }

        private:
            std::pair<void*, boost::context::detail::fcontext_t*> cb_;
            void (*funp_)(boost::context::detail::transfer_t);
            boost::context::detail::fcontext_t ctx_;
            stack_allocator alloc_;
            std::size_t stack_size_;
            void* stack_pointer_;
        };
    }}    // namespace detail::generic_context
}}}       // namespace hpx::threads::coroutines
