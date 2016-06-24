//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2012 Hartmut Kaiser
//  Copyright (c) 2009 Oliver Kowalke
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_COROUTINES_DETAIL_CONTEXT_GENERIC_HPP
#define HPX_RUNTIME_THREADS_COROUTINES_DETAIL_CONTEXT_GENERIC_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/threads/coroutines/detail/get_stack_pointer.hpp>
#include <hpx/runtime/threads/coroutines/detail/swap_context.hpp>
#include <hpx/runtime/threads/coroutines/exception.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/get_and_reset_value.hpp>

// include unist.d conditionally to check for POSIX version. Not all OSs have the
// unistd header...
#if defined(HPX_HAVE_UNISTD_H)
#include <unistd.h>
#endif

#if defined(_POSIX_VERSION)
#include <hpx/runtime/threads/coroutines/detail/posix_utility.hpp>
#endif

#include <boost/cstdint.hpp>
#include <boost/version.hpp>

#if BOOST_VERSION < 105100
#error Boost.Context is available only with Boost V1.51 or later
#endif

#include <boost/atomic.hpp>

#include <boost/context/all.hpp>

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_GENERIC_CONTEXT_USE_SEGMENTED_STACKS) && BOOST_VERSION >= 105300

#define HPX_COROUTINES_SEGMENTS 10

extern "C"
{
    void *__splitstack_makecontext(std::size_t,
        void *[HPX_COROUTINES_SEGMENTS], std::size_t *);
    void __splitstack_releasecontext(void *[HPX_COROUTINES_SEGMENTS]);
    void __splitstack_resetcontext(void *[HPX_COROUTINES_SEGMENTS]);
    void __splitstack_block_signals_context(void *[HPX_COROUTINES_SEGMENTS],
        int * new_value, int * old_value);
    void __splitstack_getcontext(void * [HPX_COROUTINES_SEGMENTS]);
    void __splitstack_setcontext(void * [HPX_COROUTINES_SEGMENTS]);
}

#if !defined (SIGSTKSZ)
# define SIGSTKSZ (8 * 1024)
# define UDEF_SIGSTKSZ
#endif

#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace coroutines
{
    // some platforms need special preparation of the main thread
    struct prepare_main_thread
    {
        prepare_main_thread() {}
        ~prepare_main_thread() {}
    };

    namespace detail { namespace generic_context
    {
        ///////////////////////////////////////////////////////////////////////
        // This is taken directly from one of the Boost.Context examples
#if !defined(HPX_GENERIC_CONTEXT_USE_SEGMENTED_STACKS)
        struct stack_allocator
        {
            static std::size_t maximum_stacksize()
            { return HPX_HUGE_STACK_SIZE; }

            static std::size_t default_stacksize()
            { return HPX_MEDIUM_STACK_SIZE; }

            static std::size_t minimum_stacksize()
            { return HPX_SMALL_STACK_SIZE; }

            void* allocate(std::size_t size) const
            {
#if defined(_POSIX_VERSION)
                void* limit = posix::alloc_stack(size);
                posix::watermark_stack(limit, size);
#else
                void* limit = std::calloc(size, sizeof(char));
                if (!limit) boost::throw_exception(std::bad_alloc());

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
            typedef void *segments_context[HPX_COROUTINES_SEGMENTS];

            static std::size_t maximum_stacksize()
            {
                HPX_ASSERT_MSG( false, "segmented stack is unbound");
                return 0;
            }

            static std::size_t default_stacksize()
            { return minimum_stacksize(); }

            static std::size_t minimum_stacksize()
            { return SIGSTKSZ + sizeof(boost::context::fcontext_t) + 15; }

            void* allocate(std::size_t size) const
            {
                HPX_ASSERT(default_stacksize() <= size);

                void* limit = __splitstack_makecontext(size, segments_ctx_, &size);
                if (!limit) boost::throw_exception(std::bad_alloc());

                int off = 0;
                 __splitstack_block_signals_context(segments_ctx_, &off, 0);

                return static_cast<char *>(limit) + size;
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
        HPX_FORCEINLINE void trampoline(intptr_t pv)
        {
            T* fun = reinterpret_cast<T*>(pv);
            HPX_ASSERT(fun);
            (*fun)();
            std::abort();
        }

        class fcontext_context_impl
        {
            HPX_NON_COPYABLE(fcontext_context_impl);

        public:
            typedef fcontext_context_impl context_impl_base;

            fcontext_context_impl()
              : cb_(0)
              , funp_(0)
#if BOOST_VERSION > 105500
              , ctx_(0)
#endif
              , alloc_()
              , stack_size_(0)
              , stack_pointer_(0)
            {}

            // Create a context that on restore invokes Functor on
            // a new stack. The stack size can be optionally specified.
            template <typename Functor>
            fcontext_context_impl(Functor& cb, std::ptrdiff_t stack_size)
              : cb_(reinterpret_cast<intptr_t>(&cb))
              , funp_(&trampoline<Functor>)
#if BOOST_VERSION > 105500
              , ctx_(0)
#endif
              , alloc_()
              , stack_size_(
                    (stack_size == -1) ?
                    alloc_.minimum_stacksize() : std::size_t(stack_size)
                )
              , stack_pointer_(alloc_.allocate(stack_size_))
            {
#if BOOST_VERSION < 105600
                boost::context::fcontext_t* ctx =
                    boost::context::make_fcontext(stack_pointer_, stack_size_, funp_);

                std::swap(*ctx, ctx_);
#else
                ctx_ =
                    boost::context::make_fcontext(stack_pointer_, stack_size_, funp_);
#endif
            }

            ~fcontext_context_impl()
            {
#if BOOST_VERSION < 105600
                if (ctx_.fc_stack.sp && stack_pointer_)
#else
                if (ctx_ && stack_pointer_)
#endif
                {
                    alloc_.deallocate(stack_pointer_, stack_size_);
#if BOOST_VERSION < 105600
                    ctx_.fc_stack.size = 0;
                    ctx_.fc_stack.sp = 0;
#endif
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
                    (reinterpret_cast<std::size_t>(stack_pointer_) - get_stack_ptr());
#else
                return (std::numeric_limits<std::ptrdiff_t>::max)();
#endif
            }

            // global functions to be called for each OS-thread after it started
            // running and before it exits
            static void thread_startup(char const* thread_type) {}
            static void thread_shutdown() {}

            // handle stack operations
            HPX_EXPORT void reset_stack();
            HPX_EXPORT void rebind_stack();

            typedef boost::atomic<boost::int64_t> counter_type;

            HPX_EXPORT static counter_type& get_stack_unbind_counter();
            HPX_EXPORT static boost::uint64_t get_stack_unbind_count(bool
                reset);
            HPX_EXPORT static boost::uint64_t increment_stack_unbind_count();

            HPX_EXPORT static counter_type& get_stack_recycle_counter();
            HPX_EXPORT static boost::uint64_t get_stack_recycle_count(bool
                reset);
            HPX_EXPORT static boost::uint64_t increment_stack_recycle_count();

        private:
            friend void swap_context(fcontext_context_impl& from,
                fcontext_context_impl const& to, detail::default_hint)
            {
#if defined(HPX_GENERIC_CONTEXT_USE_SEGMENTED_STACKS)
                __splitstack_getcontext(from.alloc_.segments_ctx_);
                __splitstack_setcontext(to.alloc_.segments_ctx);
#endif
                // switch to other coroutine context
#if BOOST_VERSION < 105600
                boost::context::jump_fcontext(&from.ctx_, &to.ctx_, to.cb_, false);
#else
                boost::context::jump_fcontext(&from.ctx_, to.ctx_, to.cb_, false);
#endif

#if defined(HPX_GENERIC_CONTEXT_USE_SEGMENTED_STACKS)
                __splitstack_setcontext(from.alloc_.segments_ctx);
#endif
            }

        private:
            intptr_t cb_;
            void (*funp_)(intptr_t);
            boost::context::fcontext_t ctx_;
            stack_allocator alloc_;
            std::size_t stack_size_;
            void * stack_pointer_;
        };

        typedef fcontext_context_impl context_impl;
    }}
}}}

#endif /*HPX_RUNTIME_THREADS_COROUTINES_DETAIL_CONTEXT_GENERIC_HPP*/
