//  Copyright (c) 2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COROUTINE_CONTEXT_GENERIC_SEP_01_2012_0519PM)
#define HPX_COROUTINE_CONTEXT_GENERIC_SEP_01_2012_0519PM

#include <boost/version.hpp>

#if BOOST_VERSION < 105100
#error Boost.Context is available only with Boost V1.51 or later
#endif

#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/detail/atomic_count.hpp>

#include <boost/context/all.hpp>
#include <boost/noncopyable.hpp>

#include <hpx/util/coroutine/exception.hpp>
#include <hpx/util/coroutine/detail/swap_context.hpp>

namespace hpx { namespace util { namespace coroutines 
{
    // some platforms need special preparation of the main thread
    struct prepare_main_thread
    {
        prepare_main_thread() {}
        ~prepare_main_thread() {}
    };

    namespace detail { namespace generic_context
    {
        // Generic implementation for the context_impl_base class based on 
        // Boost.Context.
        template <typename T>
        inline void trampoline(intptr_t pv) 
        {
            T* fun = reinterpret_cast<T*>(pv);
            BOOST_ASSERT(fun);
            (*fun)();
        }

        class fcontext_context_impl
          : private boost::noncopyable
        {
        public:
            typedef fcontext_context_impl context_impl_base;

            fcontext_context_impl()
              : cb_(0)
            {}

            // Create a context that on restore invokes Functor on
            // a new stack. The stack size can be optionally specified.
            template <typename Functor>
            explicit fcontext_context_impl(Functor& cb, std::ptrdiff_t stack_size)
              : cb_(reinterpret_cast<intptr_t>(&cb))
            {
#if defined(BOOST_WINDOWS)
//              Boost.Context on Windows currently enforces a minimum stack 
//              size of 64k, there is no way to specify a smaller stack size.
//              This will be adjusted after Boost.Context has been fixed.
                ctx_.fc_stack.size = boost::ctx::minimum_stacksize();
#else
                ctx_.fc_stack.size = (stack_size == -1) ? 
                    boost::ctx::minimum_stacksize() : std::size_t(stack_size);
#endif
                ctx_.fc_stack.sp = alloc_.allocate(ctx_.fc_stack.size);
                boost::ctx::make_fcontext(&ctx_, &trampoline<Functor>);
            }

            ~fcontext_context_impl()
            {
                if (ctx_.fc_stack.size) 
                {
                    alloc_.deallocate(ctx_.fc_stack.sp, ctx_.fc_stack.size);
                    ctx_.fc_stack.size = 0;
                }
            }

            // global functions to be called for each OS-thread after it started
            // running and before it exits
            static void thread_startup(char const* thread_type) {}
            static void thread_shutdown() {}

            // handle stack operations
            void reset_stack() {}
            void rebind_stack() 
            {
                if (ctx_.fc_stack.sp) 
                    increment_stack_recycle_count();
            }

            typedef boost::detail::atomic_count counter_type;

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
            friend void swap_context(fcontext_context_impl& from, 
                fcontext_context_impl const& to, detail::default_hint)
            {
                boost::ctx::jump_fcontext(&from.ctx_, &to.ctx_, to.cb_, false);
            }

        private:
            boost::ctx::fcontext_t ctx_;
            boost::ctx::stack_allocator alloc_;
            intptr_t cb_;
        };

        typedef fcontext_context_impl context_impl;
    }}
}}}

#endif
