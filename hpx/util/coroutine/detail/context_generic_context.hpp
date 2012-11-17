//  Copyright (c) 2012 Hartmut Kaiser
//  Copyright (c) 2009 Oliver Kowalke
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

#include <hpx/config.hpp>
#include <hpx/config/forceinline.hpp>
#include <hpx/util/coroutine/exception.hpp>
#include <hpx/util/coroutine/detail/swap_context.hpp>

#include <cstddef>
#include <cstdlib>
#include <stdexcept>

namespace hpx { namespace util { namespace coroutines 
{
    template <std::size_t Max, std::size_t Default, std::size_t Min>
    class simple_stack_allocator
    {
    public:
        static std::size_t maximum_stacksize()
        { return Max; }

        static std::size_t default_stacksize()
        { return Default; }

        static std::size_t minimum_stacksize()
        { return Min; }

        void * allocate(std::size_t size) const
        {
            BOOST_ASSERT(minimum_stacksize() <= size);
            BOOST_ASSERT(maximum_stacksize() >= size);

            void * limit = std::calloc(size, sizeof(char) );
            if (! limit) throw std::bad_alloc();

            return static_cast< char * >(limit) + size;
        }

        void deallocate(void * vp, std::size_t size) const
        {
            BOOST_ASSERT(vp);
            BOOST_ASSERT(minimum_stacksize() <= size);
            BOOST_ASSERT(maximum_stacksize() >= size);

            void * limit = static_cast< char * >(vp) - size;
            std::free(limit);
        }
    };

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
        BOOST_FORCEINLINE void trampoline(intptr_t pv) 
        {
            T* fun = reinterpret_cast<T*>(pv);
            BOOST_ASSERT(fun);
            (*fun)();
            std::abort();
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
                std::size_t stack_size_
                    = (stack_size == -1) ? 
                      alloc_.minimum_stacksize() : std::size_t(stack_size);

                void * stack_pointer_
                    = alloc_.allocate(stack_size_);

                void (*fn)(intptr_t) = &trampoline<Functor>;

                boost::context::fcontext_t * ctx = boost::context::make_fcontext(stack_pointer_, stack_size_, fn);

                std::swap(*ctx, ctx_);
            }

            ~fcontext_context_impl()
            {
                if (ctx_.fc_stack.sp) 
                {
                    alloc_.deallocate(ctx_.fc_stack.sp, ctx_.fc_stack.size);
                    ctx_.fc_stack.size = 0;
                    ctx_.fc_stack.sp = 0;
                }
            }

            // Return the size of the reserved stack address space.
            std::ptrdiff_t get_stacksize() const
            {
                return ctx_.fc_stack.size;
            }

            // global functions to be called for each OS-thread after it started
            // running and before it exits
            static void thread_startup(char const* thread_type) {}
            static void thread_shutdown() {}

            // handle stack operations
            void reset_stack()
            {
            }
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
                boost::context::jump_fcontext(&from.ctx_, &to.ctx_, to.cb_, false);
            }

        private:
            boost::context::fcontext_t ctx_;
            simple_stack_allocator<
                HPX_HUGE_STACK_SIZE, HPX_MEDIUM_STACK_SIZE, HPX_SMALL_STACK_SIZE
            > alloc_;
            intptr_t cb_;
        };

        typedef fcontext_context_impl context_impl;
    }}
}}}

#endif
