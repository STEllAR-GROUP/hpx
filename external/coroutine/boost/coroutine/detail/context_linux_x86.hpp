//  Copyright (c) 2006, Giovanni P. Deretta
//  Copyright (c) 2007 Robert Perricone
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_COROUTINE_CONTEXT_LINUX_HPP_20071126
#define BOOST_COROUTINE_CONTEXT_LINUX_HPP_20071126

#if defined(__linux) || defined(linux) || defined(__linux__)

#if !defined(BOOST_COROUTINE_USE_ATOMIC_COUNT)
#  define BOOST_COROUTINE_USE_ATOMIC_COUNT
#endif

#ifdef BOOST_COROUTINE_USE_ATOMIC_COUNT
#  include <boost/atomic.hpp>
#endif

#include <sys/param.h>
#include <cstdlib>
#include <cstddef>
#include <boost/cstdint.hpp>
#include <boost/coroutine/detail/config.hpp>
#include <boost/coroutine/detail/posix_utility.hpp>
#include <boost/coroutine/detail/swap_context.hpp>
#include <boost/coroutine/detail/static.hpp>
#include <boost/assert.hpp>

/*
 * Defining BOOST_COROUTINE_NO_SEPARATE_CALL_SITES will disable separate
 * invoke, yield and yield_to swap_context functions. Separate calls sites
 * increase performance by 25% at least on P4 for invoke+yield back loops
 * at the cost of a slightly higher instruction cache use and is thus enabled by
 * default.
 */

#if defined(__x86_64__)
extern "C" void swapcontext_stack (void***, void**) throw();
extern "C" void swapcontext_stack2 (void***, void**) throw();
extern "C" void swapcontext_stack3 (void***, void**) throw();
#else
extern "C" void swapcontext_stack (void***, void**) throw() __attribute((regparm(2)));
extern "C" void swapcontext_stack2 (void***, void**) throw()__attribute((regparm(2)));
extern "C" void swapcontext_stack3 (void***, void**) throw()__attribute((regparm(2)));
#endif

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace coroutines {
    
    namespace very_detail
    {
        template <typename TO, typename FROM> 
        TO nasty_cast(FROM f)
        {
            union {
                FROM f; TO t;
            } u;
            u.f = f;
            return u.t;
        }
    }

  // some platforms need special preparation of the main thread
  struct prepare_main_thread
  {
      prepare_main_thread() {}
      ~prepare_main_thread() {}
  };

  namespace detail { namespace lx
  {
    template<typename T>
    void trampoline(T* fun);

    template<typename T>
    inline void
    trampoline(T* fun)
    {
      (*fun)();
      std::abort();
    }

    class x86_linux_context_impl;

    class x86_linux_context_impl_base : detail::context_impl_base
    {
    public:
      x86_linux_context_impl_base() {}

      void prefetch() const
      {
#if defined(__x86_64__)
        BOOST_ASSERT(sizeof(void*) == 8);
#else
        BOOST_ASSERT(sizeof(void*) == 4);
#endif

        __builtin_prefetch (m_sp, 1, 3);
        __builtin_prefetch (m_sp, 0, 3);
        __builtin_prefetch (static_cast<void**>(m_sp)+64/sizeof(void*), 1, 3);
        __builtin_prefetch (static_cast<void**>(m_sp)+64/sizeof(void*), 0, 3);
        __builtin_prefetch (static_cast<void**>(m_sp)+32/sizeof(void*), 1, 3);
        __builtin_prefetch (static_cast<void**>(m_sp)+32/sizeof(void*), 0, 3);
        __builtin_prefetch (static_cast<void**>(m_sp)-32/sizeof(void*), 1, 3);
        __builtin_prefetch (static_cast<void**>(m_sp)-32/sizeof(void*), 0, 3);
        __builtin_prefetch (static_cast<void**>(m_sp)-64/sizeof(void*), 1, 3);
        __builtin_prefetch (static_cast<void**>(m_sp)-64/sizeof(void*), 0, 3);
      }

      /**
       * Free function. Saves the current context in @p from
       * and restores the context in @p to.
       * @note This function is found by ADL.
       */
      friend void swap_context(x86_linux_context_impl_base& from,
          x86_linux_context_impl const& to, default_hint);

      friend void swap_context(x86_linux_context_impl& from,
          x86_linux_context_impl_base const& to, yield_hint);

      friend void swap_context(x86_linux_context_impl& from,
          x86_linux_context_impl_base const& to, yield_to_hint);

    protected:
      void ** m_sp;
    };

    class x86_linux_context_impl : public x86_linux_context_impl_base
    {
    public:
      enum { default_stack_size = 4*EXEC_PAGESIZE };

      typedef x86_linux_context_impl_base context_impl_base;

      x86_linux_context_impl()
        : m_stack(0)
      {}

      /**
       * Create a context that on restore invokes Functor on
       *  a new stack. The stack size can be optionally specified.
       */
      template<typename Functor>
      x86_linux_context_impl(Functor& cb, std::ptrdiff_t stack_size = -1)
        : m_stack_size(stack_size == -1
                      ? static_cast<std::ptrdiff_t>(default_stack_size)
                      : stack_size),
          m_stack(0)
      {
        BOOST_ASSERT(0 == (m_stack_size % EXEC_PAGESIZE));
        BOOST_ASSERT(m_stack_size >= 0);
        m_stack = posix::alloc_stack(static_cast<std::size_t>(m_stack_size));
        BOOST_ASSERT(m_stack);
        m_sp = (static_cast<void**>(m_stack) + static_cast<std::size_t>(m_stack_size)/sizeof(void*));

        posix::watermark_stack(m_stack, static_cast<std::size_t>(m_stack_size));

        typedef void fun(Functor*);
        fun * funp = trampoline;

#if defined(__x86_64__)
        // we have to make sure that the stack pointer is aligned on a 16 Byte
        // boundary when the code is entering the trampoline (the stack itself
        // is already properly aligned)
        *--m_sp = 0;       // additional alignment

        *--m_sp = &cb;     // parm 0 of trampoline;
        *--m_sp = 0;       // dummy return address for trampoline
        *--m_sp = very_detail::nasty_cast<void*>( funp );// return addr (here: start addr)
        *--m_sp = 0;       // rbp
        *--m_sp = 0;       // rbx
        *--m_sp = 0;       // rsi
        *--m_sp = 0;       // rdi
        *--m_sp = 0;       // r12
        *--m_sp = 0;       // r13
        *--m_sp = 0;       // r14
        *--m_sp = 0;       // r15
#else
        *--m_sp = &cb;     // parm 0 of trampoline;
        *--m_sp = 0;       // dummy return address for trampoline
        *--m_sp = very_detail::nasty_cast<void*>( funp );// return addr (here: start addr)
        *--m_sp = 0;       // ebp
        *--m_sp = 0;       // ebx
        *--m_sp = 0;       // esi
        *--m_sp = 0;       // edi
#endif
      }

      ~x86_linux_context_impl()
      {
        if(m_stack)
          posix::free_stack(m_stack, static_cast<std::size_t>(m_stack_size));
      }

      void reset_stack()
      {
        if(m_stack) {
          if(posix::reset_stack(m_stack, static_cast<std::size_t>(m_stack_size)))
            increment_stack_unbind_count();
        }
      }

      void rebind_stack()
      {
        if(m_stack) 
          increment_stack_recycle_count();
      }

#ifndef BOOST_COROUTINE_USE_ATOMIC_COUNT
      typedef std::size_t counter_type;
#else
      typedef boost::atomic_uint64_t counter_type;
#endif

      static counter_type& get_stack_unbind_counter()
      {
          static counter_type counter(0);
          return counter;
      }
      static boost::uint64_t get_stack_unbind_count()
      {
          return get_stack_unbind_counter();
      }
      static boost::uint64_t increment_stack_unbind_count()
      {
          return ++get_stack_unbind_counter();
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

      friend void swap_context(x86_linux_context_impl_base& from,
          x86_linux_context_impl const& to, default_hint);

      friend void swap_context(x86_linux_context_impl& from,
          x86_linux_context_impl_base const& to, yield_hint);

      friend void swap_context(x86_linux_context_impl& from,
          x86_linux_context_impl_base const& to, yield_to_hint);

      // global functions to be called for each OS-thread after it started
      // running and before it exits
      static void thread_startup(char const* thread_type)
      {
      }

      static void thread_shutdown()
      {
      }

    private:
      std::ptrdiff_t m_stack_size;
      void * m_stack;
    };

    typedef x86_linux_context_impl context_impl;

    /**
     * Free function. Saves the current context in @p from
     * and restores the context in @p to.
     * @note This function is found by ADL.
     */
    inline void swap_context(x86_linux_context_impl_base& from,
        x86_linux_context_impl const& to, default_hint)
    {
//        BOOST_ASSERT(*(void**)to.m_stack == (void*)~0);
        to.prefetch();
        swapcontext_stack(&from.m_sp, to.m_sp);
    }

    inline void swap_context(x86_linux_context_impl& from,
        x86_linux_context_impl_base const& to, yield_hint)
    {
//        BOOST_ASSERT(*(void**)from.m_stack == (void*)~0);
        to.prefetch();
#ifndef BOOST_COROUTINE_NO_SEPARATE_CALL_SITES
        swapcontext_stack2(&from.m_sp, to.m_sp);
#else
        swapcontext_stack(&from.m_sp, to.m_sp);
#endif
    }

    inline void swap_context(x86_linux_context_impl& from,
        x86_linux_context_impl_base const& to, yield_to_hint)
    {
//        BOOST_ASSERT(*(void**)from.m_stack == (void*)~0);
        to.prefetch();
#ifndef BOOST_COROUTINE_NO_SEPARATE_CALL_SITES
        swapcontext_stack3(&from.m_sp, to.m_sp);
#else
        swapcontext_stack(&from.m_sp, to.m_sp);
#endif
    }

  }
}}}

#else

#error This header can only be included when compiling for linux systems.

#endif

#endif

