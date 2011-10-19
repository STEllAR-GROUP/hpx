//  Copyright (c) 2007 Robert Perricone
//  Copyright (c) 2007-2011 Hartmut Kaiser
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
#  include <boost/detail/atomic_count.hpp>
#endif

#include <sys/param.h>
#include <cstdlib>
#include <cstddef>
#include <boost/cstdint.hpp>
#include <boost/coroutine/detail/config.hpp>
#include <boost/coroutine/detail/posix_utility.hpp>
#include <boost/coroutine/detail/swap_context.hpp>
#include <boost/coroutine/detail/static.hpp>

/* 
 * Defining BOOST_COROUTINE_NO_SEPARATE_CALL_SITES will disable separate 
 * invoke, yield and yield_to swap_context functions. Separate calls sites
 * increase performance by 25% at least on P4 for invoke+yield back loops
 * at the cost of a slightly higher instruction cache use and is thus enabled by
 * default.
 */

extern "C" void swapcontext_stack (void***, void**) throw();
extern "C" void swapcontext_stack2 (void***, void**) throw();
extern "C" void swapcontext_stack3 (void***, void**) throw();

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace coroutines { 

  // some platforms need special preparation of the main thread
  struct prepare_main_thread
  {
      prepare_main_thread() {}
      ~prepare_main_thread() {}
  };

  namespace detail { namespace linux 
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
        __builtin_prefetch (m_sp, 1, 3);
        __builtin_prefetch (m_sp, 0, 3);
        __builtin_prefetch ((void**)m_sp+64/sizeof(void*), 1, 3);
        __builtin_prefetch ((void**)m_sp+64/sizeof(void*), 0, 3);
        __builtin_prefetch ((void**)m_sp+32/sizeof(void*), 1, 3);
        __builtin_prefetch ((void**)m_sp+32/sizeof(void*), 0, 3);
        __builtin_prefetch ((void**)m_sp-32/sizeof(void*), 1, 3);
        __builtin_prefetch ((void**)m_sp-32/sizeof(void*), 0, 3);
        __builtin_prefetch ((void**)m_sp-64/sizeof(void*), 1, 3);
        __builtin_prefetch ((void**)m_sp-64/sizeof(void*), 0, 3);
      }

      /**
       * Free function. Saves the current context in @p from
       * and restores the context in @p to.
       * @note This function is found by ADL.
       */     
      friend void swap_context(x86_linux_context_impl_base& from, 
          x86_linux_context_impl const& to, default_hint);

#ifndef BOOST_COROUTINE_NO_SEPARATE_CALL_SITES
      friend void swap_context(x86_linux_context_impl& from, 
          x86_linux_context_impl_base const& to, yield_hint);

      friend void swap_context(x86_linux_context_impl& from, 
          x86_linux_context_impl_base const& to, yield_to_hint);
#endif

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
      // TODO: stack size must be a multiple of EXEC_PAGESIZE. 
      template<typename Functor>
      x86_linux_context_impl(Functor& cb, std::ptrdiff_t stack_size = -1) 
        : m_stack_size(stack_size == -1
                      ? static_cast<std::ptrdiff_t>(default_stack_size)
                      : stack_size)
      {
        BOOST_ASSERT(0 == (m_stack_size % EXEC_PAGESIZE));
        m_stack = posix::alloc_stack(m_stack_size); 
        BOOST_ASSERT(m_stack);
        m_sp = ((void**)m_stack + (m_stack_size/sizeof(void*)));

        posix::watermark_stack(m_stack, m_stack_size);

        typedef void fun(Functor*);
        fun * funp = trampoline;

        // we have to make sure that the stack pointer is aligned on a 16 Byte
        // boundary when the code is entering the trampoline (the stack itself
        // is already properly aligned)
        *--m_sp = 0;       // additional alignment

        *--m_sp = &cb;     // parm 0 of trampoline;
        *--m_sp = 0;       // dummy return address for trampoline
        *--m_sp = (void*) funp ;// return addr (here: start addr)
        *--m_sp = 0;       // rbp
        *--m_sp = 0;       // rbx
        *--m_sp = 0;       // rsi
        *--m_sp = 0;       // rdi
        *--m_sp = 0;       // r12
        *--m_sp = 0;       // r13
        *--m_sp = 0;       // r14
        *--m_sp = 0;       // r15
      }
      
      ~x86_linux_context_impl() 
      {
        if(m_stack)
          posix::free_stack(m_stack, m_stack_size);
      }

      void reset_stack()
      {
        if(m_stack) {
          increment_stack_recycle_count();
          if(posix::reset_stack(m_stack, m_stack_size))
            increment_stack_unbind_count();
        }
      }

      static boost::uint64_t get_stack_unbind_count()
      {
        static_<counter_type, stack_unbind> counter(0);
        return counter.get();
      }
      static boost::uint64_t increment_stack_unbind_count()
      {
        static_<counter_type, stack_unbind> counter(0);
        return counter.get();
      }

      static boost::uint64_t get_stack_recycle_count()
      {
        static_<counter_type, stack_recycle> counter(0);
        return counter.get();
      }
      static boost::uint64_t increment_stack_recycle_count()
      {
        static_<counter_type, stack_recycle> counter(0);
        return ++counter.get();
      }

      friend void swap_context(x86_linux_context_impl_base& from, 
          x86_linux_context_impl const& to, default_hint);

#ifndef BOOST_COROUTINE_NO_SEPARATE_CALL_SITES
      friend void swap_context(x86_linux_context_impl& from, 
          x86_linux_context_impl_base const& to, yield_hint);

      friend void swap_context(x86_linux_context_impl& from, 
          x86_linux_context_impl_base const& to, yield_to_hint);
#endif

#ifndef BOOST_COROUTINE_USE_ATOMIC_COUNT
      typedef std::size_t counter_type;
#else
      typedef boost::detail::atomic_count counter_type;
#endif
    private:
      struct stack_unbind {};
      struct stack_recycle {};

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

#ifndef BOOST_COROUTINE_NO_SEPARATE_CALL_SITES
    inline void swap_context(x86_linux_context_impl& from, 
        x86_linux_context_impl_base const& to, yield_hint) 
    {
//        BOOST_ASSERT(*(void**)from.m_stack == (void*)~0);
        to.prefetch();
        swapcontext_stack2(&from.m_sp, to.m_sp);
    }

    inline void swap_context(x86_linux_context_impl& from, 
        x86_linux_context_impl_base const& to, yield_to_hint) 
    {
//        BOOST_ASSERT(*(void**)from.m_stack == (void*)~0);
        to.prefetch();
        swapcontext_stack3(&from.m_sp, to.m_sp);
    }
#endif

  }
}}}

#else

#error This header can only be included when compiling for linux systems.

#endif

#endif

