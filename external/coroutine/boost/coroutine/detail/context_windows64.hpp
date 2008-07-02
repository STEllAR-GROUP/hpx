//  Copyright (c) 2007 Robert Perricone
//  Copyright (c) 2007 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_COROUTINE_CONTEXT_WINDOWS64_HPP_20071128
#define BOOST_COROUTINE_CONTEXT_WINDOWS64_HPP_20071128

#include <cstdlib>
#include <cstddef>
#include <boost/type_traits.hpp>
#include <boost/coroutine/detail/swap_context.hpp>

/*
 * Defining BOOST_COROUTINE_INLINE_ASM will enable the inline
 * assembler version of swapcontext_stack.
 * The inline asm, with all required clobber flags, is usually no faster
 * than the out-of-line function, and it is not yet clear if
 * it is always reliable (i.e. if the compiler always saves the correct
 * registers). FIXME: it is currently missing at least MMX and XMM registers in
 * the clobber list.
 */
//#define BOOST_COROUTINE_INLINE_ASM
 
/* 
 * Defining BOOST_COROUTINE_NO_SEPARATE_CALL_SITES will disable separate 
 * invoke, yield and yield_to swap_context functions. Separate calls sites
 * increase performance by 25% at least on P4 for invoke+yield back loops
 * at the cost of a slightly higher instruction cache use and is thus enabled by
 * default.
 */
//#define BOOST_COROUTINE_NO_SEPARATE_CALL_SITES

extern "C" void swapcontext_stack (void***, void**) throw();
extern "C" void swapcontext_stack2 (void***, void**) throw();
extern "C" void swapcontext_stack3 (void***, void**) throw();

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace coroutines { namespace detail {
  namespace windows64 
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

    class ia64_win_context_impl_base 
    {
    public:
      ia64_win_context_impl_base() : m_sp(0) {}

      void prefetch() const 
      {
      }

      /**
       * Free function. Saves the current context in @p from
       * and restores the context in @p to.
       * @note This function is found by ADL.
       */     
      friend void 
      swap_context(ia64_win_context_impl_base& from, 
                   ia64_win_context_impl_base const& to, 
                   default_hint) 
      {
        to.prefetch();
        swapcontext_stack(&from.m_sp, to.m_sp);
      }

#ifndef BOOST_COROUTINE_NO_SEPARATE_CALL_SITES
      friend void 
      swap_context(ia64_win_context_impl_base& from, 
                   ia64_win_context_impl_base const& to,
                   yield_hint) 
      {
        to.prefetch();
        swapcontext_stack2(&from.m_sp, to.m_sp);
      }

      friend void 
      swap_context(ia64_win_context_impl_base& from, 
                   ia64_win_context_impl_base const& to,
                   yield_to_hint) 
      {
        to.prefetch();
        swapcontext_stack3(&from.m_sp, to.m_sp);
      }
#endif

    protected:
      void ** m_sp;
    };

    //this should be a fine default.
    static const std::size_t stack_alignment = sizeof(void*) > 8 ? sizeof(void*) : 8;

    struct stack_aligner {
      boost::type_with_alignment<stack_alignment>::type dummy;
    };

    /**
     * Stack allocator and deleter functions.
     * Better implementations are possible using
     * mmap (might be required on some systems) and/or
     * using a pooling allocator.
     * NOTE: the SuSv3 documentation explicitly allows
     * the use of malloc to allocate stacks for makectx.
     * We use new/delete for guaranteed alignment.
     */
    inline
    void* alloc_stack(std::size_t size) {
      return new stack_aligner[size/sizeof(stack_aligner)];
    }

    inline
    void free_stack(void* stack, std::size_t size) {
      delete [] static_cast<stack_aligner*>(stack);
    }

    class ia64_win_context_impl : public ia64_win_context_impl_base
    {
    public:
      enum { default_stack_size = 8192 };
      
      typedef ia64_win_context_impl_base context_impl_base;

      ia64_win_context_impl() 
        : m_stack(0) 
      {}
        
      /**
       * Create a context that on restore invokes Functor on
       *  a new stack. The stack size can be optionally specified.
       */
      template<typename Functor>
      ia64_win_context_impl(Functor& cb, std::ptrdiff_t stack_size = -1) 
        : m_stack_size(stack_size == -1 ? default_stack_size : stack_size),
          m_stack(alloc_stack(m_stack_size)) 
      {
        m_sp = ((void**)m_stack + (m_stack_size/sizeof(void*)));
        BOOST_ASSERT(m_stack);

        typedef void fun(Functor*);
        fun* funp = &trampoline<Functor>;

        *--m_sp = &cb;     // parm 0 of trampoline;
        *--m_sp = 0;       // dummy return address for trampoline
        *--m_sp = (void*) funp ;// return addr (here: start addr)  NOTE: the unsafe cast is safe on IA64
        *--m_sp = 0;       // rbp                                  
        *--m_sp = 0;       // rbx                                  
        *--m_sp = 0;       // rsi                                  
        *--m_sp = 0;       // rdi        
      }
      
      ~ia64_win_context_impl() 
      {
        if(m_stack)
          free_stack(m_stack, m_stack_size);
      }

    private:
      std::ptrdiff_t m_stack_size;
      void * m_stack;
    };
    
    typedef ia64_win_context_impl context_impl;
  }
}}}

#endif

