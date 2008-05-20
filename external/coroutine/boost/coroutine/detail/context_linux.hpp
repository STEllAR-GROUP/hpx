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

#ifndef BOOST_COROUTINE_CONTEXT_LINUX_HPP_20060601
#define BOOST_COROUTINE_CONTEXT_LINUX_HPP_20060601

#if defined(__GNUC__) && defined(__i386__) && !defined(BOOST_COROUTINE_NO_ASM)
#include <cstdlib>
#include <cstddef>
#include <boost/coroutine/detail/posix_utility.hpp>
#include <boost/coroutine/detail/swap_context.hpp>

/*
 * Defining BOOST_COROUTINE_INLINE_ASM will enable the inlin3
 * assembler version of swapcontext_stack.
 * The inline asm, with all required clobber flags, is usually no faster
 * than the out-of-line function, and it is not yet clear if
 * it is always reliable (i.e. if the compiler always saves the correct
 * registers). FIXME: it is currently missing at least MMX and XMM registers in
 * the clobber list.
 */
 
/* 
 * Defining BOOST_COROUTINE_NO_SEPARATE_CALL_SITES will disable separate 
 * invoke, yield and yield_to swap_context functions. Separate calls sites
 * increase performance by 25% at least on P4 for invoke+yield back loops
 * at the cost of a slightly higher instruction cache use and is thus enabled by
 * default.
 */
//#define BOOST_COROUTINE_NO_SEPARATE_CALL_SITES
//#define BOOST_COROUTINE_INLINE_ASM


#ifndef BOOST_COROUTINE_INLINE_ASM
extern "C" void swapcontext_stack (void***, void**) throw()  __attribute((regparm(2)));
extern "C" void swapcontext_stack2 (void***, void**) throw()  __attribute((regparm(2)));
extern "C" void swapcontext_stack3 (void***, void**) throw()  __attribute((regparm(2)));
#else

#  if 1
void 
inline
swapcontext_stack(void***from_sp, void**to_sp) throw() {
  asm volatile
    ("\n\t pushl %%ebp"
     "\n\t pushl %[from]"
     "\n\t pushl %[to]"
     "\n\t pushl $0f"
     "\n\t movl %%esp, (%[from])" 
     "\n\t movl %[to], %%esp"
     "\n\t popl %%ecx" 
     "\n\t jmp  *%%ecx"
     "\n0:\t popl %[to]"
     "\n\t popl %[from]"
     "\n\t popl %%ebp"
     :: 
     [from] "a" (from_sp),
     [to]   "d" (to_sp)
     :
     "cc", 
     "%ecx",
     "%ebx", 
     "%edi", 
     "%esi",
     "%st",
     "%st(1)",
     "%st(2)",
     "%st(3)",
     "%st(4)",
     "%st(5)",
     "%st(6)",
     "%st(7)",
     "memory"
   );
  
}
#  else
typedef   void (*fun_type)(void***from_sp, void**to_sp) 
  __attribute((regparm(2)));

fun_type get_swapper();//  __attribute__((pure));

inline 
fun_type get_swapper(){
  fun_type ptr;
  asm volatile("mov $0f, %[result]"
               "\n\t jmp 1f"
               "\n0:"
               //"\n\t movl 16(%%edx), %%ecx"
               "\n\t pushl %%ebp"
               "\n\t pushl %%ebx"
               "\n\t pushl %%esi"
               "\n\t pushl %%edi"
               "\n\t movl %%esp, (%%eax)"
               "\n\t movl %%edx, %%esp"
               "\n\t popl %%edi"
               "\n\t popl %%esi"
               "\n\t popl %%ebx"
               "\n\t popl %%ebp"
               "\n\t popl %%ecx"
               "\n\t jmp *%%ecx"
               "\n1:"
               :
               [result] "=g" (ptr) 
               :       
               );
  return ptr;
};

void 
inline
swapcontext_stack(void***from_sp, void**to_sp) throw() {
  fun_type ptr = get_swapper();
  ptr(from_sp, to_sp);
}

#  endif

void 
inline
swapcontext_stack2(void***from_sp, void**to_sp) throw() {
  swapcontext_stack(from_sp, to_sp);
}

#endif

namespace boost { namespace coroutines { namespace detail {
  namespace oslinux {
    template<typename T>
    void trampoline(T* fun);
    
    template<typename T>
    inline
    void
    trampoline(T * fun) { 
      (*fun)();
      std::abort();
    }

    class ia32_gcc_context_impl_base {
    public:
      ia32_gcc_context_impl_base() {};

      void prefetch() const {
        __builtin_prefetch (m_sp, 1, 3);
        __builtin_prefetch (m_sp, 0, 3);
        __builtin_prefetch ((void**)m_sp+64/4, 1, 3);
        __builtin_prefetch ((void**)m_sp+64/4, 0, 3);
        __builtin_prefetch ((void**)m_sp+32/4, 1, 3);
        __builtin_prefetch ((void**)m_sp+32/4, 0, 3);
        __builtin_prefetch ((void**)m_sp-32/4, 1, 3);
        __builtin_prefetch ((void**)m_sp-32/4, 0, 3);
        __builtin_prefetch ((void**)m_sp-64/4, 1, 3);
        __builtin_prefetch ((void**)m_sp-64/4, 0, 3);
      }

      /**
       * Free function. Saves the current context in @p from
       * and restores the context in @p to.
       * @note This function is found by ADL.
       */     
      friend 
      void 
      swap_context(ia32_gcc_context_impl_base& from, 
                   ia32_gcc_context_impl_base const& to, 
                   default_hint) {
        to.prefetch();
        swapcontext_stack(&from.m_sp, to.m_sp);
      }

#ifndef BOOST_COROUTINE_NO_SEPARATE_CALL_SITES
      friend 
      void 
      swap_context(ia32_gcc_context_impl_base& from, 
                   ia32_gcc_context_impl_base const& to,
                   yield_hint) {
        to.prefetch();
        swapcontext_stack2(&from.m_sp, to.m_sp);
      }

      friend 
      void 
      swap_context(ia32_gcc_context_impl_base& from, 
                   ia32_gcc_context_impl_base const& to,
                   yield_to_hint) {
        to.prefetch();
        swapcontext_stack2(&from.m_sp, to.m_sp);
      }

#endif

    protected:
      void ** m_sp;
 
    };

    class ia32_gcc_context_impl  : public ia32_gcc_context_impl_base{
    public:
      enum {default_stack_size = 8192};
      
      typedef ia32_gcc_context_impl_base context_impl_base;

      ia32_gcc_context_impl() :
        m_stack(0) {}
      /**
       * Create a context that on restore invokes Functor on
       *  a new stack. The stack size can be optionally specified.
       */
      template<typename Functor>
          ia32_gcc_context_impl(Functor& cb, std::ptrdiff_t stack_size = -1) :
        m_stack_size(stack_size == -1? default_stack_size: stack_size),
        m_stack(posix::alloc_stack(m_stack_size)) 
      {
        m_sp = ((void**)m_stack + (m_stack_size/sizeof(void*)));
        BOOST_ASSERT(m_stack);
        typedef void fun(Functor*);
        fun * funp = trampoline;
#ifndef BOOST_COROUTINE_INLINE_ASM
        *--m_sp = &cb;     // parm 0 of trampoline;
        *--m_sp = 0;        // dummy return address for trampoline
        *--m_sp = (void*) funp ;// return addr (here: start addr)  NOTE: the unsafe cast is safe on IA32
        *--m_sp = 0;       // ebp                                  
        *--m_sp = 0;       // ebx                                  
        *--m_sp = 0;       // esi                                  
        *--m_sp = 0;       // edi        
#else
        *--m_sp = &cb;     // parm 0 of trampoline;
        *--m_sp = 0;        // dummy return address for trampoline
        *--m_sp = (void*) funp ;// return addr (here: start addr)  NOTE: the unsafe cast is safe on IA32
#endif
      }
      
      ~ia32_gcc_context_impl() {
        if(m_stack)
          posix::free_stack(m_stack, m_stack_size);
      }

    private:
          std::ptrdiff_t m_stack_size;
      void * m_stack;
    };
    
    typedef ia32_gcc_context_impl context_impl;
  }
} } }

#elif defined(__linux)

/**
 * For all other linux systems use the standard posix implementation.
 */
#include <boost/coroutine/detail/context_posix.hpp>
namespace boost { namespace coroutines { namespace detail { namespace oslinux {
    typedef posix::context_impl context_impl;
} } } }

#else
#error This header can only be included when compiling for linux systems.
#endif


#endif
