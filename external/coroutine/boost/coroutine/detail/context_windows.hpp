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

#ifndef BOOST_COROUTINE_CONTEXT_WINDOWS_HPP_20060625
#define BOOST_COROUTINE_CONTEXT_WINDOWS_HPP_20060625
#include <windows.h>
#include <winnt.h>
#include <boost/config.hpp>
#include <boost/config.hpp>
#include <boost/noncopyable.hpp>
#include <boost/coroutine/exception.hpp>
#include <boost/coroutine/detail/swap_context.hpp>
namespace boost {namespace coroutines {namespace detail{
  namespace windows {

    typedef LPVOID fiber_ptr;

    /*
     * This number (0x1E00) has been sighted in the wild (at least on windows XP systems)
     * as return value from GetCurrentFiber() on non fibrous threads. This is somehow related
     * to OS/2 where the current fiber pointer is overloaded as a version field.
     * On non-NT systems, 0 is returned. 
     */
    fiber_ptr const fiber_magic = reinterpret_cast<fiber_ptr>(0x1E00);
                
    /*
     * Return true if current thread is a fiber.
     * FIXME: on longhorn should use IsThreadAFiber
     */
    inline bool is_fiber() {
      fiber_ptr current = GetCurrentFiber();
      return current != 0 && current != fiber_magic;
    }

    /*
     * Windows implementation for the context_impl_base class.
     * @note context_impl is not required to be consistent
     * If not initialized it can only be swapped out, not in 
     * (at that point it will be initialized).
     */
    class fibers_context_impl_base {
    public:
      /**
       * Create an empty context. 
       * An empty context cannot be restored from,
       * but can be saved in.
       */
      fibers_context_impl_base() :
        m_ctx(0) {}
        
      /*
       * Free function. Saves the current context in @p from
       * and restores the context in @p to. On windows the from
       * parameter is ignored. The current context is saved on the 
       * current fiber.
       * Note that if the current thread is not a fiber, it will be
       * converted to fiber on the fly on call and unconverted before
       * return. This is expensive. The user should convert the 
       * current thread to a fiber once on thread creation for better performance.
       * Note that we can't leave the thread unconverted on return or else we 
       * will leak resources on thread destruction. Do the right thing by
       * default.
       */     
      friend 
      void 
      swap_context(fibers_context_impl_base& from, 
                   const fibers_context_impl_base& to,
                   default_hint) {
        if(!is_fiber()) {
          BOOST_ASSERT(from.m_ctx == 0);
          from.m_ctx = ConvertThreadToFiber(0);
          BOOST_ASSERT(from.m_ctx != 0);
          
          SwitchToFiber(to.m_ctx); 
          
          BOOL result = ConvertFiberToThread();
          BOOST_ASSERT(result);
          (void)result;
          from.m_ctx = 0;
        } else {
          bool call_from_main = from.m_ctx == 0;
          if(call_from_main)
            from.m_ctx = GetCurrentFiber();
          SwitchToFiber(to.m_ctx); 
          if(call_from_main)
            from.m_ctx = 0;
        }
      }

      ~fibers_context_impl_base() {}
    protected:
      explicit
      fibers_context_impl_base(fiber_ptr ctx) :
        m_ctx(ctx) {}

      fiber_ptr m_ctx;
    };

    template<typename T>
    inline
    VOID CALLBACK
    trampoline(LPVOID pv) {
      T* fun = static_cast<T*>(pv);
      BOOST_ASSERT(fun);
      (*fun)();
    }

    class fibers_context_impl :
      public fibers_context_impl_base,
      private boost::noncopyable {
    public:
      typedef fibers_context_impl_base context_impl_base;

      enum {default_stack_size = 8192};

      /**
       * Create a context that on restore invokes Functor on
       *  a new stack. The stack size can be optionally specified.
       */
      template<typename Functor>
      explicit
      fibers_context_impl(Functor& cb, std::ptrdiff_t stack_size) :
        fibers_context_impl_base
      (CreateFiber(stack_size== -1? default_stack_size : stack_size,
                   static_cast<LPFIBER_START_ROUTINE>(&trampoline<Functor>),
                   static_cast<LPVOID>(&cb)))
      {
        BOOST_ASSERT(m_ctx);
      }
                  
      ~fibers_context_impl() {
        DeleteFiber(m_ctx);
      }
                  
    private:
    };

    typedef fibers_context_impl context_impl;

  }
} } }
#endif
