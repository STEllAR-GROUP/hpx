//  Copyright (c) 2007 Robert Perricone
//  Copyright (c) 2007-2010 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_COROUTINE_CONTEXT_STACKLESS_HPP_20071126
#define BOOST_COROUTINE_CONTEXT_STACKLESS_HPP_20071126

#include <boost/coroutine/detail/swap_context.hpp>

/* 
 * Defining BOOST_COROUTINE_NO_SEPARATE_CALL_SITES will disable separate 
 * invoke, yield and yield_to swap_context functions. Separate calls sites
 * increase performance by 25% at least on P4 for invoke+yield back loops
 * at the cost of a slightly higher instruction cache use and is thus enabled by
 * default.
 */

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace coroutines { 

  // some platforms need special preparation of the main thread
  struct prepare_main_thread
  {
      prepare_main_thread() {}
      ~prepare_main_thread() {}
  };

  namespace detail { namespace stackless 
  {
    template<typename T>
    void trampoline(T* fun);

    template<typename T>
    inline void
    trampoline(void* p) 
    {
      T* fun = static_cast<T*>(p);
      BOOST_ASSERT(fun);
      (*fun)();
    }

    class stackless_context_impl;

    class stackless_context_impl_base : detail::context_impl_base  
    {
    protected:
      typedef void (*trampoline_type)(void*);
      trampoline_type function;
      void* data;

    public:
      stackless_context_impl_base() 
        : function(0), data(0)
      {}

      template<typename Functor>
      stackless_context_impl_base(Functor& cb) 
        : function(static_cast<trampoline_type>(&trampoline<Functor>)), 
          data(&cb)
      {}

      /**
       * Free function. Saves the current context in @p from
       * and restores the context in @p to.
       * @note This function is found by ADL.
       */     
      friend void swap_context(stackless_context_impl_base& from, 
          stackless_context_impl_base const& to, default_hint);
    };

    class stackless_context_impl : public stackless_context_impl_base
    {
    public:

      typedef stackless_context_impl_base context_impl_base;

      stackless_context_impl() {}

      /**
       * Create a context that on restore invokes Functor on
       *  a new stack. The stack size can be optionally specified.
       */
      template<typename Functor>
      stackless_context_impl(Functor& cb, std::ptrdiff_t stack_size = -1) 
        : stackless_context_impl_base(cb)
      {
      }
      
      ~stackless_context_impl() 
      {
      }

      friend void swap_context(stackless_context_impl_base& from, 
          stackless_context_impl_base const& to, default_hint);
    };
    
    typedef stackless_context_impl context_impl;

    /**
     * Free function. Saves the current context in @p from
     * and restores the context in @p to.
     * @note This function is found by ADL.
     */     
    inline void swap_context(stackless_context_impl_base& from, 
        stackless_context_impl_base const& to, default_hint) 
    {
        (*to.function)(to.data);
    }

  }
}}}

#endif
