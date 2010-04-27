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
    trampoline(T* fun) 
    { 
      (*fun)();
      std::abort();
    }

    class context_impl;

    class context_impl_base 
    {
    public:
      context_impl_base() {}

      /**
       * Free function. Saves the current context in @p from
       * and restores the context in @p to.
       * @note This function is found by ADL.
       */     
      friend void swap_context(context_impl_base& from, 
          context_impl const& to, default_hint);
    };

    class context_impl : public context_impl_base
    {
    public:

      typedef context_impl_base context_impl_base;

      context_impl() 
      {}

      /**
       * Create a context that on restore invokes Functor on
       *  a new stack. The stack size can be optionally specified.
       */
      template<typename Functor>
      context_impl(Functor& cb, std::ptrdiff_t stack_size = -1) 
      {
      }
      
      ~context_impl() 
      {
      }

      friend void swap_context(context_impl_base& from, 
          context_impl const& to, default_hint);
    };
    
    typedef context_impl context_impl;

    /**
     * Free function. Saves the current context in @p from
     * and restores the context in @p to.
     * @note This function is found by ADL.
     */     
    inline void swap_context(context_impl_base& from, 
        context_impl const& to, default_hint) 
    {
    }

  }
}}}

#endif
