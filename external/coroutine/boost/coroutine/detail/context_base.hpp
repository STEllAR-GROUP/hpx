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

#ifndef BOOST_COROUTINE_CONTEXT_IMPL_HPP_20060810
#define BOOST_COROUTINE_CONTEXT_IMPL_HPP_20060810
/*
 * Currently asio can, in some cases. call copy constructors and
 * operator= from different threads, even if in the
 * one-thread-per-service model. (i.e. from the resolver thread)
 * This will be corrected in future versions, but for now
 * we will play it safe and use an atomic count. The overhead shouldn't
 * be big.
 */
#define BOOST_COROUTINE_USE_ATOMIC_COUNT

#ifdef BOOST_COROUTINE_USE_ATOMIC_COUNT
#  include <boost/detail/atomic_count.hpp>
#endif
#include <cstddef>
#include <algorithm> //for swap
#include <boost/coroutine/detail/swap_context.hpp> //for swap hints
#include <boost/intrusive_ptr.hpp>
#include <boost/coroutine/exception.hpp>
#include <boost/coroutine/detail/noreturn.hpp>
namespace boost { namespace coroutines { namespace detail {
 
  const std::ptrdiff_t default_stack_size = -1;

  template<typename ContextImpl>
  class context_base : public ContextImpl {
  public:

    typedef ContextImpl context_impl;
    typedef context_base<context_impl> type;
    typedef boost::intrusive_ptr<type> pointer;
    typedef void deleter_type(const type*);
    
    template<typename Derived>
	context_base(Derived& derived, 
		     std::ptrdiff_t stack_size) :
      context_impl(derived, stack_size),
      m_counter(0),
      m_deleter(&deleter<Derived>),
      m_state(ctx_ready), 
      m_exit_state(ctx_exit_not_requested),
      m_exit_status(ctx_not_exited),
      m_wait_counter(0),
      m_operation_counter(0),
      m_type_info(0) {}
    
    friend
    void intrusive_ptr_add_ref(type * ctx) {
      ctx->acquire();
    }
    
    friend
    void intrusive_ptr_release(type * ctx) {
      ctx->release();
    }
      
    bool unique() const {
      return count() == 1;
    }

    std::size_t count() const {
      return m_counter;
    }
      
    void acquire() const {
      ++m_counter;
    }
      
    void release() const {
      BOOST_ASSERT(m_counter);
      if(--m_counter == 0) {
	m_deleter(this);
      }
    }

    void count_down() throw() {
      BOOST_ASSERT(m_operation_counter) ;
      --m_operation_counter;
    }

    void count_up() throw() {
      ++m_operation_counter;
    }

    // return true if there are operations pending.
    int pending() const {
      return m_operation_counter;
    }

    /*
     * A signal may occur only when a context is 
     * not running (is delivered sinchrononously).
     * This means that state MUST NOT be busy.
     * It may be ready or waiting.
     * returns 'ready()'.
     * Nothrow.
     */
    bool signal () throw() {
      BOOST_ASSERT(!running() && !exited());
      BOOST_ASSERT(m_wait_counter) ;

      --m_wait_counter;
      if(!m_wait_counter && m_state == ctx_waiting)
	m_state = ctx_ready;      
      return ready();
    }

    /*
     * Wake up a waiting context.
     * Similar to invoke(), but *does not
     * throw* if the coroutine exited normally
     * or entered the wait state.
     * It *does throw* if the coroutine
     * exited abnormally.
     * Return: false if invoke() would have thrown,
     *         true otherwise.
     * 
     */
    bool wake_up() {
      BOOST_ASSERT(ready());
      do_invoke();
      // TODO: could use a binary 'or' here to eliminate
      // shortcut evaluation (and a branch), but maybe the compiler is
      // smart enough to do it anyway as there are no side effects.
      if(m_exit_status || m_state == ctx_waiting) {
	if(m_state == ctx_waiting)
	  return false;
	if(m_exit_status == ctx_exited_return)
	  return true;
	if(m_exit_status == ctx_exited_abnormally) {
	std::type_info const * tinfo =0;
	std::swap(m_type_info, tinfo);
	throw abnormal_exit(tinfo?*tinfo: typeid(unknown_exception_tag));
	} else if(m_exit_status == ctx_exited_exit)
	  return false;
	else {
	  BOOST_ASSERT(0 && "unkonw exit status");
	}
      }
      return true;
    }
    /*
     * Returns true if the context is runnable.
     */
    bool ready() const {
      return m_state == ctx_ready;
    }

    /*
     * Returns true if the context is in wait
     * state.
     */
    bool waiting() const {
      return m_state == ctx_waiting;
    }

    bool running() const {
      return m_state == ctx_running;
    }

    bool exited() const {
      return m_state == ctx_exited;
    }

    // Resume coroutine.
    // Pre:  The coroutine must be ready.
    // Post: The coroutine relinquished control. It might be ready, waiting
    //       or exited.
    // Throws:- 'waiting' if the coroutine entered the wait state,
    //        - 'coroutine_exited' if the coroutine exited by an uncaught
    //          'exit_exception'.
    //        - 'abnormal_exit' if the coroutine was exited by another
    //          uncaught exception.
    // Note, it guarantees that the coroutine is resumed. Can throw only
    // on return.
    void invoke() {
      BOOST_ASSERT(ready());
      do_invoke();
      // TODO: could use a binary or here to eliminate
      // shortcut evaluation (and a branch), but maybe the compiler is
      // smart enough to do it anyway as there are no side effects.
      if(m_exit_status || m_state == ctx_waiting) {
	if(m_state == ctx_waiting)
	  throw waiting();
	if(m_exit_status == ctx_exited_return)
	  return;
	if(m_exit_status == ctx_exited_abnormally) {
	std::type_info const * tinfo =0;
	std::swap(m_type_info, tinfo);
	throw abnormal_exit(tinfo?*tinfo: typeid(unknown_exception_tag));
	} else if(m_exit_status == ctx_exited_exit)
	  throw coroutine_exited();
      	else {
	  BOOST_ASSERT(0 && "unknown exit status");
	}
      }
    }

    // Put coroutine in ready state and relinquish control
    // to caller until resumed again.
    // Pre:  Coroutine is running.
    //       Exit not pending.
    //       Operations not pending.
    // Post: Coroutine is running.
    // Throws: exit_exception, if exit is pending *after* it has been
    //         resumed.
    void yield() {
      BOOST_ASSERT(m_exit_state < ctx_exit_signaled); //prevent infinite loops
      BOOST_ASSERT(running());
      BOOST_ASSERT(!pending());

      m_state = ctx_ready;
      do_yield();

      BOOST_ASSERT(running());
      check_exit_state();
    }

    //
    // If n > 0, put the coroutine in the wait state
    // then relinquish control to caller. 
    // If n = 0 do nothing.
    // The coroutine will remain in the wait state until
    // is signaled 'n' times.
    // Pre:  0 <= n < pending()
    //       Coroutine is running.
    //       Exit not pending.
    // Post: Coroutine is running.
    //       The coroutine has been signaled 'n' times unless an exit
    //        has been signaled.
    // Throws: exit_exception.
    // FIXME: currently there is a BIG problem. A coroutine cannot
    // be exited as long as there are futures pending.
    // The exit_exception would cause the future to be destroyed and
    // an assertion to be generated. Removing an assertion is not a 
    // solution because we would leak the coroutine impl. The callback
    // bound to the future in fact hold a reference to it. If the coroutine
    // is exited the callback cannot be called.
    void wait(int n) {
      BOOST_ASSERT(!(n<0));
      BOOST_ASSERT(m_exit_state < ctx_exit_signaled); //prevent infinite loop
      BOOST_ASSERT(running());
      BOOST_ASSERT(!(pending() < n));

      if(n == 0) return;
      m_wait_counter = n;

      m_state = ctx_waiting;
      do_yield();

      BOOST_ASSERT(m_state == ctx_running);
      check_exit_state();
      BOOST_ASSERT(m_wait_counter == 0);
    }

    // Throws: exit_exception.
    void yield_to(context_base& to) {
      BOOST_ASSERT(m_exit_state < ctx_exit_signaled); //prevent infinite loops
      BOOST_ASSERT(m_state == ctx_running);
      BOOST_ASSERT(to.ready());
      BOOST_ASSERT(!to.pending());

      std::swap(m_caller, to.m_caller);
      std::swap(m_state, to.m_state);
      swap_context(*this, to, detail::yield_to_hint());

      BOOST_ASSERT(m_state == ctx_running);
      check_exit_state();
    }

    // Cause this coroutine to exit.
    // Can only be called on a ready coroutine.
    // Cannot be called if there are pending operations.
    // It follows that cannot be called from 'this'.
    // Nothrow.
    void exit() throw(){
      BOOST_ASSERT(!pending());
      BOOST_ASSERT(ready()) ;
      if(m_exit_state < ctx_exit_pending) 
	m_exit_state = ctx_exit_pending;	
      do_invoke();
      BOOST_ASSERT(exited()); //at this point the coroutine MUST have exited.
    }

    // Always throw exit_exception.
    // Never returns from standard control flow.
    BOOST_COROUTINE_NORETURN(void exit_self()) {
      BOOST_ASSERT(!pending());
      BOOST_ASSERT(running());
      m_exit_state = ctx_exit_pending;	
      throw exit_exception();
    }

    // Nothrow.
    ~context_base() throw() {
      BOOST_ASSERT(!running());
      try {
	if(!exited())
	  exit();
	BOOST_ASSERT(exited());
      } catch(...) {}
    }

  protected:
    // global coroutine state
    enum context_state {
      ctx_running,  // context running.
      ctx_ready,    // context at yield point.
      ctx_waiting,     // context waiting for events.
      ctx_exited    // context is finished.
    };

    // exit request state
    enum context_exit_state {
      ctx_exit_not_requested,  // exit not requested.
      ctx_exit_pending,   // exit requested.
      ctx_exit_signaled,  // exit request delivered.
    };
    
    // exit status
    enum context_exit_status {
      ctx_not_exited,
      ctx_exited_return,  // process exited by return.
      ctx_exited_exit,    // process exited by exit().
      ctx_exited_abnormally // process exited uncleanly.
    };

    // Cause the coroutine to exit if 
    // a exit request is pending.
    // Throws: exit_exception if an exit request is pending.
    void check_exit_state() {
      BOOST_ASSERT(running());
      if(!m_exit_state) return;
      throw exit_exception();
    }

    // Nothrow.
    void do_return(context_exit_status status, std::type_info const* info) throw() {
      BOOST_ASSERT(status != ctx_not_exited);
      BOOST_ASSERT(m_state == ctx_running);
      m_type_info = info;
      m_state = ctx_exited;
      m_exit_status = status;
      do_yield();
    }

  private:

    // Nothrow.
    void do_yield() throw() {
      swap_context(*this, m_caller, detail::yield_hint());
    }

    // Nothrow.
    void do_invoke() throw (){
      BOOST_ASSERT(ready() || waiting());
      m_state = ctx_running;
      swap_context(m_caller, *this, detail::invoke_hint());
    }

    template<typename ActualCtx>
    static void deleter (const type* ctx){
      delete static_cast<ActualCtx*>(const_cast<type*>(ctx));
    }
            
    typedef typename context_impl::context_impl_base ctx_type;
    ctx_type m_caller;
    mutable 
#ifndef BOOST_COROUTINE_USE_ATOMIC_COUNT
    std::size_t
#else
    boost::detail::atomic_count
#endif
    m_counter;
    deleter_type * m_deleter;
    context_state m_state;
    context_exit_state m_exit_state;
    context_exit_status m_exit_status;
    int m_wait_counter;   
    int m_operation_counter;    

    // This is used to generate a meaningful exception trace.
    std::type_info const* m_type_info;
  };

} } }
#endif
