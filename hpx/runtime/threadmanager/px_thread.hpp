//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PX_THREAD_MAY_20_2008_910AM)
#define HPX_PX_THREAD_MAY_20_2008_910AM

#include <boost/function.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/coroutine/coroutine.hpp>
#include <boost/coroutine/shared_coroutine.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/components/component_type.hpp>

namespace hpx { namespace threadmanager 
{
    ///////////////////////////////////////////////////////////////////////////
    /// \enum thread_state
    ///
    /// The thread_state enumerator encodes the current state of a \a px_thread
    /// instance
    enum thread_state
    {
        unknown = -1,
        init = 0,       ///< thread is initializing
        depleted = 1,   ///< thread has been depleted (deeply suspended)
        suspended = 2,  ///< thread has been suspended
        pending = 3,    ///< thread is pending (ready to run)
        running = 4,    ///< thread is currently running (active)
        stopped = 5     ///< thread has been stopped an may be garbage collected
    };

    ///////////////////////////////////////////////////////////////////////////
    /// This is the representation of a ParalleX thread
    class px_thread 
    {
    private:
        typedef 
            boost::coroutines::shared_coroutine<thread_state()> 
        coroutine_type;
        
        // helper class for switching thread state in and out during execution
        class switch_status
        {
        public:
            switch_status (thread_state& my_state, thread_state new_state)
                : outer_state_(my_state), prev_state_(my_state)
            { 
                my_state = new_state;
            }

            ~switch_status ()
            {
                outer_state_ = prev_state_;
            }

            // allow to change the state the thread will be switched to after 
            // execution
            thread_state operator=(thread_state new_state)
            {
                return prev_state_ = new_state;
            }
            
        private:
            thread_state& outer_state_;
            thread_state prev_state_;
        };

    public:
        /// parcel action code: the action to be performed on the destination 
        /// object (the px_thread)
        enum actions
        {
            some_action_code = 0
        };
        
        /// This is the component id. Every component needs to have an embedded
        /// enumerator 'value' which is used by the generic action implementation
        /// to associate this component with a given action.
        enum { value = components::component_px_thread };

        /// 
        px_thread(boost::function<thread_function_type> threadfunc, 
                thread_state new_state = init) 
          : coroutine_ (threadfunc), current_state_(new_state) 
        {}
        
        ~px_thread() 
        {}

        /// execute the thread function
        thread_state operator()()
        {
            switch_status thrd_stat (current_state_, running);
            return thrd_stat = coroutine_();
        }

        thread_state get_state() const 
        {
            return current_state_ ;
        }

        void set_state(thread_state new_state)
        {
            current_state_ = new_state ;
        }

    private:
        coroutine_type coroutine_;
        thread_state current_state_;
    };

///////////////////////////////////////////////////////////////////////////////
}}

#endif