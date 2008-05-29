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
    /// This is the representation of a ParalleX thread
    class px_thread 
    {
    private:
        typedef boost::coroutines::shared_coroutine<bool()> coroutine_type;
        
    public:
        /// parcel action code: the action to be performed on the destination 
        /// object (the px_thread)
        enum actions
        {
            some_action_code = 0
        };
        enum thread_state
        {
            init = 0,
            depleted = 1,
            pending = 2,
            running = 3,
            waiting = 4,
            block = 5
        };

        /// This is the component id. Every component needs to have an embedded
        /// enumerator 'value' which is used by the generic action implementation
        /// to associate this component with a given action.
        enum { value = components::component_px_thread };

        class switch_status
        {
        public:
            switch_status (thread_state& my_state, thread_state new_state)
                : outer_state_(my_state), prev_state(my_state)
            { 
                my_state = new_state;
            }

            ~switch_status ()
            {
                outer_state_ = prev_state;
            }

        private:
            thread_state& outer_state_;
            thread_state prev_state;
        };

        /// 
        px_thread(boost::function<bool (px_thread_self&)> threadfunc, thread_state new_state=init) 
          : coroutine_ (threadfunc), current_state_ (new_state) 
        { }
        
        ~px_thread() 
        { }

        // execute the thread function
        bool operator()(void)
        {
            switch_status thrd_stat (current_state_, running);
            return coroutine_();
        }

        thread_state get_state() const 
        {
            return current_state_ ;
        }

        void set_state(thread_state new_state)
        {
            current_state_ = new_state ;
            std::cout << current_state_ << std::endl;
        }

    private:
        coroutine_type coroutine_;
        thread_state current_state_;
    };

///////////////////////////////////////////////////////////////////////////////
}}

#endif