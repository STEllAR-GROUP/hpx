//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PX_THREAD_MAY_20_2008_0910AM)
#define HPX_PX_THREAD_MAY_20_2008_0910AM

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/noncopyable.hpp>
#include <boost/coroutine/coroutine.hpp>
#include <boost/coroutine/shared_coroutine.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/components/component_type.hpp>
#include <hpx/components/server/wrapper.hpp>

namespace hpx { namespace threadmanager { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    /// This is the representation of a ParalleX thread
    class px_thread : private boost::noncopyable
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
        typedef coroutine_type::thread_id_type thread_id_type;

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
                thread_id_type id, threadmanager& tm, 
                thread_state new_state)
          : coroutine_(threadfunc, id), tm_(tm), current_state_(new_state) 
        {}

        ~px_thread() 
        {}

        /// execute the thread function
        thread_state execute()
        {
            switch_status thrd_stat (current_state_, active);
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

        threadmanager& get_thread_manager() 
        {
            return tm_;
        }

    private:
        coroutine_type coroutine_;
        threadmanager& tm_;
        thread_state current_state_;
    };

///////////////////////////////////////////////////////////////////////////////
}}}

namespace hpx { namespace threadmanager 
{
    ///////////////////////////////////////////////////////////////////////////
    class px_thread : public components::wrapper<detail::px_thread>
    {
    private:
        typedef detail::px_thread wrapped_type;
        typedef components::wrapper<wrapped_type> base_type;

        // avoid warning about using 'this' in initializer list
        px_thread* This() { return this; }

    public:
        typedef detail::px_thread::thread_id_type thread_id_type;

        px_thread(boost::function<thread_function_type> threadfunc, 
                threadmanager& tm, thread_state new_state = init)
          : base_type(new detail::px_thread(threadfunc, This(), tm, new_state))
        {}

        ~px_thread() 
        {}

        thread_id_type get_thread_id() const
        {
            return const_cast<px_thread*>(this);
        }

        thread_state get_state() const 
        {
            return base()->get_state();
        }

        void set_state(thread_state new_state)
        {
            base()->set_state(new_state);
        }

        thread_state operator()()
        {
            return base()->execute();
        }

    protected:
        base_type& base() { return *this; }
        base_type const& base() const { return *this; }
    };

    ///////////////////////////////////////////////////////////////////////////
    px_thread::thread_id_type const invalid_thread_id = 0;

}}

#endif
