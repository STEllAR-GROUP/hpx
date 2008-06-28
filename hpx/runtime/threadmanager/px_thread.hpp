//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
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
#include <hpx/lcos/base_lco.hpp>

namespace hpx { namespace threadmanager { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // This is the representation of a ParalleX thread
    class px_thread : public lcos::base_lco, private boost::noncopyable
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
        px_thread(boost::function<thread_function_type> func, 
                thread_id_type id, threadmanager& tm, thread_state newstate)
          : coroutine_(func, id), tm_(tm), current_state_(newstate)
        {}

        ~px_thread() 
        {}

        thread_state execute()
        {
            return coroutine_();
        }

        thread_state get_state() const 
        {
            return current_state_ ;
        }

        void set_state(thread_state newstate)
        {
            current_state_ = newstate;
        }

        thread_id_type get_thread_id() const
        {
            return coroutine_.get_thread_id();
        }

        threadmanager& get_thread_manager() 
        {
            return tm_;
        }

    public:
        // action support

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_px_thread };

        /// 
        thread_state set_event (px_thread_self&, applier::applier&)
        {
            // we need to reactivate the thread itself
            if (suspended == current_state_)
                tm_.set_state(get_thread_id(), pending);

            // FIXME: implement functionality required for depleted state
            return terminated;
        }

    private:
        coroutine_type coroutine_;
        threadmanager& tm_;
        thread_state current_state_;
    };

///////////////////////////////////////////////////////////////////////////////
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threadmanager 
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class px_thread px_thread.hpp hpx/runtime/threadmanager/px_thread.hpp
    ///
    /// A \a px_thread is the representation of a ParalleX thread. It's a first
    /// class object in ParalleX. In our implementation this is a user level 
    /// thread running on top of one of the OS threads spawned by the \a 
    /// threadmanager.
    ///
    /// A \a px_thread encapsulates:
    ///  - A thread status word (see the functions \a px_thread#get_state and 
    ///    \a px_thread#set_state)
    ///  - A function to execute (the thread function)
    ///  - A frame (in this implementation this is a block of memory used as 
    ///    the threads stack)
    ///  - A block of registers (not implemented yet)
    ///
    /// Generally, \a px_threads are not created or executed directly. All 
    /// functionality related to the management of \a px_thread's is 
    /// implemented by the \a threadmanager.
    class px_thread 
      : public components::wrapper<detail::px_thread, px_thread, boost::mpl::true_>
    {
    private:
        typedef detail::px_thread wrapped_type;
        typedef components::wrapper<
            wrapped_type, px_thread, boost::mpl::true_> 
        base_type;

        // avoid warning about using 'this' in initializer list
        px_thread* This() { return this; }

    public:
        /// \brief Construct a new \a px_thread
        ///
        /// \param func     [in] The thread function to execute by this 
        ///                 \a px_thread.
        /// \param tm       [in] A reference to the thread manager this 
        ///                 \a px_thread will be associated with.
        /// \param newstate [in] The initial thread state this instance will
        ///                 be initialized with.
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

        /// \brief Allow to access the thread manager instance this thread has 
        ///        been associated with.
        threadmanager& get_thread_manager() 
        {
            return base()->get_thread_manager();
        }

        /// The get_state function allows to query the state of this thread
        /// instance.
        ///
        /// \returns        This function returns the current state of this
        ///                 thread. It will return one of the values as defined 
        ///                 by the \a thread_state enumeration.
        ///
        /// \note           This function will be seldom used directly. Most of 
        ///                 the time the state of a thread will be retrieved
        ///                 by using the function \a threadmanager#get_state.
        thread_state get_state() const 
        {
            return base()->get_state();
        }

        /// The set_state function allows to change the state this thread 
        /// instance.
        ///
        /// \param newstate [in] The new state to be set for the thread 
        ///                 referenced by the \a id parameter.
        ///
        /// \note           This function will be seldom used directly. Most of 
        ///                 the time the state of a thread will have to be 
        ///                 changed using the threadmanager. Moreover,
        ///                 changing the thread state using this function does
        ///                 not change its scheduling status. It only sets the
        ///                 thread's status word. To change the thread's 
        ///                 scheduling status \a threadmanager#set_state should
        ///                 be used.
        void set_state(thread_state new_state)
        {
            base()->set_state(new_state);
        }

        /// \brief Execute the thread function
        ///
        /// \returns        This function returns the thread state the thread
        ///                 should be scheduled from this point on. The thread
        ///                 manager will use the returned value to set the 
        ///                 thread's scheduling status.
        thread_state operator()()
        {
            return base()->execute();
        }

    protected:
        base_type& base() { return *this; }
        base_type const& base() const { return *this; }
    };

    ///////////////////////////////////////////////////////////////////////////
    thread_id_type const invalid_thread_id = 0;

    // support for boost::intrusive_ptr<px_thread>
    inline void intrusive_ptr_add_ref(px_thread* p)
    {
        ++p->use_count_;
    }

    inline void intrusive_ptr_release(px_thread* p)
    {
        if (--p->use_count_ == 0)
            delete p;
    }

}}

#endif
