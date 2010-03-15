//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PX_THREAD_MAY_20_2008_0910AM)
#define HPX_PX_THREAD_MAY_20_2008_0910AM

#include <hpx/config.hpp>

#include <boost/atomic.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/noncopyable.hpp>
#include <boost/coroutine/coroutine.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/detail/tagged_thread_state.hpp>
#include <hpx/lcos/base_lco.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // This is the representation of a ParalleX thread
    class thread : public lcos::base_lco, private boost::noncopyable
    {
        typedef boost::function<thread_function_type> function_type;

    public:
        thread(thread_init_data const& init_data, thread_id_type id, 
               thread_state_enum newstate)
          : coroutine_(init_data.func, id), 
            current_state_(thread_state(newstate)), 
            current_state_ex_(thread_state_ex(wait_signaled)),
            description_(init_data.description), 
            parent_thread_id_(init_data.parent_id),
            parent_locality_prefix_(init_data.parent_prefix),
            component_id_(init_data.lva),
            marked_state_(unknown)
        {
            // store the thread id of the parent thread, mainly for debugging 
            // purposes
            if (0 == parent_thread_id_) {
                thread_self* self = get_self_ptr();
                if (self)
                    parent_thread_id_ = self->get_thread_id();
            }
            if (0 == parent_locality_prefix_) 
                parent_locality_prefix_ = applier::get_prefix_id();
        }

        /// This constructor is provided just for compatibility with the scheme
        /// of component creation. But since threads never get created 
        /// by a factory (runtime_support) instance, we can leave this 
        /// constructor empty
        thread()
          : coroutine_(function_type(), 0), description_(""), 
            parent_locality_prefix_(0), parent_thread_id_(0), component_id_(0)
        {
            BOOST_ASSERT(false);    // shouldn't ever be called
        }

        ~thread() 
        {}

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        static components::component_type get_component_type();
        static void set_component_type(components::component_type);

        thread_state_enum execute()
        {
            thread_state_ex_enum current_state_ex = get_state_ex();
            set_state_ex(wait_signaled);
            return coroutine_(current_state_ex);
        }

        thread_state get_state() const 
        {
            return current_state_.load(boost::memory_order_acquire);
        }

        thread_state set_state(thread_state_enum newstate)
        {
            thread_state prev_state = current_state_.load(boost::memory_order_acquire);
            for (;;) {
                thread_state tmp = prev_state;

                using boost::lockfree::likely;
                if (likely(current_state_.compare_exchange_strong(
                        tmp, thread_state(newstate, tmp.get_tag() + 1))))
                {
                    return prev_state;
                }
                prev_state = tmp;
            }
        }

        bool set_state_tagged(thread_state_enum newstate, 
            thread_state& prev_state, thread_state& new_tagged_state)
        {
            thread_state tmp = prev_state;
            new_tagged_state = thread_state(newstate, prev_state.get_tag() + 1);
            if (current_state_.compare_exchange_strong(tmp, new_tagged_state))
            {
                prev_state = tmp;
                return true;
            }
            return false;
        }

        bool restore_state(thread_state_enum new_state, thread_state old_state)
        {
            return current_state_.compare_exchange_strong(
                old_state, thread_state(new_state, old_state.get_tag() + 1));
        }

        thread_state_ex get_state_ex() const 
        {
            return current_state_ex_.load(boost::memory_order_acquire);
        }

        thread_state_ex set_state_ex(thread_state_ex_enum new_state)
        {
            thread_state_ex prev_state = 
                current_state_ex_.load(boost::memory_order_acquire);

            for (;;) {
                thread_state_ex tmp = prev_state;

                using boost::lockfree::likely;
                if (likely(current_state_ex_.compare_exchange_strong(
                        tmp, thread_state_ex(new_state, tmp.get_tag() + 1))))
                {
                    return prev_state;
                }
                prev_state = tmp;
            }
        }

        thread_id_type get_thread_id() const
        {
            return coroutine_.get_thread_id();
        }

        std::size_t get_thread_phase() const
        {
            return coroutine_.get_thread_phase();
        }
 
        char const* const get_description() const
        {
            return description_;
        }

        boost::uint32_t get_parent_locality_prefix() const
        {
            return parent_locality_prefix_;
        }
        thread_id_type get_parent_thread_id() const
        {
            return parent_thread_id_;
        }
        naming::address::address_type get_component_id() const
        {
            return component_id_;
        }

        void set_marked_state(thread_state mark) const
        {
            marked_state_ = mark;
        }
        thread_state get_marked_state() const
        {
            return marked_state_;
        }

        // threads use a specialized allocator for fast creation/destruction
        static void *operator new(std::size_t size);
        static void operator delete(void *p, std::size_t size);

    public:
        // action support

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_thread };

        /// 
        void set_event();

        template <typename ManagedType>
        naming::id_type const& get_gid(ManagedType* p) const
        {
            if (!id_) {
                naming::gid_type gid = p->get_base_gid(); 
                naming::strip_credit_from_gid(gid);
                id_ = naming::id_type(gid, naming::id_type::unmanaged);
            }
            return id_;
        }

    private:
        coroutine_type coroutine_;

        mutable boost::atomic<thread_state> current_state_;
        mutable boost::atomic<thread_state_ex> current_state_ex_;

        // all of the following is debug/logging support information
        char const* const description_;
        boost::uint32_t parent_locality_prefix_;
        thread_id_type parent_thread_id_;
        naming::address::address_type const component_id_;
        mutable thread_state marked_state_;

        mutable naming::id_type id_;    // that's our gid
    };

///////////////////////////////////////////////////////////////////////////////
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads 
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class thread thread.hpp hpx/runtime/threads/thread.hpp
    ///
    /// A \a thread is the representation of a ParalleX thread. It's a first
    /// class object in ParalleX. In our implementation this is a user level 
    /// thread running on top of one of the OS threads spawned by the \a 
    /// threadmanager.
    ///
    /// A \a thread encapsulates:
    ///  - A thread status word (see the functions \a thread#get_state and 
    ///    \a thread#set_state)
    ///  - A function to execute (the thread function)
    ///  - A frame (in this implementation this is a block of memory used as 
    ///    the threads stack)
    ///  - A block of registers (not implemented yet)
    ///
    /// Generally, \a threads are not created or executed directly. All 
    /// functionality related to the management of \a thread's is 
    /// implemented by the \a threadmanager.
    class thread 
      : public components::managed_component<detail::thread, thread>
    {
    private:
        typedef detail::thread wrapped_type;
        typedef 
            components::managed_component<wrapped_type, thread> 
        base_type;

        // avoid warning about using 'this' in initializer list
        thread* This() { return this; }

    public:
        thread()
        {}

        /// \brief Construct a new \a thread
        ///
        /// \param func     [in] The thread function to execute by this 
        ///                 \a thread.
        /// \param tm       [in] A reference to the thread manager this 
        ///                 \a thread will be associated with.
        /// \param newstate [in] The initial thread state this instance will
        ///                 be initialized with.
        thread(thread_init_data const& init_data, 
                thread_state_enum new_state = init)
          : base_type(new detail::thread(init_data, This(), new_state))
        {
            LTM_(debug) << "thread::thread(" << this << "), description(" 
                        << init_data.description << ")";
        }

//         thread(boost::function<thread_function_type> threadfunc, 
//               thread_state new_state = init, char const* const desc = "",
//               thread_id_type parent_id = 0, boost::uint32_t parent_prefix = 0)
//           : base_type(new detail::thread(threadfunc, This(), new_state, desc, 
//                 parent_id, parent_prefix))
//         {
//             LTM_(debug) << "thread::thread(" << this << "), description(" 
//                         << desc << ")";
//         }

        ~thread() 
        {
            LTM_(debug) << "~thread(" << this << "), description(" 
                        << get()->get_description() << ")";
        }

        thread_id_type get_thread_id() const
        {
            return const_cast<thread*>(this);
        }

        std::size_t get_thread_phase() const
        {
            detail::thread const* t = get();
            return t ? t->get_thread_phase() : 0;
        }

        /// Return the locality of the parent thread
        boost::uint32_t get_parent_locality_prefix() const
        {
            detail::thread const* t = get();
            return t ? t->get_parent_locality_prefix() : 0;
        }

        /// Return the thread id of the parent thread
        thread_id_type get_parent_thread_id() const
        {
            detail::thread const* t = get();
            return t ? t->get_parent_thread_id() : 0;
        }

        /// Return the id of the component this thread is running in
        naming::address::address_type get_component_id() const
        {
            detail::thread const* t = get();
            return t ? t->get_component_id() : 0;
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
            detail::thread const* t = get();
            return t ? t->get_state() : thread_state(terminated);
        }

        /// The set_state function allows to change the state of this thread 
        /// instance.
        ///
        /// \param newstate [in] The new state to be set for the thread.
        ///
        /// \note           This function will be seldomly used directly. Most of 
        ///                 the time the state of a thread will have to be 
        ///                 changed using the threadmanager. Moreover,
        ///                 changing the thread state using this function does
        ///                 not change its scheduling status. It only sets the
        ///                 thread's status word. To change the thread's 
        ///                 scheduling status \a threadmanager#set_state should
        ///                 be used.
        thread_state set_state(thread_state_enum new_state)
        {
            detail::thread* t = get();
            return t ? t->set_state(new_state) : thread_state(terminated);
        }

        /// The set_state function allows to change the state of this thread 
        /// instance.
        ///
        /// \param newstate [in] The new state to be set for the thread.
        /// \param new_tagged_state [out] will hold the new fully tagged state
        bool set_state_tagged(thread_state_enum new_state, 
            thread_state& prev_state, thread_state& new_tagged_state)
        {
            detail::thread* t = get();
            return t ? 
                t->set_state_tagged(new_state, prev_state, new_tagged_state) : 
                false;
        }

        /// The restore_state function allows to change the state of this thread 
        /// instance depending on its current state. It will change the state
        /// atomically only if the current state is still the same as passed
        /// as the second parameter. Otherwise it won't touch the thread state
        /// of this instance.
        ///
        /// \param newstate [in] The new state to be set for the thread.
        /// \param oldstate [in] The old state of the thread which still has to
        ///                 be the current state.
        ///
        /// \note           This function will be seldomly used directly. Most of 
        ///                 the time the state of a thread will have to be 
        ///                 changed using the threadmanager. Moreover,
        ///                 changing the thread state using this function does
        ///                 not change its scheduling status. It only sets the
        ///                 thread's status word. To change the thread's 
        ///                 scheduling status \a threadmanager#set_state should
        ///                 be used.
        ///
        /// \returns This function returns \a true if the state has been 
        ///          changed successfully
        bool restore_state(thread_state_enum newstate, thread_state oldstate)
        {
            detail::thread* t = get();
            return t ? t->restore_state(newstate, oldstate) : false;
        }

        /// The get_state_ex function allows to query the extended state of 
        /// this thread instance.
        ///
        /// \returns        This function returns the current extended state of 
        ///                 this thread. It will return one of the values as 
        ///                 defined by the \a thread_state_ex enumeration.
        ///
        /// \note           This function will be seldom used directly. Most of 
        ///                 the time the extended state of a thread will be 
        ///                 retrieved by using the function 
        ///                 \a threadmanager#get_state_ex.
        thread_state_ex get_state_ex() const 
        {
            detail::thread const* t = get();
            return t ? t->get_state_ex() : thread_state_ex(wait_unknown);
        }

        /// The set_state function allows to change the extended state of this 
        /// thread instance.
        ///
        /// \param newstate [in] The new extended state to be set for the 
        ///                 thread.
        ///
        /// \note           This function will be seldom used directly. Most of 
        ///                 the time the state of a thread will have to be 
        ///                 changed using the threadmanager. 
        thread_state_ex set_state_ex(thread_state_ex_enum new_state)
        {
            detail::thread* t = get();
            return t ? t->set_state_ex(new_state) : thread_state_ex(wait_unknown);
        }

        /// \brief Execute the thread function
        ///
        /// \returns        This function returns the thread state the thread
        ///                 should be scheduled from this point on. The thread
        ///                 manager will use the returned value to set the 
        ///                 thread's scheduling status.
        thread_state_enum operator()()
        {
            detail::thread* t = get();
            return t ? t->execute() : thread_state(terminated);
        }

        /// \brief Get the (optional) description of this thread
        char const* const get_description() const
        {
            detail::thread const* t = get();
            return t ? t->get_description() : "<terminated>";
        }

        ///////////////////////////////////////////////////////////////////////
        void set_marked_state(thread_state mark) const
        {
            detail::thread const* t = get();
            if (t) 
                t->set_marked_state(mark);
        }
        thread_state get_marked_state() const
        {
            detail::thread const* t = get();
            return t ? t->get_marked_state() : thread_state(unknown);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    thread_id_type const invalid_thread_id = 0;

}}

#include <hpx/config/warnings_suffix.hpp>

#endif
