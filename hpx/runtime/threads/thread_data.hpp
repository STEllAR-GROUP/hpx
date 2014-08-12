//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PX_THREAD_MAY_20_2008_0910AM)
#define HPX_PX_THREAD_MAY_20_2008_0910AM

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/detail/tagged_thread_state.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/backtrace.hpp>
#include <hpx/util/coroutine/coroutine.hpp>
#include <hpx/util/coroutine/stackless_coroutine.hpp>
#include <hpx/util/lockfree/freelist.hpp>

#include <boost/atomic.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/noncopyable.hpp>
#include <boost/lockfree/detail/branch_hints.hpp>

#include <stack>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    class thread_data;

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // Why do we use std::stack + a lock here?
        template <typename CoroutineImpl>
        struct coroutine_allocator
        {
            typedef lcos::local::spinlock mutex_type;

            coroutine_allocator()
            {}

            CoroutineImpl* get()
            {
                mutex_type::scoped_lock l(mtx_);
                return get_locked();
            }

            CoroutineImpl* try_get()
            {
                mutex_type::scoped_lock l(mtx_, boost::try_to_lock);
                if (!l)
                    return NULL;
                return get_locked();
            }

            void deallocate(CoroutineImpl* c)
            {
                mutex_type::scoped_lock l(mtx_);
                heap_.push(c);
            }

        private:
            CoroutineImpl* get_locked()
            {
                if (heap_.empty())
                    return NULL;

                CoroutineImpl* next = heap_.top();
                heap_.pop();
                return next;
            }

            mutex_type mtx_;
            std::stack<CoroutineImpl*> heap_;
        };

        ///////////////////////////////////////////////////////////////////////
        struct thread_exit_callback_node
        {
            HPX_STD_FUNCTION<void()> f_;
            thread_exit_callback_node* next_;

            thread_exit_callback_node(HPX_STD_FUNCTION<void()> const& f,
                    thread_exit_callback_node* next)
              : f_(f), next_(next)
            {}

            void operator()()
            {
                f_();
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    /// A \a thread is the representation of a ParalleX thread. It's a first
    /// class object in ParalleX. In our implementation this is a user level
    /// thread running on top of one of the OS threads spawned by the \a
    /// thread-manager.
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
    /// implemented by the thread-manager.
    class thread_data_base : private boost::noncopyable
    {
    public:
        typedef thread_function_type function_type;

        struct tag {};
        typedef util::spinlock_pool<tag> mutex_type;

        /// Construct a new \a thread
        thread_data_base(thread_init_data& init_data, thread_state_enum newstate)
          : current_state_(thread_state(newstate)),
            current_state_ex_(thread_state_ex(wait_signaled)),
#if HPX_THREAD_MAINTAIN_TARGET_ADDRESS
            component_id_(init_data.lva),
#endif
#if HPX_THREAD_MAINTAIN_DESCRIPTION
            description_(init_data.description ? init_data.description : ""),
            lco_description_(""),
#endif
#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
            parent_locality_id_(init_data.parent_locality_id),
            parent_thread_id_(init_data.parent_id),
            parent_thread_phase_(init_data.parent_phase),
#endif
#if HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
            marked_state_(unknown),
#endif
#if HPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION
            backtrace_(0),
#endif
            priority_(init_data.priority),
            requested_interrupt_(false),
            enabled_interrupt_(true),
            ran_exit_funcs_(false),
            exit_funcs_(0),
            scheduler_base_(init_data.scheduler_base),
            count_(0),
            stacksize_(init_data.stacksize)
        {
            LTM_(debug) << "thread::thread(" << this << "), description("
                        << get_description() << ")";

#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
            // store the thread id of the parent thread, mainly for debugging
            // purposes
            if (0 == parent_thread_id_) {
                thread_self* self = get_self_ptr();
                if (self)
                {
                    parent_thread_id_ = threads::get_self_id().get();
                    parent_thread_phase_ = self->get_thread_phase();
                }
            }
            if (0 == parent_locality_id_)
                parent_locality_id_ = get_locality_id();
#endif
        }

        void rebind_base(thread_init_data& init_data, thread_state_enum newstate)
        {
            free_thread_exit_callbacks();

            current_state_.store(thread_state(newstate));
            current_state_ex_.store(thread_state_ex(wait_signaled));
#if HPX_THREAD_MAINTAIN_TARGET_ADDRESS
            component_id_ = init_data.lva;
#endif
#if HPX_THREAD_MAINTAIN_DESCRIPTION
            description_ = (init_data.description ? init_data.description : "");
            lco_description_ = "";
#endif
#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
            parent_locality_id_ = init_data.parent_locality_id;
            parent_thread_id_ = init_data.parent_id;
            parent_thread_phase_ = init_data.parent_phase;
#endif
#if HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
            set_marked_state(thread_state(unknown));
#endif
#if HPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION
            backtrace_ = 0;
#endif
            priority_ = init_data.priority;
            requested_interrupt_ = false;
            enabled_interrupt_ = true;
            ran_exit_funcs_ = false;
            exit_funcs_ = 0;
            scheduler_base_ = init_data.scheduler_base;

            HPX_ASSERT(init_data.stacksize == get_stack_size());

            LTM_(debug) << "thread::thread(" << this << "), description("
                        << get_description() << "), rebind";

#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
            // store the thread id of the parent thread, mainly for debugging
            // purposes
            if (0 == parent_thread_id_) {
                thread_self* self = get_self_ptr();
                if (self)
                {
                    parent_thread_id_ = threads::get_self_id().get();
                    parent_thread_phase_ = self->get_thread_phase();
                }
            }
            if (0 == parent_locality_id_)
                parent_locality_id_ = get_locality_id();
#endif
        }

        virtual ~thread_data_base()
        {
            free_thread_exit_callbacks();
        }

        /// The get_state function queries the state of this thread instance.
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
            return current_state_.load(boost::memory_order_acquire);
        }

        // REVIEW: Should this be private, and threadmanager be made a friend of
        // thread_data?
        /// The set_state function changes the state of this thread instance.
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
        thread_state set_state(thread_state_enum newstate)
        {
            thread_state prev_state = current_state_.load(boost::memory_order_acquire);
            for (;;) {
                thread_state tmp = prev_state;

                if (HPX_LIKELY(current_state_.compare_exchange_strong(
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
                return true;

            prev_state = tmp;
            return false;
        }

        /// The restore_state function changes the state of this thread
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
        bool restore_state(thread_state_enum new_state, thread_state old_state)
        {
            return current_state_.compare_exchange_strong(
                old_state, thread_state(new_state, old_state.get_tag() + 1));
        }

        /// The get_state_ex function queries the extended state of this
        /// thread instance.
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
            return current_state_ex_.load(boost::memory_order_acquire);
        }

        /// The set_state function changes the extended state of this
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
            thread_state_ex prev_state =
                current_state_ex_.load(boost::memory_order_acquire);

            for (;;) {
                thread_state_ex tmp = prev_state;

                if (HPX_LIKELY(current_state_ex_.compare_exchange_strong(
                        tmp, thread_state_ex(new_state, tmp.get_tag() + 1))))
                {
                    return prev_state;
                }
                prev_state = tmp;
            }
        }

        /// Return the id of the component this thread is running in
        naming::address::address_type get_component_id() const
        {
#if HPX_THREAD_MAINTAIN_TARGET_ADDRESS == 0
            return 0;
#else
            return component_id_;
#endif
        }

#if HPX_THREAD_MAINTAIN_DESCRIPTION == 0
        char const* get_description() const
        {
            return "<unknown>";
        }
        char const* set_description(char const* /*value*/)
        {
            return "<unknown>";
        }

        char const* get_lco_description() const
        {
            return "<unknown>";
        }
        char const* set_lco_description(char const* /*value*/)
        {
            return "<unknown>";
        }
#else
        char const* get_description() const
        {
            mutex_type::scoped_lock l(this);
            return description_ ? description_ : "<unknown>";
        }
        char const* set_description(char const* value)
        {
            mutex_type::scoped_lock l(this);
            std::swap(description_, value);
            return value;
        }

        char const* get_lco_description() const
        {
            mutex_type::scoped_lock l(this);
            return lco_description_ ? lco_description_ : "<unknown>";
        }
        char const* set_lco_description(char const* value)
        {
            mutex_type::scoped_lock l(this);
            std::swap(lco_description_, value);
            return value;
        }
#endif

#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE == 0
        /// Return the locality of the parent thread
        boost::uint32_t get_parent_locality_id() const
        {
            return naming::invalid_locality_id;
        }

        /// Return the thread id of the parent thread
        thread_id_repr_type get_parent_thread_id() const
        {
            return threads::invalid_thread_id_repr;
        }

        /// Return the phase of the parent thread
        std::size_t get_parent_thread_phase() const
        {
            return 0;
        }
#else
        /// Return the locality of the parent thread
        boost::uint32_t get_parent_locality_id() const
        {
            return parent_locality_id_;
        }

        /// Return the thread id of the parent thread
        thread_id_repr_type get_parent_thread_id() const
        {
            return parent_thread_id_;
        }

        /// Return the phase of the parent thread
        std::size_t get_parent_thread_phase() const
        {
            return parent_thread_phase_;
        }
#endif

#if HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
        void set_marked_state(thread_state mark) const
        {
            marked_state_ = mark;
        }
        thread_state get_marked_state() const
        {
            return marked_state_;
        }
#endif

#if HPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION == 0

# if HPX_THREAD_MAINTAIN_FULLBACKTRACE_ON_SUSPENSION != 0
        char const* get_backtrace() const
        {
            return 0;
        }
        char const* set_backtrace(char const*)
        {
            return 0;
        }
# else
        util::backtrace const* get_backtrace() const
        {
            return 0;
        }
        util::backtrace const* set_backtrace(util::backtrace const*)
        {
            return 0;
        }
# endif

#else  // HPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION == 0

# if HPX_THREAD_MAINTAIN_FULLBACKTRACE_ON_SUSPENSION != 0
        char const* get_backtrace() const
        {
            mutex_type::scoped_lock l(this);
            return backtrace_;
        }
        char const* set_backtrace(char const* value)
        {
            mutex_type::scoped_lock l(this);

            char const* bt = backtrace_;
            backtrace_ = value;
            return bt;
        }
# else
        util::backtrace const* get_backtrace() const
        {
            mutex_type::scoped_lock l(this);
            return backtrace_;
        }
        util::backtrace const* set_backtrace(util::backtrace const* value)
        {
            mutex_type::scoped_lock l(this);

            util::backtrace const* bt = backtrace_;
            backtrace_ = value;
            return bt;
        }
# endif

        // Generate full backtrace for captured stack
        std::string backtrace()
        {
            mutex_type::scoped_lock l(this);
            std::string bt;
            if (0 != backtrace_)
            {
# if HPX_THREAD_MAINTAIN_FULLBACKTRACE_ON_SUSPENSION != 0
                bt = *backtrace_;
#else
                bt = backtrace_->trace();
#endif
            }
            return bt;
        }
#endif

        thread_priority get_priority() const
        {
            return priority_;
        }
        void set_priority(thread_priority priority)
        {
            priority_ = priority;
        }

        // handle thread interruption
        bool interruption_requested() const
        {
            mutex_type::scoped_lock l(this);
            return requested_interrupt_;
        }

        bool interruption_enabled() const
        {
            mutex_type::scoped_lock l(this);
            return enabled_interrupt_;
        }

        bool set_interruption_enabled(bool enable)
        {
            mutex_type::scoped_lock l(this);
            std::swap(enabled_interrupt_, enable);
            return enable;
        }

        void interrupt(bool flag = true)
        {
            mutex_type::scoped_lock l(this);
            if (flag && !enabled_interrupt_) {
                HPX_THROW_EXCEPTION(thread_not_interruptable,
                    "thread_data::interrupt",
                    "interrupts are disabled for this thread");
                return;
            }
            requested_interrupt_ = flag;
        }

        bool interruption_point(bool throw_on_interrupt = true);

        bool add_thread_exit_callback(HPX_STD_FUNCTION<void()> const& f);
        void run_thread_exit_callbacks();
        void free_thread_exit_callbacks();

        policies::scheduler_base* get_scheduler_base() const
        {
            return scheduler_base_;
        }

        std::ptrdiff_t get_stack_size() const
        {
            return stacksize_;
        }

        virtual bool is_created_from(void* pool) const = 0;
        virtual thread_state_enum operator()() = 0;
        virtual thread_id_type get_thread_id() const = 0;
        virtual std::size_t get_thread_phase() const = 0;
#if HPX_THREAD_MAINTAIN_LOCAL_STORAGE
        virtual std::size_t get_thread_data() const = 0;
        virtual std::size_t set_thread_data(std::size_t data) = 0;
#endif

        virtual void rebind(thread_init_data& init_data,
            thread_state_enum newstate) = 0;

        /// This function will be called when the thread is about to be deleted
        //virtual void reset() {}

        friend HPX_EXPORT void intrusive_ptr_add_ref(thread_data_base* p);
        friend HPX_EXPORT void intrusive_ptr_release(thread_data_base* p);

    protected:
        mutable boost::atomic<thread_state> current_state_;
        mutable boost::atomic<thread_state_ex> current_state_ex_;

        ///////////////////////////////////////////////////////////////////////
        // Debugging/logging information
#if HPX_THREAD_MAINTAIN_TARGET_ADDRESS
        naming::address::address_type component_id_;
#endif

#if HPX_THREAD_MAINTAIN_DESCRIPTION
        char const* description_;
        char const* lco_description_;
#endif

#if HPX_THREAD_MAINTAIN_PARENT_REFERENCE
        boost::uint32_t parent_locality_id_;
        thread_id_repr_type parent_thread_id_;
        std::size_t parent_thread_phase_;
#endif

#if HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
        mutable thread_state marked_state_;
#endif

#if HPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION
# if HPX_THREAD_MAINTAIN_FULLBACKTRACE_ON_SUSPENSION != 0
        char const* backtrace_;
# else
        util::backtrace const* backtrace_;
# endif
#endif

        ///////////////////////////////////////////////////////////////////////
        thread_priority priority_;

        bool requested_interrupt_;
        bool enabled_interrupt_;
        bool ran_exit_funcs_;

        // Singly linked list (heap-allocated)
        detail::thread_exit_callback_node* exit_funcs_;

        // reference to scheduler which created/manages this thread
        policies::scheduler_base* scheduler_base_;

        //reference count
        boost::detail::atomic_count count_;

        std::ptrdiff_t stacksize_;
    };

    ///////////////////////////////////////////////////////////////////////////
    class thread_data : public thread_data_base
    {
        // Avoid warning about using 'this' in initializer list
        thread_data_base* this_() { return this; }

    public:
        typedef boost::lockfree::caching_freelist<thread_data> pool_type;

        thread_data(thread_init_data& init_data,
               pool_type& pool, thread_state_enum newstate)
          : thread_data_base(init_data, newstate),
            coroutine_(std::move(init_data.func), std::move(init_data.target),
                this_(), init_data.stacksize),
            pool_(&pool)
        {
            HPX_ASSERT(init_data.stacksize != 0);
            HPX_ASSERT(coroutine_.is_ready());
        }

        ~thread_data()
        {
            LTM_(debug) << "~thread(" << this << "), description(" //-V128
                        << get_description() << "), phase("
                        << get_thread_phase() << ")";
        }

        void rebind(thread_init_data& init_data, thread_state_enum newstate)
        {
            LTM_(debug) << "~thread(" << this << "), description(" //-V128
                        << get_description() << "), phase("
                        << get_thread_phase() << "), rebind";

            this->thread_data_base::rebind_base(init_data, newstate);

            coroutine_.rebind(boost::move(init_data.func),
                boost::move(init_data.target), this_());

            HPX_ASSERT(init_data.stacksize != 0);
            HPX_ASSERT(coroutine_.is_ready());
        }

        ///////////////////////////////////////////////////////////////////////
        // Memory management
        static void* operator new(std::size_t size, pool_type&);
        static void operator delete(void* p, std::size_t size);
        static void operator delete(void*, pool_type&);

        // Won't be called.
        static void* operator new(std::size_t) throw()
        {
            HPX_ASSERT(false);
            return NULL;
        }

        bool is_created_from(void* pool) const
        {
            return pool_ == pool;
        }

        /// \brief Execute the thread function
        ///
        /// \returns        This function returns the thread state the thread
        ///                 should be scheduled from this point on. The thread
        ///                 manager will use the returned value to set the
        ///                 thread's scheduling status.
        thread_state_enum operator()()
        {
            thread_state_ex current_state_ex = get_state_ex();
            current_state_ex_.store(thread_state_ex(wait_signaled,
                current_state_ex.get_tag() + 1), boost::memory_order_release);

            HPX_ASSERT(this_() == coroutine_.get_thread_id());

            return coroutine_(current_state_ex);
        }

        thread_id_type get_thread_id() const
        {
            return thread_id_type(
                    reinterpret_cast<thread_data_base*>(coroutine_.get_thread_id())
                );
        }

        std::size_t get_thread_phase() const
        {
#if HPX_THREAD_MAINTAIN_PHASE_INFORMATION == 0
            return 0;
#else
            return coroutine_.get_thread_phase();
#endif
        }

#if HPX_THREAD_MAINTAIN_LOCAL_STORAGE
        std::size_t get_thread_data() const
        {
            return coroutine_.get_thread_data();
        }
        std::size_t set_thread_data(std::size_t data)
        {
            return coroutine_.set_thread_data(data);
        }
#endif

        /// This function will be called when the thread is about to be deleted
        //void reset()
        //{
        //    thread_data_base::reset();
        //    coroutine_.reset();
        //}

    private:
        coroutine_type coroutine_;
        pool_type* pool_;
    };

    typedef thread_data::pool_type thread_pool;

    ///////////////////////////////////////////////////////////////////////////
    class stackless_thread_data : public thread_data_base
    {
        // Avoid warning about using 'this' in initializer list
        thread_data_base* this_() { return this; }

    public:
        stackless_thread_data(thread_init_data& init_data,
                void* pool, thread_state_enum newstate)
          : thread_data_base(init_data, newstate),
            coroutine_(std::move(init_data.func), std::move(init_data.target), this_()),
            pool_(pool)
        {
            HPX_ASSERT(init_data.stacksize == 0);
        }

        ~stackless_thread_data()
        {
            LTM_(debug) << "~stackless_thread(" << this << "), description(" //-V128
                        << get_description() << "), phase("
                        << get_thread_phase() << ")";
        }

        void rebind(thread_init_data& init_data, thread_state_enum newstate)
        {
            LTM_(debug) << "~stackless_thread(" << this << "), description(" //-V128
                        << get_description() << "), phase("
                        << get_thread_phase() << "), rebind";

            this->thread_data_base::rebind_base(init_data, newstate);

            coroutine_.rebind(boost::move(init_data.func),
                boost::move(init_data.target), this_());

            HPX_ASSERT(init_data.stacksize == 0);
        }

        ///////////////////////////////////////////////////////////////////////
        // Memory management
        bool is_created_from(void* pool) const
        {
            return pool_ == pool;
        }

        /// \brief Execute the thread function
        ///
        /// \returns        This function returns the thread state the thread
        ///                 should be scheduled from this point on. The thread
        ///                 manager will use the returned value to set the
        ///                 thread's scheduling status.
        thread_state_enum operator()()
        {
            thread_state_ex current_state_ex = get_state_ex();
            current_state_ex_.store(thread_state_ex(wait_signaled,
                current_state_ex.get_tag() + 1), boost::memory_order_release);

            return coroutine_(current_state_ex);
        }

        thread_id_type get_thread_id() const
        {
            return thread_id_type(
                    reinterpret_cast<thread_data_base*>(coroutine_.get_thread_id())
                );
        }

        std::size_t get_thread_phase() const
        {
#if HPX_THREAD_MAINTAIN_PHASE_INFORMATION == 0
            return 0;
#else
            return coroutine_.get_thread_phase();
#endif
        }

#if HPX_THREAD_MAINTAIN_LOCAL_STORAGE
        std::size_t get_thread_data() const
        {
            return coroutine_.get_thread_data();
        }
        std::size_t set_thread_data(std::size_t data)
        {
            return coroutine_.set_thread_data(data);
        }
#endif

        /// This function will be called when the thread is about to be deleted
        //void reset()
        //{
        //    thread_data_base::reset();
        //    coroutine_.reset();
        //}

    private:
        typedef util::coroutines::stackless_coroutine<
            thread_function_sig
        > coroutine_type;

        coroutine_type coroutine_;
        void* pool_;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
