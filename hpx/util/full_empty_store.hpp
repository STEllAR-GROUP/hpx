//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_FULLEMPTYSTORE_JUN_16_2008_0128APM)
#define HPX_UTIL_FULLEMPTYSTORE_JUN_16_2008_0128APM

#include <set>
#include <queue>
#include <memory>

#include <boost/thread.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/intrusive/slist.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/unlock_lock.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    enum full_empty
    {
        empty = false,
        full = true
    };

    ///////////////////////////////////////////////////////////////////////////
    struct no_full_empty_action_type
    {
        full_empty operator()(full_empty state) const
        {
            return state;
        }
    };
    no_full_empty_action_type const no_full_empty_action = 
        no_full_empty_action_type();

    ///////////////////////////////////////////////////////////////////////////
    class full_empty_entry
    {
    private:
        typedef threadmanager::thread_id_type thread_id_type;

        // define data structures needed for intrusive slist container used for
        // the queues
        struct full_empty_queue_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>
            > hook_type;

            full_empty_queue_entry(thread_id_type id)
              : id_(id)
            {}

            thread_id_type id_;
            hook_type slist_hook_;
        };

        typedef boost::intrusive::member_hook<
            full_empty_queue_entry, full_empty_queue_entry::hook_type,
            &full_empty_queue_entry::slist_hook_
        > slist_option_type;

        typedef boost::intrusive::slist<
            full_empty_queue_entry, slist_option_type, 
            boost::intrusive::cache_last<true>, 
            boost::intrusive::constant_time_size<false>
        > queue_type;

    public:
        typedef boost::mutex mutex_type;
        typedef mutex_type::scoped_lock scoped_lock;

        explicit full_empty_entry(void* entry)
          : entry_(entry), state_(full)
        {}

        ~full_empty_entry()
        {
        }

        // returns whether this entry is currently empty
        template <typename Lock>
        bool is_empty(Lock& outer_lock) const
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();
            return state_ == empty;
        }

        // sets this entry to empty
        template <typename Lock, typename Action>
        bool set_empty(Lock& outer_lock, Action const& f) 
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();
            return set_empty_locked(f);
        }

        // sets this entry to full
        template <typename Lock, typename Action>
        bool set_full(Lock& outer_lock, Action const& f) 
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();
            return set_full_locked(f);
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue a get operation if full/full queue if entry is empty
        template <typename Lock, typename T>
        void enqueue_full_full(Lock& outer_lock, 
            threadmanager::px_thread_self& self, T& dest)
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();

            // block if this entry is empty
            if (state_ == empty) {
                // enqueue the request and block this thread
                full_empty_queue_entry f(self.get_thread_id());
                read_queue_.push_back(f);

                util::unlock_the_lock<scoped_lock> ul(l);
                self.yield(threadmanager::suspended);
            }

            // copy the data to the destination
            if (entry_ && entry_ != &dest) 
                dest = *static_cast<T const*>(entry_);
        }

        // same as above, but for entries without associated data
        template <typename Lock>
        void enqueue_full_full(Lock& outer_lock, 
            threadmanager::px_thread_self& self)
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();

            // block if this entry is empty
            if (state_ == empty) {
                // enqueue the request and block this thread
                full_empty_queue_entry f(self.get_thread_id());
                read_queue_.push_back(f);

                util::unlock_the_lock<scoped_lock> ul(l);
                self.yield(threadmanager::suspended);
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue a get operation in full/empty queue if entry is empty
        template <typename Lock, typename T, typename Action>
        void enqueue_full_empty(Lock& outer_lock, 
            threadmanager::px_thread_self& self, T& dest, Action const& f)
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();

            // block if this entry is empty
            if (state_ == empty) {
                // enqueue the request and block this thread
                full_empty_queue_entry f(self.get_thread_id());
                read_and_empty_queue_.push_back(f);

                // yield this thread
                {
                    util::unlock_the_lock<scoped_lock> ul(l);
                    self.yield(threadmanager::suspended);
                }

                // copy the data to the destination
                if (entry_ && entry_ != &dest) 
                    dest = *static_cast<T const*>(entry_);
            }
            else {
                // copy the data to the destination
                if (entry_ && entry_ != &dest) 
                    dest = *static_cast<T const*>(entry_);
                set_empty_locked(f);   // state_ = empty;
            }
        }

        // same as above, but for entries without associated data
        template <typename Lock, typename Action>
        void enqueue_full_empty(Lock& outer_lock, 
            threadmanager::px_thread_self& self, Action const& f)
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();

            // block if this entry is empty
            if (state_ == empty) {
                // enqueue the request and block this thread
                full_empty_queue_entry f(self.get_thread_id());
                read_and_empty_queue_.push_back(f);

                // yield this thread
                util::unlock_the_lock<scoped_lock> ul(l);
                self.yield(threadmanager::suspended);
            }
            else {
                set_empty_locked(f);   // state_ = empty
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue if entry is full, otherwise fill it
        template <typename Lock, typename T, typename Action>
        void enqueue_if_full(Lock& outer_lock, 
            threadmanager::px_thread_self& self, T const& src, Action const& f)
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();

            // block if this entry is already full
            if (state_ == full) {
                // enqueue the request and block this thread
                full_empty_queue_entry f(self.get_thread_id());
                write_queue_.push_back(f);

                util::unlock_the_lock<scoped_lock> ul(l);
                self.yield(threadmanager::suspended);
            }

            // set the data
            if (entry_ && entry_ != &src) 
                *static_cast<T*>(entry_) = src;

            // make sure the entry is full
            set_full_locked(f);    // state_ = full
        }

        // same as above, but for entries without associated data
        template <typename Lock, typename Action>
        void enqueue_if_full(Lock& outer_lock, 
            threadmanager::px_thread_self& self, Action const& f)
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();

            // block if this entry is already full
            if (state_ == full) {
                // enqueue the request and block this thread
                full_empty_queue_entry f(self.get_thread_id());
                write_queue_.push_back(f);

                util::unlock_the_lock<scoped_lock> ul(l);
                self.yield(threadmanager::suspended);
            }

            // make sure the entry is full
            set_full_locked(f);    // state_ = full
        }

        ///////////////////////////////////////////////////////////////////////
        // unconditionally set the data and set the entry to full
        template <typename Lock, typename T, typename Action>
        void set_and_fill(Lock& outer_lock, T const& src, Action const& f)
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();

            // set the data
            if (entry_ && entry_ != &src) 
                *static_cast<T*>(entry_) = src;

            // make sure the entry is full
            set_full_locked(f);    // state_ = full
        }

        // same as above, but for entries without associated data
        template <typename Lock, typename Action>
        void set_and_fill(Lock& outer_lock, Action const& f)
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();

            // make sure the entry is full
            set_full_locked(f);    // state_ = full
        }

        // returns whether this entry is still in use
        bool is_used() const
        {
            scoped_lock l(mtx_);
            return is_used_locked();
        }

    protected:
        template <typename Action>
        bool set_empty_locked(Action const& f)
        {
            state_ = f(empty);
            if (state_ != empty)
                return !is_used_locked();  // callback routine prevented state change

            if (!write_queue_.empty()) {
                full_empty_queue_entry& e (write_queue_.front());
                write_queue_.pop_front();
                threadmanager::set_thread_state(e.id_, threadmanager::pending);
                set_full_locked(f);    // state_ = full
            }

            // return whether this block needs to be removed
            return state_ == full && !is_used_locked();
        }

        template <typename Action>
        bool set_full_locked(Action const& f)
        {
            state_ = f(full);
            if (state_ != full)
                return false;         // callback routine prevented state change

            // handle all threads waiting for the block to become full
            while (!read_queue_.empty()) {
                full_empty_queue_entry& e(read_queue_.front());
                read_queue_.pop_front();
                threadmanager::set_thread_state(e.id_, threadmanager::pending);
            }

            // since we got full now we need to re-activate one thread waiting
            // for the block to become full
            if (!read_and_empty_queue_.empty()) {
                full_empty_queue_entry& e(read_and_empty_queue_.front());
                read_and_empty_queue_.pop_front();
                threadmanager::set_thread_state(e.id_, threadmanager::pending);
                set_empty_locked(f);   // state_ = empty
            }

            // return whether this block needs to be removed
            return state_ == full && !is_used_locked();
        }

        bool is_used_locked() const
        {
            return !(write_queue_.empty() && read_and_empty_queue_.empty() && read_queue_.empty());
        }

    private:
        mutable mutex_type mtx_;
        queue_type write_queue_;              // threads waiting in write
        queue_type read_and_empty_queue_;     // threads waiting in read_and_empty
        queue_type read_queue_;               // threads waiting in read
        void* entry_;                         // pointer to protected data item
        full_empty state_;                    // current full/empty state
    };

    ///////////////////////////////////////////////////////////////////////////
    class full_empty_store
    {
    private:
        typedef boost::shared_mutex mutex_type;
        typedef boost::ptr_map<void*, full_empty_entry> store_type;

    protected:
        template <typename Lock>
        store_type::iterator create(Lock& l, void* entry)
        {
            boost::upgrade_to_unique_lock<mutex_type> ul(l);
            std::pair<store_type::iterator, bool> p = 
                store_.insert(entry, new full_empty_entry(entry));
            if (!p.second) {
                boost::throw_exception(std::bad_alloc());
                return store_type::iterator();
            }
            return p.first;
        }

        template <typename Lock>
        store_type::iterator find_or_create(Lock& l, void* entry)
        {
            // create a new entry for this unknown (full) block 
            store_type::iterator it = store_.find(entry);
            if (it == store_.end()) 
                return create(l, entry);
            return it;
        }

        template <typename Lock>
        bool remove(Lock& l, void* entry)
        {
            store_type::iterator it = store_.find(entry);
            if (it != store_.end() && !(*it).second->is_used()) {
                boost::upgrade_to_unique_lock<mutex_type> ul(l);
                store_.erase(it);
                return true;
            }
            return false;
        }

    public:
        full_empty_store()
        {}

        ///
        template <typename Action>
        void set_empty(Action const& f, void* entry)
        {
            boost::upgrade_lock<mutex_type> l(mtx_);
            store_type::iterator it = store_.find(entry);
            if (it == store_.end()) {
                // create a new entry for this unknown (full) block 
                it = create(l, entry);
                (*it).second->set_empty(l, f);
            }
            else {
                // set the entry to empty state if it's not newly created
                if ((*it).second->set_empty(l, f)) 
                    remove(entry);    // remove if no more threads are waiting
            }
        }
        void set_empty(void* entry)
        {
            boost::upgrade_lock<mutex_type> l(mtx_);
            store_type::iterator it = store_.find(entry);
            if (it == store_.end()) {
                // create a new entry for this unknown (full) block 
                it = create(l, entry);
                (*it).second->set_empty(l, no_full_empty_action);
            }
            else {
                // set the entry to empty state if it's not newly created
                if ((*it).second->set_empty(l, no_full_empty_action)) 
                    remove(entry);    // remove if no more threads are waiting
            }
        }

        ///
        template <typename Action>
        void set_full(Action const& f, void* entry)
        {
            boost::shared_lock<mutex_type> l(mtx_);
            store_type::iterator it = store_.find(entry);
            if (it != store_.end() && (*it).second->set_full(l, f)) 
                remove(entry);    // remove if no more threads are waiting
        }
        void set_full(void* entry)
        {
            boost::shared_lock<mutex_type> l(mtx_);
            store_type::iterator it = store_.find(entry);
            if (it != store_.end() && 
                (*it).second->set_full(l, no_full_empty_action)) 
            {
                remove(entry);    // remove if no more threads are waiting
            }
        }

        /// The function \a is_empty returns whether the given memory block is 
        /// currently known to be empty.
        ///
        /// \param    [in]
        ///
        /// \returns  This returns \a true if the referenced memory block is 
        ///           empty, otherwise it returns \a false. If nothing is known
        ///           about the given address this function returns \a false
        ///           as well (all memory is full by default)
        bool is_empty(void const* entry) const
        {
            boost::shared_lock<mutex_type> l(mtx_);
            store_type::const_iterator it = 
                store_.find(const_cast<void*>(entry));
            if (it != store_.end()) 
                return (*it).second->is_empty(l);

            return false;   // by default all memory blocks are full
        }

        ///
        bool is_used(void* entry) const
        {
            boost::shared_lock<mutex_type> l(mtx_);
            store_type::const_iterator it = 
                store_.find(const_cast<void*>(entry));
            if (it != store_.end()) 
                return (*it).second->is_used();

            return false;   // not existing entries are not used by any thread
        }

        /// The function \a remove erases the given entry from the store if it
        /// is not used by any other thread anymore.
        ///
        /// \returns  This returns \a true if the entry has been removed from
        ///           the store, otherwise it returns \a false, which either 
        ///           means that the entry is unknown to the store or that the 
        ///           entry is still in use (other threads are waiting on this
        ///           entry).
        bool remove(void* entry)
        {
            boost::upgrade_lock<mutex_type> l(mtx_);
            return remove(l, entry);
        }

    public:
        /// \brief  Waits for the memory to become full and then reads it, 
        ///         leaves memory in full state.
        template <typename T>
        void read(threadmanager::px_thread_self& self, void* entry, T& dest) 
        {
            boost::shared_lock<mutex_type> l(mtx_);
            store_type::iterator it = store_.find(entry);
            if (it == store_.end()) {
            // just copy the data to the destination
                if (entry && entry != &dest)
                    dest = *static_cast<T const*>(entry);
            }
            else {
                (*it).second->enqueue_full_full(l, self, dest);
            }
        }

        void read(threadmanager::px_thread_self& self, void* entry) 
        {
            boost::shared_lock<mutex_type> l(mtx_);
            store_type::iterator it = store_.find(entry);
            if (it != store_.end()) 
                (*it).second->enqueue_full_full(l, self);
        }

        /// \brief Wait for memory to become full and then reads it, sets 
        /// memory to empty.
        template <typename Action, typename T>
        void read_and_empty(Action const& f, threadmanager::px_thread_self& self, 
            void* entry, T& dest)
        {
            boost::upgrade_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(l, entry);
            (*it).second->enqueue_full_empty(l, self, dest, f);
        }
        template <typename T>
        void read_and_empty(threadmanager::px_thread_self& self, void* entry, 
            T& dest)
        {
            boost::upgrade_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(l, entry);
            (*it).second->enqueue_full_empty(l, self, dest, no_full_empty_action);
        }

        template <typename Action>
        void read_and_empty(Action const& f, threadmanager::px_thread_self& self, 
            void* entry)
        {
            boost::upgrade_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(l, entry);
            (*it).second->enqueue_full_empty(l, self, f);
        }
        void read_and_empty(threadmanager::px_thread_self& self, void* entry)
        {
            boost::upgrade_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(l, entry);
            (*it).second->enqueue_full_empty(l, self, no_full_empty_action);
        }

        /// \brief Writes memory and atomically sets its state to full without 
        /// waiting for it to become empty.
        template <typename Action, typename T>
        void set(Action const& f, void* entry, T const& src)
        {
            boost::upgrade_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(l, entry);
            (*it).second->set_and_fill(l, src, f);
        }
        template <typename T>
        void set(void* entry, T const& src)
        {
            boost::upgrade_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(l, entry);
            (*it).second->set_and_fill(l, src, no_full_empty_action);
        }


        template <typename Action>
        void set(Action const& f, void* entry)
        {
            boost::upgrade_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(l, entry);
            (*it).second->set_and_fill(l, f);
        }
        void set(void* entry)
        {
            boost::upgrade_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(l, entry);
            (*it).second->set_and_fill(l, no_full_empty_action);
        }

        /// \brief Wait for memory to become empty, and then fill it.
        template <typename Action, typename T>
        void write(threadmanager::px_thread_self& self, Action const& f, 
            void* entry, T const& src)
        {
            boost::upgrade_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(l, entry);
            (*it).second->enqueue_if_full(l, self, src, f);
        }
        template <typename T>
        void write(threadmanager::px_thread_self& self, void* entry, 
            T const& src)
        {
            boost::upgrade_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(l, entry);
            (*it).second->enqueue_if_full(l, self, src, no_full_empty_action);
        }

        template <typename Action>
        void write(threadmanager::px_thread_self& self, Action const& f, 
            void* entry)
        {
            boost::upgrade_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(l, entry);
            (*it).second->enqueue_if_full(l, self, f);
        }
        void write(threadmanager::px_thread_self& self, void* entry)
        {
            boost::upgrade_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(l, entry);
            (*it).second->enqueue_if_full(l, self, no_full_empty_action);
        }

    private:
        mutable mutex_type mtx_;
        store_type store_;
    };

}}}

#endif

