//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_FULLEMPTYSTORE_JUN_16_2008_0128APM)
#define HPX_LCOS_FULLEMPTYSTORE_JUN_16_2008_0128APM

#include <set>
#include <queue>
#include <memory>

#include <boost/thread.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/intrusive/slist.hpp>

#include <hpx/runtime/threadmanager/px_thread.hpp>
#include <hpx/runtime/threadmanager/threadmanager.hpp>
#include <hpx/util/unlock_lock.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    enum full_empty
    {
        empty = false,
        full = true
    };

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
        template <typename Lock>
        bool set_empty(Lock& outer_lock) 
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();
            return set_empty_locked();
        }

        // sets this entry to full
        template <typename Lock>
        bool set_full(Lock& outer_lock) 
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();
            return set_full_locked();
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
                full_empty_queue_entry* f = 
                    new full_empty_queue_entry(self.get_thread_id());
                full_full_.push_back(*f);

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
                full_empty_queue_entry* f = 
                    new full_empty_queue_entry(self.get_thread_id());
                full_full_.push_back(*f);

                util::unlock_the_lock<scoped_lock> ul(l);
                self.yield(threadmanager::suspended);
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue a get operation in full/empty queue if entry is empty
        template <typename Lock, typename T>
        void enqueue_full_empty(Lock& outer_lock, 
            threadmanager::px_thread_self& self, T& dest)
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();

            // block if this entry is empty
            if (state_ == empty) {
                // enqueue the request and block this thread
                full_empty_queue_entry* f = 
                    new full_empty_queue_entry(self.get_thread_id());
                full_empty_.push_back(*f);

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
                set_empty_locked();   // state_ = empty;
            }
        }

        // same as above, but for entries without associated data
        template <typename Lock>
        void enqueue_full_empty(Lock& outer_lock, 
            threadmanager::px_thread_self& self)
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();

            // block if this entry is empty
            if (state_ == empty) {
                // enqueue the request and block this thread
                full_empty_queue_entry* f = 
                    new full_empty_queue_entry(self.get_thread_id());
                full_empty_.push_back(*f);

                // yield this thread
                util::unlock_the_lock<scoped_lock> ul(l);
                self.yield(threadmanager::suspended);
            }
            else {
                set_empty_locked();   // state_ = empty
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue if entry is full, otherwise fill it
        template <typename Lock, typename T>
        void enqueue_if_full(Lock& outer_lock, 
            threadmanager::px_thread_self& self, T const& src)
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();

            // block if this entry is already full
            if (state_ == full) {
                // enqueue the request and block this thread
                full_empty_queue_entry* f = 
                    new full_empty_queue_entry(self.get_thread_id());
                empty_full_.push_back(*f);

                util::unlock_the_lock<scoped_lock> ul(l);
                self.yield(threadmanager::suspended);
            }

            // set the data
            if (entry_ && entry_ != &src) 
                *static_cast<T*>(entry_) = src;

            // make sure the entry is full
            set_full_locked();    // state_ = full
        }

        // same as above, but for entries without associated data
        template <typename Lock>
        void enqueue_if_full(Lock& outer_lock, 
            threadmanager::px_thread_self& self)
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();

            // block if this entry is already full
            if (state_ == full) {
                // enqueue the request and block this thread
                full_empty_queue_entry* f = 
                    new full_empty_queue_entry(self.get_thread_id());
                empty_full_.push_back(*f);

                util::unlock_the_lock<scoped_lock> ul(l);
                self.yield(threadmanager::suspended);
            }

            // make sure the entry is full
            set_full_locked();    // state_ = full
        }

        ///////////////////////////////////////////////////////////////////////
        // unconditionally set the data and set the entry to full
        template <typename Lock, typename T>
        void set_and_fill(Lock& outer_lock, T const& src)
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();

            // set the data
            if (entry_ && entry_ != &src) 
                *static_cast<T*>(entry_) = src;

            // make sure the entry is full
            set_full_locked();    // state_ = full
        }

        // same as above, but for entries without associated data
        template <typename Lock>
        void set_and_fill(Lock& outer_lock)
        {
            scoped_lock l(mtx_);
            outer_lock.unlock();

            // make sure the entry is full
            set_full_locked();    // state_ = full
        }

        // returns whether this entry is still in use
        bool is_used() const
        {
            scoped_lock l(mtx_);
            return is_used_locked();
        }

    protected:
        bool set_empty_locked()
        {
            state_ = empty;
            if (!empty_full_.empty()) {
                std::auto_ptr<full_empty_queue_entry> f (&full_empty_.front());
                empty_full_.pop_front();
                threadmanager::set_thread_state(f->id_, threadmanager::pending);
                set_full_locked();    // state_ = full
            }

            // return whether this block needs to be removed
            return state_ == full && !is_used_locked();
        }

        bool set_full_locked()
        {
            state_ = full;

            // handle all threads waiting for the block to become full
            while (!full_full_.empty()) {
                std::auto_ptr<full_empty_queue_entry> f (&full_empty_.front());
                full_full_.pop_front();
                threadmanager::set_thread_state(f->id_, threadmanager::pending);
            }

            // since we got full now we need to re-activate one thread waiting
            // for the block to become full
            if (!full_empty_.empty()) {
                std::auto_ptr<full_empty_queue_entry> f (&full_empty_.front());
                full_empty_.pop_front();
                threadmanager::set_thread_state(f->id_, threadmanager::pending);
                set_empty_locked();   // state_ = empty
            }

            // return whether this block needs to be removed
            return state_ == full && !is_used_locked();
        }

        bool is_used_locked() const
        {
            return !(empty_full_.empty() && full_empty_.empty() && full_full_.empty());
        }

    private:
        mutable mutex_type mtx_;
        queue_type empty_full_;     // threads waiting in write
        queue_type full_empty_;     // threads waiting in read_and_empty
        queue_type full_full_;      // threads waiting in read
        void* entry_;
        full_empty state_;
    };

    ///////////////////////////////////////////////////////////////////////////
    class full_empty_store
    {
    private:
        typedef boost::shared_mutex mutex_type;
        typedef boost::ptr_map<void*, full_empty_entry> store_type;

    public:
        full_empty_store()
        {}

        ///
        void set_empty(void* entry)
        {
            boost::unique_lock<mutex_type> l(mtx_);
            store_type::iterator it = store_.find(entry);
            if (it == store_.end()) {
            // create a new entry for this unknown (full) block 
                std::pair<store_type::iterator, bool> p = store_.insert(
                    entry, new full_empty_entry(entry));
                if (!p.second) {
                    boost::throw_exception(std::bad_alloc());
                    return;
                }
                (*p.first).second->set_empty(l);
            }
            else {
                // set the entry to empty state if it's not newly created
                if ((*it).second->set_empty(l)) 
                    remove(entry);    // remove if no more threads are waiting
            }
        }

        ///
        void set_full(void* entry)
        {
            store_type::iterator it;
            {
                boost::unique_lock<mutex_type> l(mtx_);
                store_type::iterator it = store_.find(entry);
                if (it == store_.end()) 
                    return;       // entry doesn't exist

                // set the entry to full state
                if ((*it).second->set_full(l))
                    remove(entry);    // remove if no more threads are waiting
            }
        }

        /// The function \a remove erases the given entry from the store if it
        /// is not used by any other thread anymore.
        ///
        /// \returns  This returns \a true if the entry has been removed from
        ///           the store, otherwise it returns \a false, which either 
        ///           means that the entry is unknown to the store or that the 
        ///           entry is still in use (other threads are waiting on this
        ///           entry).
        bool remove(void*entry)
        {
            boost::unique_lock<mutex_type> l(mtx_);
            store_type::iterator it = store_.find(entry);
            if (it != store_.end() && !(*it).second->is_used()) {
                store_.erase(it);
                return true;
            }
            return false;
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

    protected:
        store_type::iterator find_or_create(void* entry)
        {
            store_type::iterator it = store_.find(entry);
            if (it == store_.end()) {
                // create a new entry for this unknown (full) block 
                std::pair<store_type::iterator, bool> p = store_.insert(
                    entry, new full_empty_entry(entry));
                if (!p.second) {
                    boost::throw_exception(std::bad_alloc());
                    return store_type::iterator();
                }
                it = p.first;
            }
            return it;
        }

    public:
        /// \brief Wait for the memory to become full and then reads it, leaves 
        /// memory in full state.
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
        template <typename T>
        void read_and_empty(threadmanager::px_thread_self& self, void* entry, 
            T& dest)
        {
            boost::shared_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(entry);
            (*it).second->enqueue_full_empty(l, self, dest);
        }

        void read_and_empty(threadmanager::px_thread_self& self, void* entry)
        {
            boost::shared_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(entry);
            (*it).second->enqueue_full_empty(l, self);
        }

        /// \brief Writes memory and atomically sets its state to full without 
        /// waiting for it to become empty.
        template <typename T>
        void set(void* entry, T const& src)
        {
            boost::unique_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(entry);
            (*it).second->set_and_fill(l, src);
        }

        void set(void* entry)
        {
            boost::unique_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(entry);
            (*it).second->set_and_fill(l);
        }

        /// \brief Wait for memory to become empty, and then fill it.
        template <typename T>
        void write(threadmanager::px_thread_self& self, void* entry, 
            T const& src)
        {
            boost::unique_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(entry);
            (*it).second->enqueue_if_full(l, self, src);
        }

        void write(threadmanager::px_thread_self& self, void* entry)
        {
            boost::unique_lock<mutex_type> l(mtx_);
            store_type::iterator it = find_or_create(entry);
            (*it).second->enqueue_if_full(l, self);
        }

    private:
        mutable mutex_type mtx_;
        store_type store_;
    };
    
}}}

#endif

