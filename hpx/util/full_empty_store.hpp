//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_FULLEMPTYSTORE_JUN_16_2008_0128APM)
#define HPX_UTIL_FULLEMPTYSTORE_JUN_16_2008_0128APM

#include <memory>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <boost/thread.hpp>
#include <boost/aligned_storage.hpp>
#include <boost/type_traits/alignment_of.hpp>
#include <boost/type_traits/add_pointer.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/intrusive/slist.hpp>

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
    template <typename Data>
    class full_empty_entry
    {
    private:
        typedef threads::thread_id_type thread_id_type;

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
            full_empty_queue_entry, 
            typename full_empty_queue_entry::hook_type,
            &full_empty_queue_entry::slist_hook_
        > slist_option_type;

        typedef boost::intrusive::slist<
            full_empty_queue_entry, slist_option_type, 
            boost::intrusive::cache_last<true>, 
            boost::intrusive::constant_time_size<false>
        > queue_type;

    public:
        typedef boost::mutex mutex_type;
        typedef typename mutex_type::scoped_lock scoped_lock;

        full_empty_entry()
          : state_(empty)
        {
            ::new (get_address()) Data();      // properly initialize memory
        }

        template <typename T0>
        explicit full_empty_entry(T0 const& t0)
          : state_(empty)
        {
            ::new (get_address()) Data(t0);    // properly initialize memory
        }

        ~full_empty_entry()
        {
            BOOST_ASSERT(!is_used());
            get_address()->Data::~Data();      // properly destruct value in memory
        }

        // returns whether this entry is currently empty
        bool is_empty() const
        {
            scoped_lock l(mtx_);
            return state_ == empty;
        }

        // sets this entry to empty
        bool set_empty() 
        {
            scoped_lock l(mtx_);
            return set_empty_locked();
        }

        // sets this entry to full
        bool set_full() 
        {
            scoped_lock l(mtx_);
            return set_full_locked();
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue a get operation if full/full queue if entry is empty
        template <typename T>
        void enqueue_full_full(T& dest)
        {
            scoped_lock l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                // enqueue the request and block this thread
                threads::thread_self& self = threads::get_self();
                full_empty_queue_entry f(self.get_thread_id());
                read_queue_.push_back(f);

                util::unlock_the_lock<scoped_lock> ul(l);
                self.yield(threads::suspended);
            }

            // copy the data to the destination
            if (get_address() != &dest) 
                dest = *get_address();
        }

        // same as above, but for entries without associated data
        void enqueue_full_full()
        {
            scoped_lock l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                // enqueue the request and block this thread
                threads::thread_self& self = threads::get_self();
                full_empty_queue_entry f(self.get_thread_id());
                read_queue_.push_back(f);

                util::unlock_the_lock<scoped_lock> ul(l);
                self.yield(threads::suspended);
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue a get operation in full/empty queue if entry is empty
        template <typename T>
        void enqueue_full_empty(T& dest)
        {
            scoped_lock l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                // enqueue the request and block this thread
                threads::thread_self& self = threads::get_self();
                full_empty_queue_entry f(self.get_thread_id());
                read_and_empty_queue_.push_back(f);

                // yield this thread
                {
                    util::unlock_the_lock<scoped_lock> ul(l);
                    self.yield(threads::suspended);
                }

                // copy the data to the destination
                if (get_address() != &dest) 
                    dest = *get_address();
            }
            else {
                // copy the data to the destination
                if (get_address() != &dest) 
                    dest = *get_address();
                set_empty_locked();   // state_ = empty;
            }
        }

        // same as above, but for entries without associated data
        void enqueue_full_empty()
        {
            scoped_lock l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                // enqueue the request and block this thread
                threads::thread_self& self = threads::get_self();
                full_empty_queue_entry f(self.get_thread_id());
                read_and_empty_queue_.push_back(f);

                // yield this thread
                util::unlock_the_lock<scoped_lock> ul(l);
                self.yield(threads::suspended);
            }
            else {
                set_empty_locked();   // state_ = empty
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue if entry is full, otherwise fill it
        template <typename T>
        void enqueue_if_full(T const& src)
        {
            scoped_lock l(mtx_);

            // block if this entry is already full
            if (state_ == full) {
                // enqueue the request and block this thread
                threads::thread_self& self = threads::get_self();
                full_empty_queue_entry f(self.get_thread_id());
                write_queue_.push_back(f);

                util::unlock_the_lock<scoped_lock> ul(l);
                self.yield(threads::suspended);
            }

            // set the data
            if (get_address() != &src) 
                *get_address() = src;

            // make sure the entry is full
            set_full_locked();    // state_ = full
        }

        // same as above, but for entries without associated data
        void enqueue_if_full()
        {
            scoped_lock l(mtx_);

            // block if this entry is already full
            if (state_ == full) {
                // enqueue the request and block this thread
                threads::thread_self& self = threads::get_self();
                full_empty_queue_entry f(self.get_thread_id());
                write_queue_.push_back(f);

                util::unlock_the_lock<scoped_lock> ul(l);
                self.yield(threads::suspended);
            }

            // make sure the entry is full
            set_full_locked();    // state_ = full
        }

        ///////////////////////////////////////////////////////////////////////
        // unconditionally set the data and set the entry to full
        template <typename T>
        void set_and_fill(T const& src)
        {
            scoped_lock l(mtx_);

            // set the data
            if (get_address() != &src) 
                *get_address() = src;

            // make sure the entry is full
            set_full_locked();    // state_ = full
        }

        // same as above, but for entries without associated data
        void set_and_fill()
        {
            scoped_lock l(mtx_);

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

            if (!write_queue_.empty()) {
                full_empty_queue_entry& e (write_queue_.front());
                write_queue_.pop_front();
                threads::set_thread_state(e.id_, threads::pending);
                set_full_locked();    // state_ = full
            }

            // return whether this block needs to be removed
            return state_ == full && !is_used_locked();
        }

        bool set_full_locked()
        {
            state_ = full;

            // handle all threads waiting for the block to become full
            while (!read_queue_.empty()) {
                full_empty_queue_entry& e(read_queue_.front());
                read_queue_.pop_front();
                threads::set_thread_state(e.id_, threads::pending);
            }

            // since we got full now we need to re-activate one thread waiting
            // for the block to become full
            if (!read_and_empty_queue_.empty()) {
                full_empty_queue_entry& e(read_and_empty_queue_.front());
                read_and_empty_queue_.pop_front();
                threads::set_thread_state(e.id_, threads::pending);
                set_empty_locked();   // state_ = empty
            }

            // return whether this block needs to be removed
            return state_ == full && !is_used_locked();
        }

        bool is_used_locked() const
        {
            return !(write_queue_.empty() && read_and_empty_queue_.empty() && read_queue_.empty());
        }

    private:
        typedef Data value_type;
        typedef boost::aligned_storage<sizeof(value_type),
            boost::alignment_of<value_type>::value> storage_type;

        // type safe accessors to the stored data
        typedef typename boost::add_pointer<value_type>::type pointer;
        typedef typename boost::add_pointer<value_type const>::type const_pointer;

        pointer get_address()
        {
            return static_cast<pointer>(data_.address());
        }
        const_pointer get_address() const
        {
            return static_cast<const_pointer>(data_.address());
        }

        mutable mutex_type mtx_;
        queue_type write_queue_;              // threads waiting in write
        queue_type read_and_empty_queue_;     // threads waiting in read_and_empty
        queue_type read_queue_;               // threads waiting in read
        storage_type data_;                   // protected data
        full_empty state_;                    // current full/empty state
    };

}}}

#endif

