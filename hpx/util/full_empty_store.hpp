//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_FULLEMPTYSTORE_JUN_16_2008_0128APM)
#define HPX_UTIL_FULLEMPTYSTORE_JUN_16_2008_0128APM

#include <memory>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <boost/aligned_storage.hpp>
#include <boost/type_traits/alignment_of.hpp>
#include <boost/type_traits/add_pointer.hpp>
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
        struct tag {};

    public:
        typedef hpx::util::spinlock_pool<tag> mutex_type;
        typedef typename mutex_type::scoped_lock scoped_lock;

    private:
        typedef threads::thread_id_type thread_id_type;

        // define data structures needed for intrusive slist container used for
        // the queues
        struct queue_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>
            > hook_type;

            queue_entry(thread_id_type id)
              : id_(id)
            {}

            thread_id_type id_;
            hook_type list_hook_;
        };

        typedef boost::intrusive::member_hook<
            queue_entry, typename queue_entry::hook_type,
            &queue_entry::list_hook_
        > list_option_type;

        typedef boost::intrusive::slist<
            queue_entry, list_option_type, 
            boost::intrusive::cache_last<true>,
            boost::intrusive::constant_time_size<false>
        > queue_type;

        void log_non_empty_queue(char const* const desc, queue_type& queue)
        {
            scoped_lock l(this);
            while (!queue.empty()) {
                threads::thread_id_type id = queue.front().id_;
                queue.front().id_ = 0;
                queue.pop_front();

                // we know that the id is actually the pointer to the thread
                threads::thread* thrd = reinterpret_cast<threads::thread*>(id);
                LERR_(info) << "~full_empty_entry: pending thread in " 
                        << desc << ": " 
                        << get_thread_state_name(thrd->get_state()) 
                        << "(" << id << "): " << thrd->get_description();

                // forcefully abort thread, do not throw
                error_code ec;
                threads::set_thread_state(id, threads::pending,
                    threads::wait_abort, threads::thread_priority_normal, ec);
                if (ec) {
                    LERR_(info) << "~full_empty_entry: could not abort thread"
                        << get_thread_state_name(thrd->get_state()) 
                        << "(" << id << "): " << thrd->get_description();
                }
            }
        }

    public:
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
            if (is_used()) {
                LERR_(info) << "~full_empty_entry: one of the queues is not empty";
                log_non_empty_queue("write_queue", write_queue_);
                log_non_empty_queue("read_and_empty_queue", read_and_empty_queue_);
                log_non_empty_queue("read_queue", read_queue_);
            }
            get_address()->Data::~Data();      // properly destruct value in memory
        }

        // returns whether this entry is currently empty
        bool is_empty() const
        {
            scoped_lock l(this);
            return state_ == empty;
        }

        // sets this entry to empty
        bool set_empty() 
        {
            scoped_lock l(this);
            return set_empty_locked();
        }

        // sets this entry to full
        bool set_full() 
        {
            scoped_lock l(this);
            return set_full_locked();
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue a get operation if full/full queue if entry is empty
        template <typename T>
        void enqueue_full_full(T& dest, error_code& ec = throws)
        {
            threads::thread_self* self = threads::get_self_ptr_checked(ec);

            if (&ec != &throws && ec)
                return;

            threads::thread_id_type id = self->get_thread_id();

            scoped_lock l(this);

            // block if this entry is empty
            if (state_ == empty) {
                // enqueue the request and block this thread
                threads::set_thread_lco_description(id, "enqueue_full_full", ec);

                if (&ec != &throws && ec)
                    return;  

                queue_entry f(id);
                read_queue_.push_back(f);

                typename queue_type::const_iterator last = read_queue_.last();
                {
                    // yield this thread
                    util::unlock_the_lock<scoped_lock> ul(l);
                    threads::thread_state_ex_enum statex = self->yield(threads::suspended);
                    if (statex == threads::wait_abort) {
                        hpx::util::osstream strm;

                        error_code ig;
                        std::string desc = threads::get_thread_description(id, ig);

                        strm << "thread(" << id
                             << (desc.empty() ? "" : ", " ) << desc
                             << ") aborted (yield returned wait_abort)";
                        HPX_THROWS_IF(ec, yield_aborted, 
                            "full_empty_entry::enqueue_full_full",
                            hpx::util::osstream_get_string(strm));
                        return;
                    }
                }
                if (f.id_) 
                    read_queue_.erase(last);     // remove entry from queue
            }

            // copy the data to the destination
            if (get_address() != &dest) 
                dest = *get_address();

            if (&ec != &throws)
                ec = make_success_code();
        }

        // same as above, but for entries without associated data
        void enqueue_full_full(error_code& ec = throws)
        {
            threads::thread_self* self = threads::get_self_ptr_checked(ec);

            if (&ec != &throws && ec)
                return;

            threads::thread_id_type id = self->get_thread_id();

            scoped_lock l(this);

            // block if this entry is empty
            if (state_ == empty) {
                // enqueue the request and block this thread
                threads::set_thread_lco_description(id, "enqueue_full_full", ec);

                if (&ec != &throws && ec)
                    return;  

                queue_entry f(id);
                read_queue_.push_back(f);

                typename queue_type::const_iterator last = read_queue_.last();
                {
                    // yield this thread
                    util::unlock_the_lock<scoped_lock> ul(l);
                    threads::thread_state_ex_enum statex = self->yield(threads::suspended);
                    if (statex == threads::wait_abort) {
                        hpx::util::osstream strm;

                        error_code ig;
                        std::string desc = threads::get_thread_description(id, ig);

                        strm << "thread(" << id
                             << (desc.empty() ? "" : ", " ) << desc
                             << ") aborted (yield returned wait_abort)";
                        HPX_THROWS_IF(ec, yield_aborted, 
                            "full_empty_entry::enqueue_full_full",
                            hpx::util::osstream_get_string(strm));
                        return;
                    }
                }
                if (f.id_) 
                    read_queue_.erase(last);     // remove entry from queue
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue a get operation in full/empty queue if entry is empty
        template <typename T>
        void enqueue_full_empty(T& dest, error_code& ec = throws)
        {
            threads::thread_self* self = threads::get_self_ptr_checked(ec);

            if (&ec != &throws && ec)
                return;

            threads::thread_id_type id = self->get_thread_id();

            scoped_lock l(this);

            // block if this entry is empty
            if (state_ == empty) {
                // enqueue the request and block this thread
                threads::set_thread_lco_description(id, "enqueue_full_empty", ec);

                if (&ec != &throws && ec)
                    return;  

                queue_entry f(id);
                read_and_empty_queue_.push_back(f);

                typename queue_type::const_iterator last = read_and_empty_queue_.last();
                {
                    // yield this thread
                    util::unlock_the_lock<scoped_lock> ul(l);
                    threads::thread_state_ex_enum statex = self->yield(threads::suspended);
                    if (statex == threads::wait_abort) {
                        hpx::util::osstream strm;

                        error_code ig;
                        std::string desc = threads::get_thread_description(id, ig);

                        strm << "thread(" << id
                             << (desc.empty() ? "" : ", " ) << desc
                             << ") aborted (yield returned wait_abort)";
                        HPX_THROWS_IF(ec, yield_aborted, 
                            "full_empty_entry::enqueue_full_empty",
                            hpx::util::osstream_get_string(strm));
                        return;
                    }
                }
                if (f.id_) 
                    read_and_empty_queue_.erase(last);     // remove entry from queue

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

            if (&ec != &throws)
                ec = make_success_code();
        }

        // same as above, but for entries without associated data
        void enqueue_full_empty(error_code& ec = throws)
        {
            threads::thread_self* self = threads::get_self_ptr_checked(ec);

            if (&ec != &throws && ec)
                return;

            threads::thread_id_type id = self->get_thread_id();

            scoped_lock l(this);

            // block if this entry is empty
            if (state_ == empty) {
                // enqueue the request and block this thread
                threads::set_thread_lco_description(id, "enqueue_full_empty", ec);

                if (&ec != &throws && ec)
                    return;  

                queue_entry f(id);
                read_and_empty_queue_.push_back(f);

                typename queue_type::const_iterator last = read_and_empty_queue_.last();
                {
                    // yield this thread
                    util::unlock_the_lock<scoped_lock> ul(l);
                    threads::thread_state_ex_enum statex = self->yield(threads::suspended);
                    if (statex == threads::wait_abort) {
                        hpx::util::osstream strm;

                        error_code ig;
                        std::string desc = threads::get_thread_description(id, ig);

                        strm << "thread(" << id
                             << (desc.empty() ? "" : ", " ) << desc
                             << ") aborted (yield returned wait_abort)";
                        HPX_THROWS_IF(ec, yield_aborted, 
                            "full_empty_entry::enqueue_full_empty",
                            hpx::util::osstream_get_string(strm));
                        return;
                    }
                }
                if (f.id_) 
                    read_and_empty_queue_.erase(last);     // remove entry from queue
            }
            else {
                set_empty_locked();   // state_ = empty
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue if entry is full, otherwise fill it
        template <typename T>
        void enqueue_if_full(T const& src, error_code& ec = throws)
        {
            threads::thread_self* self = threads::get_self_ptr_checked(ec);

            if (&ec != &throws && ec)
                return;

            threads::thread_id_type id = self->get_thread_id();

            scoped_lock l(this);

            // block if this entry is already full
            if (state_ == full) {
                // enqueue the request and block this thread
                threads::set_thread_lco_description(id, "enqueue_if_full", ec);

                if (&ec != &throws && ec)
                    return;  

                queue_entry f(id);
                write_queue_.push_back(f);

                typename queue_type::const_iterator last = write_queue_.last();
                {
                    // yield this thread
                    util::unlock_the_lock<scoped_lock> ul(l);
                    threads::thread_state_ex_enum statex = self->yield(threads::suspended);
                    if (statex == threads::wait_abort) {
                        hpx::util::osstream strm;

                        error_code ig;
                        std::string desc = threads::get_thread_description(id, ig);

                        strm << "thread(" << id
                             << (desc.empty() ? "" : ", " ) << desc
                             << ") aborted (yield returned wait_abort)";
                        HPX_THROWS_IF(ec, yield_aborted, 
                            "full_empty_entry::enqueue_if_full",
                            hpx::util::osstream_get_string(strm));
                        return;
                    }
                }
                if (f.id_) 
                    write_queue_.erase(last);     // remove entry from queue
            }

            // set the data
            if (get_address() != &src) 
                *get_address() = src;

            // make sure the entry is full
            set_full_locked();    // state_ = full

            if (&ec != &throws)
                ec = make_success_code();
        }

        // same as above, but for entries without associated data
        void enqueue_if_full(error_code& ec = throws)
        {
            threads::thread_self* self = threads::get_self_ptr_checked(ec);

            if (&ec != &throws && ec)
                return;

            threads::thread_id_type id = self->get_thread_id();

            scoped_lock l(this);

            // block if this entry is already full
            if (state_ == full) {
                // enqueue the request and block this thread
                threads::set_thread_lco_description(id, "enqueue_if_full", ec);

                if (&ec != &throws && ec)
                    return;  
 
                queue_entry f(id);
                write_queue_.push_back(f);

                typename queue_type::const_iterator last = write_queue_.last();
                {
                    // yield this thread
                    util::unlock_the_lock<scoped_lock> ul(l);
                    threads::thread_state_ex_enum statex = self->yield(threads::suspended);
                    if (statex == threads::wait_abort) {
                        hpx::util::osstream strm;

                        error_code ig;
                        std::string desc = threads::get_thread_description(id, ig);

                        strm << "thread(" << id
                             << (desc.empty() ? "" : ", " ) << desc
                             << ") aborted (yield returned wait_abort)";
                        HPX_THROWS_IF(ec, yield_aborted, 
                            "full_empty_entry::enqueue_if_full",
                            hpx::util::osstream_get_string(strm));
                        return;
                    }
                }
                if (f.id_) 
                    write_queue_.erase(last);     // remove entry from queue
            }

            // make sure the entry is full
            set_full_locked();    // state_ = full

            if (&ec != &throws)
                ec = make_success_code();
        }

        ///////////////////////////////////////////////////////////////////////
        // unconditionally set the data and set the entry to full
        template <typename T>
        void set_and_fill(T const& src)
        {
            scoped_lock l(this);

            // set the data
            if (get_address() != &src) 
                *get_address() = src;

            // make sure the entry is full
            set_full_locked();    // state_ = full
        }

        // same as above, but for entries without associated data
        void set_and_fill()
        {
            scoped_lock l(this);

            // make sure the entry is full
            set_full_locked();    // state_ = full
        }

        // returns whether this entry is still in use
        bool is_used() const
        {
            scoped_lock l(this);
            return is_used_locked();
        }

    protected:
        bool set_empty_locked()
        {
            state_ = empty;

            if (!write_queue_.empty()) {
                threads::thread_id_type id = write_queue_.front().id_;
                write_queue_.front().id_ = 0;
                write_queue_.pop_front();
                threads::set_thread_lco_description(id);
                threads::set_thread_state(id, threads::pending);
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
                threads::thread_id_type id = read_queue_.front().id_;
                read_queue_.front().id_ = 0;
                read_queue_.pop_front();
                threads::set_thread_lco_description(id);
                threads::set_thread_state(id, threads::pending);
            }

            // since we got full now we need to re-activate one thread waiting
            // for the block to become full
            if (!read_and_empty_queue_.empty()) {
                threads::thread_id_type id = read_and_empty_queue_.front().id_;
                read_and_empty_queue_.front().id_ = 0;
                read_and_empty_queue_.pop_front();
                threads::set_thread_lco_description(id);
                threads::set_thread_state(id, threads::pending);
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

