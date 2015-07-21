//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_FULLEMPTYSTORE_JUN_16_2008_0128APM)
#define HPX_UTIL_FULLEMPTYSTORE_JUN_16_2008_0128APM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/lcos/local/no_mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/unlock_guard.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <boost/aligned_storage.hpp>
#include <boost/type_traits/alignment_of.hpp>
#include <boost/type_traits/add_pointer.hpp>
#include <boost/intrusive/slist.hpp>
#include <boost/thread/locks.hpp>

#include <memory>
#include <sstream>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // data structure holding all counters for full_empty entries
    struct full_empty_counter_data
    {
        full_empty_counter_data()
          : constructed_(0), destructed_(0),
            read_enqueued_(0), read_dequeued_(0), set_full_(0)
        {}

        boost::atomic_int64_t constructed_;
        boost::atomic_int64_t destructed_;
        boost::atomic_int64_t read_enqueued_;
        boost::atomic_int64_t read_dequeued_;
        boost::atomic_int64_t set_full_;
    };
    HPX_EXPORT extern full_empty_counter_data full_empty_counter_data_;

    // call this to register all counter types for full_empty entries
    void register_counter_types();

    ///////////////////////////////////////////////////////////////////////////
    template <typename Data, typename Mutex = lcos::local::spinlock>
    class full_empty_entry
    {
    public:
        typedef Mutex mutex_type;

    private:
        typedef threads::thread_id_type thread_id_type;

        // define data structures needed for intrusive slist container used for
        // the queues
        struct queue_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>
            > hook_type;

            queue_entry(thread_id_type const& id)
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

        struct reset_queue_entry
        {
            reset_queue_entry(queue_entry& e, queue_type& q)
              : e_(e), q_(q), last_(q.last())
            {}

            ~reset_queue_entry()
            {
                if (e_.id_)
                    q_.erase(last_);     // remove entry from queue
            }

            queue_entry& e_;
            queue_type& q_;
            typename queue_type::const_iterator last_;
        };

        void log_non_empty_queue(char const* const desc, queue_type& queue)
        {
            boost::lock_guard<mutex_type> l(mtx_);
            while (!queue.empty()) {
                threads::thread_id_type id = queue.front().id_;
                queue.front().id_ = threads::invalid_thread_id;
                queue.pop_front();

                // we know that the id is actually the pointer to the thread
                LERR_(info) << "~full_empty_entry: aborting pending thread in "
                        << desc << ": "
                        << get_thread_state_name(threads::get_thread_state(id))
                        << "(" << id.get() << "): " << threads::get_thread_description(id);

                // forcefully abort thread, do not throw
                error_code ec(lightweight);
                threads::set_thread_state(id, threads::pending,
                    threads::wait_abort, threads::thread_priority_default, ec);
                if (ec) {
                    LERR_(error) << "~full_empty_entry: could not abort thread"
                        << get_thread_state_name(threads::get_thread_state(id))
                        << "(" << id.get() << "): " << threads::get_thread_description(id);
                }
            }
        }

    public:
        full_empty_entry()
          : data_(), state_(empty)
        {
            ++full_empty_counter_data_.constructed_;
        }

        template <typename T0>
        explicit full_empty_entry(T0 && t0)
          : data_(std::forward<T0>(t0)), state_(empty)
        {
            ++full_empty_counter_data_.constructed_;
        }

        ~full_empty_entry()
        {
            if (is_used()) {
                LERR_(info) << "~full_empty_entry: one of the queues is not empty";
                log_non_empty_queue("write_queue", write_queue_);
                log_non_empty_queue("read_and_empty_queue", read_and_empty_queue_);
                log_non_empty_queue("read_queue", read_queue_);
            }

            ++full_empty_counter_data_.destructed_;
        }

        // returns whether this entry is currently empty
        bool is_empty() const
        {
            boost::lock_guard<mutex_type> l(mtx_);
            return state_ == empty;
        }

        // sets this entry to empty
        bool set_empty(error_code& ec = throws)
        {
            boost::lock_guard<mutex_type> l(mtx_);
            return set_empty_locked(ec);
        }

        // sets this entry to full
        bool set_full(error_code& ec = throws)
        {
            boost::lock_guard<mutex_type> l(mtx_);
            return set_full_locked(ec);
        }

        template <typename F>
        bool peek(F f) const
        {
            boost::lock_guard<mutex_type> l(mtx_);
            if (state_ == empty)
                return false;
            return f(data_);      // pass the data to the provided function
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue a get operation if full/full queue if entry is empty
        template <typename T>
        void enqueue_full_full(T& dest, error_code& ec = throws)
        {
            boost::unique_lock<mutex_type> l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                threads::thread_self* self = threads::get_self_ptr_checked(ec);
                if (0 == self || ec) return;

                // enqueue the request and block this thread
                queue_entry f(threads::get_self_id());
                read_queue_.push_back(f);

                ++full_empty_counter_data_.read_enqueued_;

                reset_queue_entry r(f, read_queue_);

                {
                    // yield this thread
                    util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);
                    this_thread::suspend(threads::suspended,
                        "full_empty_entry::enqueue_full_full", ec);
                    if (ec) return;
                }

                ++full_empty_counter_data_.read_dequeued_;
            }

            // copy the data to the destination
            dest = data_;

            if (&ec != &throws)
                ec = make_success_code();
        }

        // same as above, but for entries without associated data
        void enqueue_full_full(error_code& ec = throws)
        {
            boost::unique_lock<mutex_type> l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                threads::thread_self* self = threads::get_self_ptr_checked(ec);
                if (0 == self || ec) return;

                // enqueue the request and block this thread
                queue_entry f(threads::get_self_id());
                read_queue_.push_back(f);

                ++full_empty_counter_data_.read_enqueued_;

                reset_queue_entry r(f, read_queue_);

                {
                    // yield this thread
                    util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);
                    this_thread::suspend(threads::suspended,
                        "full_empty_entry::enqueue_full_full", ec);
                    if (ec) return;
                }

                ++full_empty_counter_data_.read_dequeued_;
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

        // enqueue a move operation if full/full queue if entry is empty
        template <typename T>
        void enqueue_full_full_move(T& dest, error_code& ec = throws)
        {
            boost::unique_lock<mutex_type> l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                threads::thread_self* self = threads::get_self_ptr_checked(ec);
                if (0 == self || ec) return;

                // enqueue the request and block this thread
                queue_entry f(threads::get_self_id());
                read_queue_.push_back(f);

                ++full_empty_counter_data_.read_enqueued_;

                reset_queue_entry r(f, read_queue_);

                {
                    // yield this thread
                    util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);
                    this_thread::suspend(threads::suspended,
                        "full_empty_entry::enqueue_full_full", ec);
                    if (ec) return;
                }

                ++full_empty_counter_data_.read_dequeued_;
            }

            // move the data to the destination
            dest = std::move(data_);

            if (&ec != &throws)
                ec = make_success_code();
        }

        // same as above, but for entries without associated data
        void enqueue_full_full_move(error_code& ec = throws)
        {
            enqueue_full_full(ec);
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue a get operation in full/empty queue if entry is empty
        template <typename T>
        void enqueue_full_empty(T& dest, error_code& ec = throws)
        {
            boost::unique_lock<mutex_type> l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                threads::thread_self* self = threads::get_self_ptr_checked(ec);
                if (0 == self || ec) return;

                // enqueue the request and block this thread
                queue_entry f(threads::get_self_id());
                read_and_empty_queue_.push_back(f);

                reset_queue_entry r(f, read_and_empty_queue_);

                {
                    // yield this thread
                    util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);
                    this_thread::suspend(threads::suspended,
                        "full_empty_entry::enqueue_full_empty", ec);
                    if (ec) return;
                }
            }

            // move the data to the destination
            dest = std::move(data_);

            set_empty_locked(ec);   // state_ = empty;
            if (ec) return;

            if (&ec != &throws)
                ec = make_success_code();
        }

        // same as above, but for entries without associated data
        void enqueue_full_empty(error_code& ec = throws)
        {
            boost::unique_lock<mutex_type> l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                threads::thread_self* self = threads::get_self_ptr_checked(ec);
                if (0 == self || ec) return;

                // enqueue the request and block this thread
                queue_entry f(threads::get_self_id());
                read_and_empty_queue_.push_back(f);

                reset_queue_entry r(f, read_and_empty_queue_);

                {
                    // yield this thread
                    util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);
                    this_thread::suspend(threads::suspended,
                        "full_empty_entry::enqueue_full_empty", ec);
                    if (ec) return;
                }
            }

            set_empty_locked(ec);   // state_ = empty
            if (ec) return;

            if (&ec != &throws)
                ec = make_success_code();
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue if entry is full, otherwise fill it
        template <typename T>
        void enqueue_if_full(T && src, error_code& ec = throws)
        {
            boost::unique_lock<mutex_type> l(mtx_);

            // block if this entry is already full
            if (state_ == full) {
                threads::thread_self* self = threads::get_self_ptr_checked(ec);
                if (0 == self || ec) return;

                // enqueue the request and block this thread
                queue_entry f(threads::get_self_id());
                write_queue_.push_back(f);

                reset_queue_entry r(f, write_queue_);

                {
                    // yield this thread
                    util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);
                    this_thread::suspend(threads::suspended,
                        "full_empty_entry::enqueue_if_full", ec);
                    if (ec) return;
                }
            }

            // set the data
            data_ = std::forward<T>(src);

            // make sure the entry is full
            set_full_locked(ec);    // state_ = full
            if (ec) return;

            if (&ec != &throws)
                ec = make_success_code();
        }

        // same as above, but for entries without associated data
        void enqueue_if_full(error_code& ec = throws)
        {
            boost::unique_lock<mutex_type> l(mtx_);

            // block if this entry is already full
            if (state_ == full) {
                threads::thread_self* self = threads::get_self_ptr_checked(ec);
                if (0 == self || ec) return;

                // enqueue the request and block this thread
                queue_entry f(threads::get_self_id());
                write_queue_.push_back(f);

                reset_queue_entry r(f, write_queue_);

                {
                    // yield this thread
                    util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);
                    this_thread::suspend(threads::suspended,
                        "full_empty_entry::enqueue_if_full", ec);
                    if (ec) return;
                }
            }

            // make sure the entry is full
            set_full_locked(ec);    // state_ = full
            if (ec) return;

            if (&ec != &throws)
                ec = make_success_code();
        }

        ///////////////////////////////////////////////////////////////////////
        // unconditionally set the data and set the entry to full
        template <typename T>
        void set_and_fill(T && src, error_code& ec = throws)
        {
            boost::lock_guard<mutex_type> l(mtx_);

            // set the data
            data_ = std::forward<T>(src);

            // make sure the entry is full
            set_full_locked(ec);    // state_ = full
        }

        // same as above, but for entries without associated data
        void set_and_fill(error_code& ec = throws)
        {
            boost::lock_guard<mutex_type> l(mtx_);

            // make sure the entry is full
            set_full_locked(ec);    // state_ = full
        }

        // returns whether this entry is still in use
        bool is_used() const
        {
            boost::lock_guard<mutex_type> l(mtx_);
            return is_used_locked();
        }

    protected:
        bool set_empty_locked(error_code& ec)
        {
            state_ = empty;

            if (!write_queue_.empty()) {
                threads::thread_id_type id = write_queue_.front().id_;
                write_queue_.front().id_ = threads::invalid_thread_id;
                write_queue_.pop_front();

                threads::set_thread_state(id, threads::pending,
                    threads::wait_timeout, threads::thread_priority_default, ec);

                set_full_locked(ec);    // state_ = full
                if (ec) return false;
            }

            // return whether this block needs to be removed
            return state_ == full && !is_used_locked();
        }

        bool set_full_locked(error_code& ec)
        {
            state_ = full;

            // handle all threads waiting for the block to become full
            while (!read_queue_.empty()) {
                threads::thread_id_type id = read_queue_.front().id_;
                read_queue_.front().id_ = threads::invalid_thread_id;
                read_queue_.pop_front();

                threads::set_thread_state(id, threads::pending,
                    threads::wait_timeout, threads::thread_priority_default, ec);
                if (ec) return false;

                ++full_empty_counter_data_.set_full_;
            }

            // since we got full now we need to re-activate one thread waiting
            // for the block to become full
            if (!read_and_empty_queue_.empty()) {
                threads::thread_id_type id = read_and_empty_queue_.front().id_;
                read_and_empty_queue_.front().id_ = threads::invalid_thread_id;
                read_and_empty_queue_.pop_front();

                threads::set_thread_state(id, threads::pending,
                    threads::wait_timeout, threads::thread_priority_default, ec);
                if (ec) return false;

                set_empty_locked(ec);   // state_ = empty
                if (ec) return false;
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

        mutable mutex_type mtx_;
        queue_type write_queue_;              // threads waiting in write
        queue_type read_and_empty_queue_;     // threads waiting in read_and_empty
        queue_type read_queue_;               // threads waiting in read
        value_type data_;                     // protected data
        full_empty_state state_;              // current full/empty state
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Data>
    class full_empty_entry<Data, lcos::local::no_mutex>
    {
    private:
        typedef threads::thread_id_type thread_id_type;

        // define data structures needed for intrusive slist container used for
        // the queues
        struct queue_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>
            > hook_type;

            queue_entry(thread_id_type const& id)
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

        struct reset_queue_entry
        {
            reset_queue_entry(queue_entry& e, queue_type& q)
              : e_(e), q_(q), last_(q.last())
            {}

            ~reset_queue_entry()
            {
                if (e_.id_)
                    q_.erase(last_);     // remove entry from queue
            }

            queue_entry& e_;
            queue_type& q_;
            typename queue_type::const_iterator last_;
        };

        void log_non_empty_queue(char const* const desc, queue_type& queue)
        {
            while (!queue.empty()) {
                threads::thread_id_type id = queue.front().id_;
                queue.front().id_ = threads::invalid_thread_id;
                queue.pop_front();

                // we know that the id is actually the pointer to the thread
                LERR_(info) << "~full_empty_entry: aborting pending thread in "
                        << desc << ": "
                        << get_thread_state_name(threads::get_thread_state(id))
                        << "(" << id.get() << "): " << threads::get_thread_description(id);

                // forcefully abort thread, do not throw
                error_code ec(lightweight);
                threads::set_thread_state(id, threads::pending,
                    threads::wait_abort, threads::thread_priority_default, ec);
                if (ec) {
                    LERR_(error) << "~full_empty_entry: could not abort thread"
                        << get_thread_state_name(threads::get_thread_state(id))
                        << "(" << id.get() << "): " << threads::get_thread_description(id);
                }
            }
        }

    public:
        full_empty_entry()
          : data_(), state_(empty)
        {
            ++full_empty_counter_data_.constructed_;
        }

        template <typename T0>
        explicit full_empty_entry(T0 && t0)
          : data_(std::forward<T0>(t0)), state_(empty)
        {
            ++full_empty_counter_data_.constructed_;
        }

        ~full_empty_entry()
        {
            if (is_used()) {
                LERR_(info) << "~full_empty_entry: one of the queues is not empty";
                log_non_empty_queue("write_queue", write_queue_);
                log_non_empty_queue("read_and_empty_queue", read_and_empty_queue_);
                log_non_empty_queue("read_queue", read_queue_);
            }

            ++full_empty_counter_data_.destructed_;
        }

        // returns whether this entry is currently empty
        bool is_empty() const
        {
            return state_ == empty;
        }

        // sets this entry to empty
        bool set_empty(error_code& ec = throws)
        {
            return set_empty_locked(ec);
        }

        // sets this entry to full
        bool set_full(error_code& ec = throws)
        {
            return set_full_locked(ec);
        }

        template <typename F>
        bool peek(F f) const
        {
            if (state_ == empty)
                return false;
            return f(data_);      // pass the data to the provided function
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue a get operation if full/full queue if entry is empty
        template <typename T, typename Lock>
        void enqueue_full_full(T& dest, Lock& l, error_code& ec = throws)
        {
            // block if this entry is empty
            if (state_ == empty) {
                threads::thread_self* self = threads::get_self_ptr_checked(ec);
                if (0 == self || ec) return;

                // enqueue the request and block this thread
                queue_entry f(threads::get_self_id());
                read_queue_.push_back(f);

                ++full_empty_counter_data_.read_enqueued_;

                reset_queue_entry r(f, read_queue_);

                {
                    // yield this thread
                    util::unlock_guard<Lock> ul(l);
                    this_thread::suspend(threads::suspended,
                        "full_empty_entry::enqueue_full_full", ec);
                    if (ec) return;
                }

                ++full_empty_counter_data_.read_dequeued_;
            }

            // copy the data to the destination
            dest = data_;

            if (&ec != &throws)
                ec = make_success_code();
        }

        // same as above, but for entries without associated data
        template <typename Lock>
        void enqueue_full_full(Lock& l, error_code& ec = throws)
        {
            // block if this entry is empty
            if (state_ == empty) {
                threads::thread_self* self = threads::get_self_ptr_checked(ec);
                if (0 == self || ec) return;

                // enqueue the request and block this thread
                queue_entry f(threads::get_self_id());
                read_queue_.push_back(f);

                ++full_empty_counter_data_.read_enqueued_;

                reset_queue_entry r(f, read_queue_);

                {
                    // yield this thread
                    util::unlock_guard<Lock> ul(l);
                    this_thread::suspend(threads::suspended,
                        "full_empty_entry::enqueue_full_full", ec);
                    if (ec) return;
                }

                ++full_empty_counter_data_.read_dequeued_;
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

        // enqueue a move operation if full/full queue if entry is empty
        template <typename T, typename Lock>
        void enqueue_full_full_move(T& dest, Lock& l, error_code& ec = throws)
        {
            // block if this entry is empty
            if (state_ == empty) {
                threads::thread_self* self = threads::get_self_ptr_checked(ec);
                if (0 == self || ec) return;

                // enqueue the request and block this thread
                queue_entry f(threads::get_self_id());
                read_queue_.push_back(f);

                ++full_empty_counter_data_.read_enqueued_;

                reset_queue_entry r(f, read_queue_);

                {
                    // yield this thread
                    util::unlock_guard<Lock> ul(l);
                    this_thread::suspend(threads::suspended,
                        "full_empty_entry::enqueue_full_full", ec);
                    if (ec) return;
                }

                ++full_empty_counter_data_.read_dequeued_;
            }

            // move the data to the destination
            dest = std::move(data_);

            if (&ec != &throws)
                ec = make_success_code();
        }

        // same as above, but for entries without associated data
        template <typename Lock>
        void enqueue_full_full_move(Lock& l, error_code& ec = throws)
        {
            enqueue_full_full(l, ec);
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue a get operation in full/empty queue if entry is empty
        template <typename T, typename Lock>
        void enqueue_full_empty(T& dest, Lock& l, error_code& ec = throws)
        {
            // block if this entry is empty
            if (state_ == empty) {
                threads::thread_self* self = threads::get_self_ptr_checked(ec);
                if (0 == self || ec) return;

                // enqueue the request and block this thread
                queue_entry f(threads::get_self_id());
                read_and_empty_queue_.push_back(f);

                reset_queue_entry r(f, read_and_empty_queue_);

                {
                    // yield this thread
                    util::unlock_guard<Lock> ul(l);
                    this_thread::suspend(threads::suspended,
                        "full_empty_entry::enqueue_full_empty", ec);
                    if (ec) return;
                }
            }

            // move the data to the destination
            dest = std::move(data_);

            set_empty_locked(ec);   // state_ = empty;
            if (ec) return;

            if (&ec != &throws)
                ec = make_success_code();
        }

        // same as above, but for entries without associated data
        template <typename Lock>
        void enqueue_full_empty(Lock& l, error_code& ec = throws)
        {
            // block if this entry is empty
            if (state_ == empty) {
                threads::thread_self* self = threads::get_self_ptr_checked(ec);
                if (0 == self || ec) return;

                // enqueue the request and block this thread
                queue_entry f(threads::get_self_id());
                read_and_empty_queue_.push_back(f);

                reset_queue_entry r(f, read_and_empty_queue_);

                {
                    // yield this thread
                    util::unlock_guard<Lock> ul(l);
                    this_thread::suspend(threads::suspended,
                        "full_empty_entry::enqueue_full_empty", ec);
                    if (ec) return;
                }
            }

            set_empty_locked(ec);   // state_ = empty
            if (ec) return;

            if (&ec != &throws)
                ec = make_success_code();
        }

        ///////////////////////////////////////////////////////////////////////
        // enqueue if entry is full, otherwise fill it
        template <typename T, typename Lock>
        void enqueue_if_full(T && src, Lock& l, error_code& ec = throws)
        {
            // block if this entry is already full
            if (state_ == full) {
                threads::thread_self* self = threads::get_self_ptr_checked(ec);
                if (0 == self || ec) return;

                // enqueue the request and block this thread
                queue_entry f(threads::get_self_id());
                write_queue_.push_back(f);

                reset_queue_entry r(f, write_queue_);

                {
                    // yield this thread
                    util::unlock_guard<Lock> ul(l);
                    this_thread::suspend(threads::suspended,
                        "full_empty_entry::enqueue_if_full", ec);
                    if (ec) return;
                }
            }

            // set the data
            data_ = std::forward<T>(src);

            // make sure the entry is full
            set_full_locked(ec);    // state_ = full
            if (ec) return;

            if (&ec != &throws)
                ec = make_success_code();
        }

        // same as above, but for entries without associated data
        template <typename Lock>
        void enqueue_if_full(Lock& l, error_code& ec = throws)
        {
            // block if this entry is already full
            if (state_ == full) {
                threads::thread_self* self = threads::get_self_ptr_checked(ec);
                if (0 == self || ec) return;

                // enqueue the request and block this thread
                queue_entry f(threads::get_self_id());
                write_queue_.push_back(f);

                reset_queue_entry r(f, write_queue_);

                {
                    // yield this thread
                    util::unlock_guard<Lock> ul(l);
                    this_thread::suspend(threads::suspended,
                        "full_empty_entry::enqueue_if_full", ec);
                    if (ec) return;
                }
            }

            // make sure the entry is full
            set_full_locked(ec);    // state_ = full
            if (ec) return;

            if (&ec != &throws)
                ec = make_success_code();
        }

        ///////////////////////////////////////////////////////////////////////
        // unconditionally set the data and set the entry to full
        template <typename T>
        void set_and_fill(T && src, error_code& ec = throws)
        {
            // set the data
            data_ = std::forward<T>(src);

            // make sure the entry is full
            set_full_locked(ec);    // state_ = full
        }

        // same as above, but for entries without associated data
        void set_and_fill(error_code& ec = throws)
        {
            // make sure the entry is full
            set_full_locked(ec);    // state_ = full
        }

        // returns whether this entry is still in use
        bool is_used() const
        {
            return is_used_locked();
        }

    protected:
        bool set_empty_locked(error_code& ec)
        {
            state_ = empty;

            if (!write_queue_.empty()) {
                threads::thread_id_type id = write_queue_.front().id_;
                write_queue_.front().id_ = threads::invalid_thread_id;
                write_queue_.pop_front();

                threads::set_thread_state(id, threads::pending,
                    threads::wait_timeout, threads::thread_priority_default, ec);

                set_full_locked(ec);    // state_ = full
                if (ec) return false;
            }

            // return whether this block needs to be removed
            return state_ == full && !is_used_locked();
        }

        bool set_full_locked(error_code& ec)
        {
            state_ = full;

            // handle all threads waiting for the block to become full
            while (!read_queue_.empty()) {
                threads::thread_id_type id = read_queue_.front().id_;
                read_queue_.front().id_ = threads::invalid_thread_id;
                read_queue_.pop_front();

                threads::set_thread_state(id, threads::pending,
                    threads::wait_timeout, threads::thread_priority_default, ec);
                if (ec) return false;

                ++full_empty_counter_data_.set_full_;
            }

            // since we got full now we need to re-activate one thread waiting
            // for the block to become full
            if (!read_and_empty_queue_.empty()) {
                threads::thread_id_type id = read_and_empty_queue_.front().id_;
                read_and_empty_queue_.front().id_ = threads::invalid_thread_id;
                read_and_empty_queue_.pop_front();

                threads::set_thread_state(id, threads::pending,
                    threads::wait_timeout, threads::thread_priority_default, ec);
                if (ec) return false;

                set_empty_locked(ec);   // state_ = empty
                if (ec) return false;
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

        queue_type write_queue_;              // threads waiting in write
        queue_type read_and_empty_queue_;     // threads waiting in read_and_empty
        queue_type read_queue_;               // threads waiting in read
        value_type data_;                     // protected data
        full_empty_state state_;              // current full/empty state
    };
}}}

#endif

