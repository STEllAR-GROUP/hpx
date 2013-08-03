//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_SERVER_BARRIER_MAR_10_2010_0310PM)
#define HPX_LCOS_SERVER_BARRIER_MAR_10_2010_0310PM

#include <hpx/hpx_fwd.hpp>

#include <boost/version.hpp>
#include <boost/intrusive/slist.hpp>

#include <hpx/lcos/local/mutex.hpp>
#include <hpx/util/scoped_unlock.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace server
{
    /// A barrier can be used to synchronize a specific number of threads,
    /// blocking all of the entering threads until all of the threads have
    /// entered the barrier.
    class barrier
      : public lcos::base_lco
      , public components::managed_component_base<barrier>
    {
    public:
        typedef lcos::base_lco base_type_holder;

    private:
        typedef components::managed_component_base<barrier> base_type;

        typedef hpx::lcos::local::mutex mutex_type;
        mutex_type mtx_;

        // define data structures needed for intrusive slist container used for
        // the queues
        struct barrier_queue_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>
            > hook_type;

            barrier_queue_entry(threads::thread_id_type id)
              : id_(id)
            {}

            threads::thread_id_type id_;
            hook_type slist_hook_;
        };

        typedef boost::intrusive::member_hook<
            barrier_queue_entry, barrier_queue_entry::hook_type,
            &barrier_queue_entry::slist_hook_
        > slist_option_type;

        typedef boost::intrusive::slist<
            barrier_queue_entry, slist_option_type,
            boost::intrusive::cache_last<true>,
            boost::intrusive::constant_time_size<false>
        > queue_type;

        struct reset_queue_entry
        {
            reset_queue_entry(barrier_queue_entry& e, queue_type& q)
              : e_(e), q_(q), last_(q.last())
            {}

            ~reset_queue_entry()
            {
                if (e_.id_)
                    q_.erase(last_);     // remove entry from queue
            }

            barrier_queue_entry& e_;
            queue_type& q_;
            queue_type::const_iterator last_;
        };

    public:
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_barrier };

        barrier()
          : number_of_threads_(1)
        {}

        barrier(std::size_t number_of_threads)
          : number_of_threads_(number_of_threads)
        {}

        ~barrier()
        {
            if (!queue_.empty()) {
                LERR_(fatal) << "~barrier: thread_queue is not empty, aborting threads";

                mutex_type::scoped_lock l(mtx_);

                while (!queue_.empty()) {
                    threads::thread_id_type id = queue_.front().id_;
                    queue_.front().id_ = 0;
                    queue_.pop_front();

                    // we know that the id is actually the pointer to the thread
                    threads::thread_data_base* thrd =
                        static_cast<threads::thread_data_base*>(id);
                    LERR_(fatal) << "~barrier: pending thread: "
                            << get_thread_state_name(thrd->get_state())
                            << "(" << id << "): " << thrd->get_description();

                    // forcefully abort thread, do not throw
                    error_code ec(lightweight);
                    threads::set_thread_state(id, threads::pending,
                        threads::wait_abort, threads::thread_priority_default, ec);
                    if (ec) {
                        LERR_(fatal) << "~barrier: could not abort thread"
                            << get_thread_state_name(thrd->get_state())
                            << "(" << id << "): " << thrd->get_description();
                    }
                }
            }
        }

        // disambiguate base classes
        using base_type::finalize;
        typedef base_type::wrapping_type wrapping_type;

        static components::component_type get_component_type()
        {
            return components::component_barrier;
        }
        static void set_component_type(components::component_type) {}

        // standard LCO action implementations

        /// The function \a set_event will block the number of entering
        /// \a threads (as given by the constructor parameter \a number_of_threads),
        /// releasing all waiting threads as soon as the last \a thread
        /// entered this function.
        void set_event()
        {
            threads::thread_self& self = threads::get_self();

            mutex_type::scoped_lock l(mtx_);

            if (queue_.size() < number_of_threads_-1) {
                barrier_queue_entry e(self.get_thread_id());
                queue_.push_back(e);

                reset_queue_entry r(e, queue_);
                {
                    util::scoped_unlock<mutex_type::scoped_lock> ul(l);
                    this_thread::suspend(threads::suspended,
                        "barrier::set_event");
                }
            }
            else {
            // slist::swap has a bug in Boost 1.35.0
#if BOOST_VERSION < 103600
                // release the threads
                while (!queue_.empty()) {
                    threads::thread_id_type id = queue_.front().id_;
                    queue_.front().id_ = 0;
                    queue_.pop_front();
                    set_thread_state(id, threads::pending);
                }
#else
                // swap the list
                queue_type queue;
                queue.swap(queue_);
                l.unlock();

                // release the threads
                while (!queue.empty()) {
                    threads::thread_id_type id = queue.front().id_;
                    queue.front().id_ = 0;
                    queue.pop_front();
                    set_thread_state(id, threads::pending);
                }
#endif
            }
        }

        /// The \a function set_exception is called whenever a
        /// \a set_exception_action is applied on an instance of a LCO. This
        /// function just forwards to the virtual function \a set_exception, which
        /// is overloaded by the derived concrete LCO.
        ///
        /// \param e      [in] The exception encapsulating the error to report
        ///               to this LCO instance.
        void set_exception(boost::exception_ptr const& e)
        {
            try {
                mutex_type::scoped_lock l(mtx_);

                while (!queue_.empty()) {
                    threads::thread_id_type id = queue_.front().id_;
                    queue_.front().id_ = 0;
                    queue_.pop_front();

                    threads::set_thread_state(id, threads::pending,
                        threads::wait_abort);
                }

                boost::rethrow_exception(e);
            }
            catch (boost::exception const& be) {
                // rethrow again, but this time using the native hpx mechanics
                HPX_THROW_EXCEPTION(hpx::no_success, "barrier::set_exception",
                    boost::diagnostic_information(be));
            }
        }

        typedef
            hpx::components::server::create_component_action1<
                barrier
              , std::size_t
            >
            create_component_action;

    private:
        std::size_t const number_of_threads_;
        queue_type queue_;
    };
}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::server::barrier::create_component_action
  , hpx_lcos_server_barrier_create_component_action
)

#endif

