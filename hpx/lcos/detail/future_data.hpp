//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_DETAIL_FUTURE_DATA_MAR_06_2012_1055AM)
#define HPX_LCOS_DETAIL_FUTURE_DATA_MAR_06_2012_1055AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/full_empty_memory.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/util/value_or_error.hpp>

#include <boost/move/move.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/detail/atomic_count.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    namespace future_state
    {
        enum state { uninitialized, deferred, ready, timeout };
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    struct future_data_refcnt_base;

    void intrusive_ptr_add_ref(future_data_refcnt_base* p);
    void intrusive_ptr_release(future_data_refcnt_base* p);

    ///////////////////////////////////////////////////////////////////////
    struct future_data_refcnt_base
    {
    public:
        typedef void has_future_data_refcnt_base;

        virtual ~future_data_refcnt_base() {}

    protected:
        future_data_refcnt_base() : count_(0) {}

    private:
        // reference counting
        friend void intrusive_ptr_add_ref(future_data_refcnt_base* p);
        friend void intrusive_ptr_release(future_data_refcnt_base* p);

        boost::detail::atomic_count count_;
    };

    /// support functions for boost::intrusive_ptr
    inline void intrusive_ptr_add_ref(future_data_refcnt_base* p)
    {
        ++p->count_;
    }
    inline void intrusive_ptr_release(future_data_refcnt_base* p)
    {
        if (0 == --p->count_)
            delete p;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename RemoteResult =
        typename traits::promise_remote_result<Result>::type>
    struct future_data_base : future_data_refcnt_base
    {
        typedef typename boost::mpl::if_<
            boost::is_same<void, Result>, util::unused_type, Result
        >::type result_type;

        virtual result_type get_data(error_code& ec = throws) = 0;
        virtual result_type move_data(error_code& ec = throws) = 0;
        virtual bool is_ready() const = 0;
        virtual bool has_value() const = 0;
        virtual bool has_exception() const = 0;
        virtual future_state::state get_state() const = 0;

    protected:
        future_data_base() {}
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename RemoteResult =
        typename traits::promise_remote_result<Result>::type>
    struct future_data : future_data_base<Result, RemoteResult>
    {
    public:
        typedef future_data_base<Result, RemoteResult> base_type;
        typedef typename base_type::result_type result_type;
        typedef boost::exception_ptr error_type;
        typedef util::value_or_error<result_type> data_type;

        typedef HPX_STD_FUNCTION<void(future<Result, RemoteResult>)>
            completed_callback_type;

    protected:
        future_data() {}

        future_data(completed_callback_type const& data_sink)
          : on_completed_(data_sink)
        {}

    public:
        result_type handle_error(data_type const& d, error_code &ec)
        {
            // an error has been reported in the meantime, throw or set
            // the error code
            if (&ec == &throws) {
                boost::rethrow_exception(d.get_error());
                // never reached
            }
            else {
                try {
                    boost::rethrow_exception(d.get_error());
                }
                catch (hpx::exception const& he) {
                    ec = make_error_code(he.get_error(), he.what(),
                        hpx::rethrow);
                }
            }
            return result_type();
        }

        /// Get the result of the requested action. This call blocks (yields
        /// control) if the result is not ready. As soon as the result has been
        /// returned and the waiting thread has been re-scheduled by the thread
        /// manager the function will return.
        ///
        /// \param ec     [in,out] this represents the error status on exit,
        ///               if this is pre-initialized to \a hpx#throws
        ///               the function will throw on error instead. If the
        ///               operation blocks and is aborted because the object
        ///               went out of scope, the code \a hpx#yield_aborted is
        ///               set or thrown.
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_exception), this function will throw an
        ///               exception encapsulating the reported error code and
        ///               error description if <code>&ec == &throws</code>.
        result_type get_data(error_code& ec = throws)
        {
            // yields control if needed
            data_type d;
            data_.read(d, ec);      // copies the data out of the store
            if (ec) return result_type();

            // the thread has been re-activated by one of the actions
            // supported by this promise (see \a promise::set_event
            // and promise::set_exception).
            if (!d.stores_value())
                return handle_error(d, ec);

            // no error has been reported, return the result
            return boost::move(d.get_value());
        }

        result_type move_data(error_code& ec = throws)
        {
            // yields control if needed
            data_type d;
            data_.read_and_empty(d, ec); // moves the data from the store
            if (ec) return result_type();

            // the thread has been re-activated by one of the actions
            // supported by this promise (see \a promise::set_event
            // and promise::set_exception).
            if (!d.stores_value())
                return handle_error(d, ec);

            // no error has been reported, return the result
            return boost::move(d.get_value());
        }

        // helper functions for setting data (if successful) or the error (if
        // non-successful)
        template <typename T>
        void set_data(BOOST_FWD_REF(T) result)
        {
            // set the received result, reset error status
            try {
                // check weather the data already has been set
                if (!data_.is_empty()) {
                    HPX_THROW_EXCEPTION(future_already_satisfied,
                        "packaged_task::set_data<Result>",
                        "data has already been set for this future");
                }

               typedef typename boost::remove_const<
                    typename hpx::util::detail::remove_reference<T>::type
                >::type naked_type;

                typedef traits::get_remote_result<
                    result_type, naked_type
                > get_remote_result_type;

                // store the value
                if (!on_completed_.empty()) {
                    // this future coincidentally instance keeps us alive
                    future<Result, RemoteResult> f(this);

                    data_.set(boost::move(get_remote_result_type::call(
                        boost::forward<T>(result))));
                    on_completed_(f);
                }
                else {
                  data_.set(boost::move(get_remote_result_type::call(
                        boost::forward<T>(result))));
                }
            }
            catch (hpx::exception const&) {
                // store the error instead
                set_exception(boost::current_exception());
            }
        }

        // trigger the future with the given error condition
        void set_exception(boost::exception_ptr const& e)
        {
            // store the error code
            if (!on_completed_.empty()) {
                // this future coincidentally instance keeps us alive
                future<Result, RemoteResult> f(this);
                data_.set(e);
                on_completed_(f);
            }
            else {
                data_.set(e);
            }
        }

        void set_error(error e, char const* f, char const* msg)
        {
            try {
                HPX_THROW_EXCEPTION(e, f, msg);
            }
            catch (hpx::exception const&) {
                // store the error code
                set_exception(boost::current_exception());
            }
        }

        /// Return whether or not the data is available for this
        /// \a promise.
        bool is_ready() const
        {
            return !data_.is_empty();
        }

    private:
        static bool has_data_helper(data_type const& d)
        {
            return d.stores_value();
        }

    public:
        bool has_value() const
        {
            return is_ready() && data_.peek(&has_data_helper);
        }

        bool has_exception() const
        {
            return is_ready() && !data_.peek(&has_data_helper);
        }

        future_state::state get_state() const
        {
            return is_ready() ? future_state::ready : future_state::deferred;
        }

        /// Reset the promise to allow to restart an asynchronous
        /// operation. Allows any subsequent set_data operation to succeed.
        void reset(error_code& ec = throws)
        {
            data_.set_empty(ec);
        }

    private:
        util::full_empty<data_type> data_;
        completed_callback_type on_completed_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct task_base : future_data<Result>
    {
        typedef typename future_data<Result>::result_type result_type;

    public:
        task_base()
          : started_(false)
        {}

        // retrieving the value
        result_type get(error_code& ec = throws)
        {
            if (!started_)
                run(ec);

            return boost::move(this->get_data(ec));
        }

        // moving out the value
        result_type move(error_code& ec = throws)
        {
            if (!started_)
                run(ec);

            return boost::move(this->move_data(ec));
        }

        void run(error_code& ec)
        {
            if (started_) {
                HPX_THROWS_IF(ec, task_already_started,
                    "task_base::run", "this task has already been started");
                return;
            }
            started_ = true;

            applier::register_thread_plain(
                HPX_STD_BIND(&task_base::run_impl, this, threads::wait_signaled),
                "task_base::run");

            if (&ec != &throws)
                ec = make_success_code();
        }

        threads::thread_state_enum run_impl(threads::thread_state_ex_enum)
        {
            this->do_run();
            return threads::terminated;
        }

        void deleting_owner()
        {
            if (!started_) {
                started_ = true;
                this->set_error(broken_task, "task_base::deleting_owner",
                    "deleting task owner before future has been executed");
            }
        }

        virtual void do_run() = 0;

    private:
        bool started_;
    };
}}}

#endif
