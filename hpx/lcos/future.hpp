//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_FUTURE_MAR_06_2012_1059AM)
#define HPX_LCOS_FUTURE_MAR_06_2012_1059AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/util/date_time_chrono.hpp>

#include <boost/move/move.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    namespace local
    {
        template <typename Result> class promise;
        template <typename Result> class packaged_task;

        template <typename ContResult, typename Result>
        class packaged_continuation;

        namespace detail
        {
            template <typename ContResult> struct continuation_base;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename RemoteResult>
    class future
    {
    private:
        BOOST_COPYABLE_AND_MOVABLE(future)

        typedef lcos::detail::future_data_base<Result, RemoteResult>
            future_data_type;

        future(future_data_type* p)
          : future_data_(p)
        {}

        future(boost::intrusive_ptr<future_data_type> p)
          : future_data_(p)
        {}

        friend class local::promise<Result>;
        friend class local::packaged_task<Result()>;
        template <typename ContResult, typename Result_>
        friend class local::packaged_continuation;
        template <typename ContResult>
        friend struct local::detail::continuation_base;

        friend class promise<Result, RemoteResult>;
        friend class hpx::thread;
        friend struct detail::future_data<Result, RemoteResult>;

    public:
        typedef Result result_type;

        future()
        {}

        ~future()
        {}

        future(future const& other)
          : future_data_(other.future_data_)
        {
        }

        future(BOOST_RV_REF(future) other)
          : future_data_(other.future_data_)
        {
            other.future_data_.reset();
        }

        // extension: init from given value, set future to ready right away
        future(BOOST_RV_REF(Result) init)
          : future_data_(lcos::detail::future_data<Result>())
        {
            future_data_->set_data(boost::move(init));
        }

        // assignment
        future& operator=(BOOST_COPY_ASSIGN_REF(future) other)
        {
            future_data_ = other.future_data_;
            return *this;
        }

        future& operator=(BOOST_RV_REF(future) other)
        {
            future_data_ = other.future_data_;
            other.future_data_.reset();
            return *this;
        }

        void swap(future& other)
        {
            future_data_.swap(other.future_data_);
        }

        // retrieving the value
        Result get(error_code& ec = throws) const
        {
            if (!future_data_) {
                HPX_THROWS_IF(ec, future_uninitialized,
                    "future<Result, Remoteresult>::get",
                    "this future has not been initialized");
                return Result();
            }
            return future_data_->get_data(ec);
        }

        Result move(error_code& ec = throws)
        {
            return future_data_->move_data(ec);
        }

        // state introspection
        bool is_ready() const
        {
            return future_data_ && future_data_->is_ready();
        }

        bool has_value() const
        {
            return future_data_ && future_data_->has_value();
        }

        bool has_exception() const
        {
            return future_data_ && future_data_->has_exception();
        }

        BOOST_SCOPED_ENUM(future_status) get_state() const
        {
            if (!future_data_)
                return future_status::uninitialized;

            return future_data_->get_state();
        }

        // cancellation support
        bool is_cancelable() const
        {
            return future_data_->is_cancelable();
        }

        void cancel()
        {
            future_data_->cancel();
        }

        // continuation support
        template <typename F>
        future<typename boost::result_of<F()>::type>
        when(BOOST_FWD_REF(F) f);

        // reset any pending continuation function
        void when()
        {
            future_data_->reset_on_completed();
        }

        // wait support
        void wait() const
        {
            future_data_->wait();
        }

        BOOST_SCOPED_ENUM(future_status)
        wait_until(boost::posix_time::ptime const& at)
        {
            return future_data_->wait_until(at);
        }

        BOOST_SCOPED_ENUM(future_status)
        wait_for(boost::posix_time::time_duration const& p)
        {
            return future_data_->wait_for(p);
        }
        template <typename Clock, typename Duration>
        BOOST_SCOPED_ENUM(future_status)
        wait_until(boost::chrono::time_point<Clock, Duration> const& abs_time) const
        {
            return wait_until(util::to_ptime(abs_time));
        }

        template <typename Rep, typename Period>
        BOOST_SCOPED_ENUM(future_status)
        wait_for(boost::chrono::duration<Rep, Period> const& rel_time) const
        {
            return wait_for(util::to_time_duration(rel_time));
        }

    private:
        boost::intrusive_ptr<future_data_type> future_data_;
    };

    // extension: create a pre-initialized future object
    template <typename Result>
    future<Result> create_value(BOOST_FWD_REF(Result) init)
    {
        return future<Result>(boost::forward<Result>(init));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class future<void, util::unused_type>
    {
    private:
        BOOST_COPYABLE_AND_MOVABLE(future)

        typedef lcos::detail::future_data_base<void, util::unused_type>
            future_data_type;

        future(future_data_type* p)
          : future_data_(p)
        {}

        future(boost::intrusive_ptr<future_data_type> p)
          : future_data_(p)
        {}

        friend class local::promise<void>;
        friend class local::packaged_task<void()>;
        template <typename ContResult, typename Result_>
        friend class local::packaged_continuation;
        template <typename ContResult>
        friend struct local::detail::continuation_base;

        friend class promise<void, util::unused_type>;
        friend class hpx::thread;
        friend struct detail::future_data<void, util::unused_type>;

    public:
        typedef void result_type;

        future()
        {}

        ~future()
        {}

        future(future const& other)
          : future_data_(other.future_data_)
        {
        }

        future(BOOST_RV_REF(future) other)
          : future_data_(other.future_data_)
        {
            other.future_data_.reset();
        }

        future& operator=(BOOST_COPY_ASSIGN_REF(future) other)
        {
            future_data_ = other.future_data_;
            return *this;
        }

        future& operator=(BOOST_RV_REF(future) other)
        {
            future_data_ = other.future_data_;
            other.future_data_.reset();
            return *this;
        }

        void swap(future& other)
        {
            future_data_.swap(other.future_data_);
        }

        // retrieving the value
        void get(error_code& ec = throws) const
        {
            future_data_->get_data(ec);
        }

        void move(error_code& ec = throws)
        {
            future_data_->move_data(ec);
        }

        // state introspection
        bool is_ready() const
        {
            return future_data_->is_ready();
        }

        bool has_value() const
        {
            return future_data_->has_value();
        }

        bool has_exception() const
        {
            return future_data_->has_exception();
        }

        BOOST_SCOPED_ENUM(future_status) get_state() const
        {
            if (!future_data_)
                return future_status::uninitialized;

            return future_data_->get_state();
        }

        // cancellation support
        bool is_cancelable() const
        {
            return future_data_->is_cancelable();
        }

        void cancel()
        {
            future_data_->cancel();
        }

        bool valid() const BOOST_NOEXCEPT
        {
            return future_data_;
        }

        // continuation support
        template <typename F>
        future<typename boost::result_of<F()>::type>
        when(BOOST_FWD_REF(F) f);

        // reset any pending continuation function
        void when()
        {
            future_data_->reset_on_completed();
        }

        // wait support
        void wait() const
        {
            future_data_->wait();
        }

        BOOST_SCOPED_ENUM(future_status)
        wait_until(boost::posix_time::ptime const& at)
        {
            return future_data_->wait_until(at);
        }

        BOOST_SCOPED_ENUM(future_status)
        wait_for(boost::posix_time::time_duration const& p)
        {
            return future_data_->wait_for(p);
        }

        template <typename Clock, typename Duration>
        BOOST_SCOPED_ENUM(future_status)
        wait_until(boost::chrono::time_point<Clock, Duration> const& abs_time) const
        {
            return wait_until(util::to_ptime(abs_time));
        }

        template <typename Rep, typename Period>
        BOOST_SCOPED_ENUM(future_status)
        wait_for(boost::chrono::duration<Rep, Period> const& rel_time) const
        {
            return wait_for(util::to_time_duration(rel_time));
        }

    private:
        boost::intrusive_ptr<future_data_type> future_data_;
    };
}}

#endif
