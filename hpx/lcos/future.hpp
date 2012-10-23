//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_FUTURE_MAR_06_2012_1059AM)
#define HPX_LCOS_FUTURE_MAR_06_2012_1059AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/util/detail/remove_reference.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/function_types/result_type.hpp>

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    namespace local
    {
        template <typename Result> class promise;
        template <typename Func> class packaged_task;
        template <typename Func> class futures_factory;

        template <typename ContResult, typename Result>
        class packaged_continuation;

        namespace detail
        {
            template <typename ContResult> struct continuation_base;
        }
    }

//     namespace detail
//     {
//         template <typename Future, typename F, typename Enable = void>
//         struct future_when_result;
//
//         template <typename Future, typename F>
//         struct future_when_result<
//             Future, F, typename boost::result_of<F(Future)>::type
//         >
//         {
//             typedef typename boost::result_of<F(Future)>::type result_type;
//             typedef typename traits::promise_remote_result<result_type>::type remote_type;
//             typedef lcos::future<typename boost::result_of<F(Future)>::type> type;
//         };
//
//         template <typename Future, typename F>
//         struct future_when_result<
//             Future, F,
//             typename boost::result_of<F(typename Future::result_type)>::type
//         >
//         {
//             typedef typename boost::result_of<F(typename Future::result_type)>::type result_type;
//             typedef typename traits::promise_remote_result<result_type>::type remote_type;
//             typedef lcos::future<result_type> type;
//         };
//     }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    class future
    {
    private:
        BOOST_COPYABLE_AND_MOVABLE(future)

        typedef lcos::detail::future_data_base<Result> future_data_type;

        explicit future(future_data_type* p)
          : future_data_(p)
        {}

        explicit future(boost::intrusive_ptr<future_data_type> p)
          : future_data_(p)
        {}

        friend class local::promise<Result>;
        friend class local::packaged_task<Result()>;
        friend class local::futures_factory<Result()>;

        template <typename ContResult, typename Result_>
        friend class local::packaged_continuation;
        template <typename ContResult>
        friend struct local::detail::continuation_base;
        template <typename Result_, typename RemoteResult_>
        friend class promise;
        friend struct detail::future_data<Result>;

        friend class hpx::thread;

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
        explicit future(Result const& init)
        {
            typedef lcos::detail::future_data<Result> impl_type;
            boost::intrusive_ptr<future_data_type> p = new impl_type();
            static_cast<impl_type*>(p.get())->set_data(init);
            future_data_ = p;
        }

        explicit future(BOOST_RV_REF(Result) init)
        {
            typedef lcos::detail::future_data<Result> impl_type;
            boost::intrusive_ptr<future_data_type> p = new impl_type();
            static_cast<impl_type*>(p.get())->set_data(boost::move(init));
            future_data_ = p;
        }

        // assignment
        future& operator=(BOOST_COPY_ASSIGN_REF(future) other)
        {
            if (this != &other)
                future_data_ = other.future_data_;
            return *this;
        }

        future& operator=(BOOST_RV_REF(future) other)
        {
            if (this != &other) {
                future_data_ = boost::move(other.future_data_);
                other.future_data_.reset();
            }
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
                    "future<Result>::get",
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
//         typename detail::future_when_result<future, F>::type
        future<typename boost::result_of<F(future)>::type>
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
    future<Result> create_value(Result const& init)
    {
        return future<Result>(init);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class future<void>
    {
    private:
        BOOST_COPYABLE_AND_MOVABLE(future)

        typedef lcos::detail::future_data_base<void> future_data_type;

        explicit future(future_data_type* p)
          : future_data_(p)
        {}

        explicit future(boost::intrusive_ptr<future_data_type> p)
          : future_data_(p)
        {}

        friend class local::promise<void>;
        friend class local::packaged_task<void()>;
        friend class local::futures_factory<void()>;
        template <typename ContResult, typename Result_>
        friend class local::packaged_continuation;
        template <typename ContResult>
        friend struct local::detail::continuation_base;

        friend class promise<void, util::unused_type>;
        friend struct detail::future_data<void>;

        friend class hpx::thread;

        // create_void uses the dummy argument constructor below
        friend future<void> create_void();

        explicit future(int)
        {
            boost::intrusive_ptr<future_data_type> p =
                new lcos::detail::future_data<void>();
            static_cast<lcos::detail::future_data<void> *>(p.get())->
                set_data(util::unused);
            future_data_ = p;
        }

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
            if (this != &other)
                future_data_ = other.future_data_;
            return *this;
        }

        future& operator=(BOOST_RV_REF(future) other)
        {
            if (this != &other)
            {
                future_data_ = boost::move(other.future_data_);
                other.future_data_.reset();
            }
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
        future<typename boost::result_of<F(future)>::type>
//         typename detail::future_when_result<future, F>::type
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
    inline future<void> create_void()
    {
        return future<void>(1);   // dummy argument
    }
}}

#endif
