//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_FUTURE_MAR_06_2012_1059AM)
#define HPX_LCOS_FUTURE_MAR_06_2012_1059AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/config/forceinline.hpp>
#include <hpx/util/move.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/util/detail/remove_reference.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/function_types/result_type.hpp>
#include <boost/detail/iterator.hpp>
#include <boost/move/move.hpp>

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    class future
    {
    private:
        template <typename Result_>
        friend lcos::future<Result_> detail::make_future_from_data(
            boost::intrusive_ptr<detail::future_data_base<Result_> > const&);

        template <typename Result_>
        friend lcos::future<Result_> detail::make_future_from_data(
            BOOST_RV_REF(boost::intrusive_ptr<detail::future_data_base<Result_> >));

        template <typename Result_>
        friend lcos::future<Result_> detail::make_future_from_data(
            detail::future_data_base<Result_>* p);

        template <typename Result_>
        friend detail::future_data_base<Result_>*
            detail::get_future_data(lcos::future<Result_>&);

        template <typename Result_>
        friend detail::future_data_base<Result_> const*
            detail::get_future_data(lcos::future<Result_> const&);

    private:
        BOOST_COPYABLE_AND_MOVABLE(future)

        typedef lcos::detail::future_data_base<Result> future_data_type;

        explicit future(future_data_type* p)
          : future_data_(p)
        {}

        explicit future(boost::intrusive_ptr<future_data_type> const& p)
          : future_data_(p)
        {}

        explicit future(BOOST_RV_REF(boost::intrusive_ptr<future_data_type>) p)
        {
            future_data_.swap(p);
        }

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
        {
            future_data_.swap(other.future_data_);
        }

        // extension: init from given value, set future to ready right away
        explicit future(Result const& init)
        {
            typedef lcos::detail::future_data<Result> impl_type;
            boost::intrusive_ptr<future_data_type> p(new impl_type());
            static_cast<impl_type*>(p.get())->set_data(init);
            future_data_.swap(p);
        }

        explicit future(BOOST_RV_REF(Result) init)
        {
            typedef lcos::detail::future_data<Result> impl_type;
            boost::intrusive_ptr<future_data_type> p(new impl_type());
            static_cast<impl_type*>(p.get())->set_data(boost::move(init));
            future_data_.swap(p);
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
                future_data_.swap(other.future_data_);
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

        bool valid() const BOOST_NOEXCEPT
        {
            return future_data_.get() ? true : false;
        }

        // continuation support
        template <typename F>
        future<typename boost::result_of<F(future)>::type>
        then(BOOST_FWD_REF(F) f);

        // reset any pending continuation function
        void then()
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
    future<Result> make_ready_future(Result const& init)
    {
        return future<Result>(init);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class future<void>
    {
    private:
        template <typename Result_>
        friend lcos::future<Result_> detail::make_future_from_data(
            boost::intrusive_ptr<detail::future_data_base<Result_> > const&);

        template <typename Result_>
        friend lcos::future<Result_> detail::make_future_from_data(
            BOOST_RV_REF(boost::intrusive_ptr<detail::future_data_base<Result_> >));

        template <typename Result_>
        friend lcos::future<Result_> detail::make_future_from_data(
            detail::future_data_base<Result_>*);

        template <typename Result_>
        friend detail::future_data_base<Result_>*
            detail::get_future_data(lcos::future<Result_>&);

        template <typename Result_>
        friend detail::future_data_base<Result_> const*
            detail::get_future_data(lcos::future<Result_> const&);

        // make_future uses the dummy argument constructor below
        friend future<void> make_ready_future();

    private:
        BOOST_COPYABLE_AND_MOVABLE(future)

        typedef lcos::detail::future_data_base<void> future_data_type;

        explicit future(future_data_type* p)
          : future_data_(p)
        {}

        explicit future(boost::intrusive_ptr<future_data_type> const& p)
          : future_data_(p)
        {}

        explicit future(BOOST_RV_REF(boost::intrusive_ptr<future_data_type>) p)
        {
            future_data_.swap(p);
        }

        explicit future(int)
        {
            boost::intrusive_ptr<future_data_type> p(
                new lcos::detail::future_data<void>());
            static_cast<lcos::detail::future_data<void> *>(p.get())->
                set_data(util::unused);
            future_data_.swap(p);
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
        {
            future_data_.swap(other.future_data_);
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
                future_data_.swap(other.future_data_);
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
            return future_data_.get() ? true : false;
        }

        // continuation support
        template <typename F>
        future<typename boost::result_of<F(future)>::type>
        then(BOOST_FWD_REF(F) f);

        // reset any pending continuation function
        void then()
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
    inline future<void> make_ready_future()
    {
        return future<void>(1);   // dummy argument
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Result>
        inline lcos::future<Result> make_future_from_data(
            boost::intrusive_ptr<detail::future_data_base<Result> > const& p)
        {
            return lcos::future<Result>(p);
        }

        template <typename Result>
        inline lcos::future<Result> make_future_from_data( //-V659
            BOOST_RV_REF(boost::intrusive_ptr<detail::future_data_base<Result> >) p)
        {
            return lcos::future<Result>(boost::move(p));
        }

        template <typename Result>
        inline lcos::future<Result> make_future_from_data(
            detail::future_data_base<Result>* p)
        {
            return lcos::future<Result>(p);
        }

        template <typename Result>
        inline detail::future_data_base<Result>*
            get_future_data(lcos::future<Result>& f)
        {
            return f.future_data_.get();
        }

        template <typename Result>
        inline detail::future_data_base<Result> const*
            get_future_data(lcos::future<Result> const& f)
        {
            return f.future_data_.get();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct future_traits
    {
    };

    template <typename T>
    struct future_traits<lcos::future<T> >
    {
        typedef T value_type;
    };

    template <typename Iter>
    struct future_iterator_traits
    {
        typedef future_traits<
            typename boost::detail::iterator_traits<Iter>::value_type
        > traits_type;
    };

    template <typename T>
    struct future_iterator_traits<future<T> >
    {
    };
}}

#endif
