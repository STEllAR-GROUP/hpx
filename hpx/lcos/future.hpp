//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_FUTURE_MAR_06_2012_1059AM)
#define HPX_LCOS_FUTURE_MAR_06_2012_1059AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/config/forceinline.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/util/detail/remove_reference.hpp>
#include <hpx/runtime/actions/continuation.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/function_types/result_type.hpp>
#include <boost/detail/iterator.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_void.hpp>
#include <boost/mpl/if.hpp>

namespace hpx { namespace lcos
{
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

    template <typename T>
    struct future_traits<lcos::future<T> const>
    {
        typedef T value_type;
    };

    template <typename T>
    struct future_traits<lcos::future<T> &>
    {
        typedef T value_type;
    };

    template <typename T>
    struct future_traits<lcos::future<T> const &>
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

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    class future
    {
    public:
        typedef lcos::detail::future_data_base<Result> future_data_type;

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

        // accept wrapped future
        future(BOOST_RV_REF(future<future>) other)
        {
            future f = boost::move(other.unwrap());
            (*this).swap(f);
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
            return boost::move(future_data_->move_data(ec));
        }

        // state introspection
        bool ready() const
        {
            return future_data_ && future_data_->ready();
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
        bool cancelable() const
        {
            return future_data_->cancelable();
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

        template <typename F>
        future<typename boost::result_of<F(future)>::type>
        then(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f);

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
        wait_until(boost::chrono::time_point<Clock, Duration> const& abs_time) 
        {
            return wait_until(util::to_ptime(abs_time));
        }

        template <typename Rep, typename Period>
        BOOST_SCOPED_ENUM(future_status)
        wait_for(boost::chrono::duration<Rep, Period> const& rel_time)
        {
            return wait_for(util::to_time_duration(rel_time));
        }

        future<typename future_traits<
            typename boost::mpl::if_<
                traits::is_future<Result>, Result, future<void>
            >::type
        >::value_type> unwrap(error_code& ec = throws);

    private:
        template <typename InnerResult, typename UnwrapResult>
        void on_inner_ready(future<InnerResult>& inner,
            boost::intrusive_ptr<lcos::detail::future_data<UnwrapResult> > p);

        template <typename UnwrapResult>
        void on_outer_ready(
            boost::intrusive_ptr<lcos::detail::future_data<UnwrapResult> > p);

    private:
        boost::intrusive_ptr<future_data_type> future_data_;
    };

    // extension: create a pre-initialized future object
    template <typename Result>
    future<typename util::detail::remove_reference<Result>::type>
    make_ready_future(BOOST_FWD_REF(Result) init)
    {
        return future<typename util::detail::remove_reference<Result>::type>(
            boost::forward<Result>(init));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class future<void>
    {
    public:
        typedef lcos::detail::future_data_base<void> future_data_type;

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

        // accept wrapped future
        future(BOOST_RV_REF(future<future>) other)
        {
            future f = boost::move(other.unwrap());
            (*this).swap(f);
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
        bool ready() const
        {
            return future_data_->ready();
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
        bool cancelable() const
        {
            return future_data_->cancelable();
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

        template <typename F>
        future<typename boost::result_of<F(future)>::type>
        then(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f);

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
        wait_until(boost::chrono::time_point<Clock, Duration> const& abs_time)
        {
            return wait_until(util::to_ptime(abs_time));
        }

        template <typename Rep, typename Period>
        BOOST_SCOPED_ENUM(future_status)
        wait_for(boost::chrono::duration<Rep, Period> const& rel_time)
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
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    // special handling of actions returning a future
    template <typename Result>
    struct typed_continuation<lcos::future<Result> > : continuation
    {
        typed_continuation()
        {}

        explicit typed_continuation(naming::id_type const& gid)
          : continuation(gid)
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid,
                BOOST_FWD_REF(F) f)
          : continuation(gid), f_(boost::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(BOOST_FWD_REF(F) f)
          : f_(boost::forward<F>(f))
        {}

        virtual ~typed_continuation()
        {
            detail::guid_initialization<typed_continuation>();
        }

        void deferred_trigger(lcos::future<Result> result) const
        {
            if (f_.empty()) {
                if (!this->get_gid()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<lcos::future<Result> >::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                hpx::set_lco_value(this->get_gid(), result.move());
            }
            else {
                f_(this->get_gid(), result.move());
            }
        }

        void trigger_value(BOOST_RV_REF(lcos::future<Result>) result) const
        {
            LLCO_(info)
                << "typed_continuation<lcos::future<hpx::lcos::future<Result> > >::trigger("
                << this->get_gid() << ")";

            // attach continuation to this future which will send the result back
            // once its ready
            deferred_result_ = result.then(
                util::bind(&typed_continuation::deferred_trigger,
                    boost::static_pointer_cast<typed_continuation const>(shared_from_this()),
                    util::placeholders::_1));
        }

        static void register_base()
        {
            util::void_cast_register_nonvirt<typed_continuation, continuation>();
        }

    private:
        /// serialization support
        friend class boost::serialization::access;
        typedef continuation base_type;

        template <class Archive>
        BOOST_FORCEINLINE void serialize(Archive& ar, const unsigned int /*version*/)
        {
            // serialize function
            bool have_function = !f_.empty();
            ar & have_function;
            if (have_function)
                ar & f_;

            // serialize base class
            ar & util::base_object_nonvirt<base_type>(*this);
        }

        util::function<void(naming::id_type, Result)> f_;
        mutable lcos::future<void> deferred_result_;
    };
}}

#endif
