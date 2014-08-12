//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_PROMISE_MAR_01_2012_0121PM)
#define HPX_LCOS_LOCAL_PROMISE_MAR_01_2012_0121PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/unused.hpp>

#include <boost/exception_ptr.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/utility/swap.hpp>

namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        template <typename R>
        class promise_base
        {
            HPX_MOVABLE_BUT_NOT_COPYABLE(promise_base);

            typedef R result_type;
            typedef lcos::detail::future_data<R> shared_state_type;

        public:
            promise_base()
              : shared_state_(new shared_state_type())
              , future_retrieved_(false)
              , has_result_(false)
            {}

            promise_base(promise_base&& other) BOOST_NOEXCEPT
              : shared_state_(std::move(other.shared_state_))
              , future_retrieved_(other.future_retrieved_)
              , has_result_(other.has_result_)
            {
                other.shared_state_ = 0;
                other.future_retrieved_ = false;
                other.has_result_ = false;
            }

            ~promise_base()
            {
                if (shared_state_ != 0 && future_retrieved_ && !has_result_)
                {
                    shared_state_->set_error(broken_promise,
                        "promise_base<R>::~promise_base",
                        "abandoning not ready shared state");
                }
            }

            promise_base& operator=(promise_base&& other) BOOST_NOEXCEPT
            {
                if (this != &other)
                {
                    if (shared_state_ != 0 && future_retrieved_ && !has_result_)
                    {
                        shared_state_->set_error(broken_promise,
                            "promise_base<R>::operator=",
                            "abandoning not ready shared state");
                    }

                    shared_state_ = std::move(other.shared_state_);
                    future_retrieved_ = other.future_retrieved_;
                    has_result_ = other.has_result_;

                    other.shared_state_ = 0;
                    other.future_retrieved_ = false;
                    other.has_result_ = false;
                }
                return *this;
            }

            void swap(promise_base& other) BOOST_NOEXCEPT
            {
                boost::swap(shared_state_, other.shared_state_);
                boost::swap(future_retrieved_, other.future_retrieved_);
                boost::swap(has_result_, other.has_result_);
            }

            bool valid() const BOOST_NOEXCEPT
            {
                return shared_state_ != 0;
            }

            future<R> get_future(error_code& ec = throws)
            {
                if (future_retrieved_)
                {
                    HPX_THROWS_IF(ec, future_already_retrieved,
                        "promise_base<R>::get_future",
                        "future has already been retrieved from this promise");
                    return future<R>();
                }

                if (shared_state_ == 0)
                {
                    HPX_THROWS_IF(ec, no_state,
                        "promise_base<R>::get_future",
                        "this promise has no valid shared state");
                    return future<R>();
                }

                future_retrieved_ = true;
                return traits::future_access<future<R> >::create(shared_state_);
            }

            template <typename T>
            void set_result(T&& value, error_code& ec = throws)
            {
                if (has_result_)
                {
                    HPX_THROWS_IF(ec, promise_already_satisfied,
                        "promise_base<R>::set_result",
                        "result has already been stored for this promise");
                    return;
                }

                if (shared_state_ == 0)
                {
                    HPX_THROWS_IF(ec, no_state,
                        "promise_base<R>::set_result",
                        "this promise has no valid shared state");
                    return;
                }

                shared_state_->set_result(std::forward<T>(value), ec);
                if (ec) return;

                has_result_ = true;
            }

#ifndef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
            // [N3722, 4.1] asks for this...
            explicit operator lcos::future<R>()
            {
                return get_future();
            }

            explicit operator lcos::shared_future<R>()
            {
                return get_future();
            }
#endif

        private:
            boost::intrusive_ptr<shared_state_type> shared_state_;
            bool future_retrieved_;
            bool has_result_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    class promise : public detail::promise_base<R>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(promise);

        typedef detail::promise_base<R> base_type;

    public:
        // Effects: constructs a promise object and a shared state.
        promise()
          : base_type()
        {}

        // Effects: constructs a new promise object and transfers ownership of
        //          the shared state of other (if any) to the newly-
        //          constructed object.
        // Postcondition: other has no shared state.
        promise(promise&& other) BOOST_NOEXCEPT
          : base_type(std::move(other))
        {}

        // Effects: Abandons any shared state
        ~promise()
        {}

        // Effects: Abandons any shared state (30.6.4) and then as if
        //          promise(std::move(other)).swap(*this).
        // Returns: *this.
        promise& operator=(promise&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(std::move(other));
            return *this;
        }

        // Effects: Exchanges the shared state of *this and other.
        // Postcondition: *this has the shared state (if any) that other had
        //                prior to the call to swap. other has the shared state
        //                (if any) that *this had prior to the call to swap.
        void swap(promise& other) BOOST_NOEXCEPT
        {
            base_type::swap(other);
        }

        // Returns: true only if *this refers to a shared state.
        bool valid() const BOOST_NOEXCEPT
        {
            return base_type::valid();
        }

        // Returns: A future<R> object with the same shared state as *this.
        // Throws: future_error if *this has no shared state or if get_future
        //         has already been called on a promise with the same shared
        //         state as *this.
        // Error conditions:
        //   - future_already_retrieved if get_future has already been called
        //     on a promise with the same shared state as *this.
        //   - no_state if *this has no shared state.
        future<R> get_future(error_code& ec = throws)
        {
            return base_type::get_future(ec);
        }

        // Effects: atomically stores the value r in the shared state and makes
        //          that state ready (30.6.4).
        // Throws:
        //   - future_error if its shared state already has a stored value or
        //     exception, or
        //   - any exception thrown by the constructor selected to copy an
        //     object of R.
        // Error conditions:
        //   - promise_already_satisfied if its shared state already has a
        //     stored value or exception.
        //   - no_state if *this has no shared state.
        void set_value(R const& r, error_code& ec = throws)
        {
            base_type::set_result(r, ec);
        }

        // Effects: atomically stores the value r in the shared state and makes
        //          that state ready (30.6.4).
        // Throws:
        //   - future_error if its shared state already has a stored value or
        //     exception, or
        //   - any exception thrown by the constructor selected to move an
        //     object of R.
        // Error conditions:
        //   - promise_already_satisfied if its shared state already has a
        //     stored value or exception.
        //   - no_state if *this has no shared state.
        void set_value(R&& r, error_code& ec = throws)
        {
            base_type::set_result(std::move(r), ec);
        }

        // Effects: atomically stores the exception pointer p in the shared
        //          state and makes that state ready (30.6.4).
        // Throws: future_error if its shared state already has a stored value
        //         or exception.
        // Error conditions:
        //   - promise_already_satisfied if its shared state already has a
        //     stored value or exception.
        //   - no_state if *this has no shared state.
        void set_exception(boost::exception_ptr const& e, error_code& ec = throws)
        {
            base_type::set_result(e, ec);
        }
    };

    template <typename R>
    class promise<R&> : public detail::promise_base<R&>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(promise);

        typedef detail::promise_base<R&> base_type;

    public:
        // Effects: constructs a promise object and a shared state.
        promise()
          : base_type()
        {}

        // Effects: constructs a new promise object and transfers ownership of
        //          the shared state of other (if any) to the newly-
        //          constructed object.
        // Postcondition: other has no shared state.
        promise(promise&& other) BOOST_NOEXCEPT
          : base_type(std::move(other))
        {}

        // Effects: Abandons any shared state
        ~promise()
        {}

        // Effects: Abandons any shared state (30.6.4) and then as if
        //          promise(std::move(other)).swap(*this).
        // Returns: *this.
        promise& operator=(promise&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(std::move(other));
            return *this;
        }

        // Effects: Exchanges the shared state of *this and other.
        // Postcondition: *this has the shared state (if any) that other had
        //                prior to the call to swap. other has the shared state
        //                (if any) that *this had prior to the call to swap.
        void swap(promise& other) BOOST_NOEXCEPT
        {
            base_type::swap(other);
        }

        // Returns: true only if *this refers to a shared state.
        bool valid() const BOOST_NOEXCEPT
        {
            return base_type::valid();
        }

        // Returns: A future<R> object with the same shared state as *this.
        // Throws: future_error if *this has no shared state or if get_future
        //         has already been called on a promise with the same shared
        //         state as *this.
        // Error conditions:
        //   - future_already_retrieved if get_future has already been called
        //     on a promise with the same shared state as *this.
        //   - no_state if *this has no shared state.
        future<R&> get_future(error_code& ec = throws)
        {
            return base_type::get_future(ec);
        }

        // Effects: atomically stores the value r in the shared state and makes
        //          that state ready (30.6.4).
        // Throws:
        //   - future_error if its shared state already has a stored value or
        //     exception.
        // Error conditions:
        //   - promise_already_satisfied if its shared state already has a
        //     stored value or exception.
        //   - no_state if *this has no shared state.
        void set_value(R& r, error_code& ec = throws)
        {
            base_type::set_result(r, ec);
        }

        // Effects: atomically stores the exception pointer p in the shared
        //          state and makes that state ready (30.6.4).
        // Throws: future_error if its shared state already has a stored value
        //         or exception.
        // Error conditions:
        //   - promise_already_satisfied if its shared state already has a
        //     stored value or exception.
        //   - no_state if *this has no shared state.
        void set_exception(boost::exception_ptr const& e, error_code& ec = throws)
        {
            base_type::set_result(e, ec);
        }
    };

    template <>
    class promise<void> : public detail::promise_base<void>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(promise);

        typedef detail::promise_base<void> base_type;

    public:
        // Effects: constructs a promise object and a shared state.
        promise()
          : base_type()
        {}

        // Effects: constructs a new promise object and transfers ownership of
        //          the shared state of other (if any) to the newly-
        //          constructed object.
        // Postcondition: other has no shared state.
        promise(promise&& other) BOOST_NOEXCEPT
          : base_type(std::move(other))
        {}

        // Effects: Abandons any shared state
        ~promise()
        {}

        // Effects: Abandons any shared state (30.6.4) and then as if
        //          promise(std::move(other)).swap(*this).
        // Returns: *this.
        promise& operator=(promise&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(std::move(other));
            return *this;
        }

        // Effects: Exchanges the shared state of *this and other.
        // Postcondition: *this has the shared state (if any) that other had
        //                prior to the call to swap. other has the shared state
        //                (if any) that *this had prior to the call to swap.
        void swap(promise& other) BOOST_NOEXCEPT
        {
            base_type::swap(other);
        }

        // Returns: true only if *this refers to a shared state.
        bool valid() const BOOST_NOEXCEPT
        {
            return base_type::valid();
        }

        // Returns: A future<R> object with the same shared state as *this.
        // Throws: future_error if *this has no shared state or if get_future
        //         has already been called on a promise with the same shared
        //         state as *this.
        // Error conditions:
        //   - future_already_retrieved if get_future has already been called
        //     on a promise with the same shared state as *this.
        //   - no_state if *this has no shared state.
        future<void> get_future(error_code& ec = throws)
        {
            return base_type::get_future(ec);
        }

        // Effects: atomically stores the value r in the shared state and makes
        //          that state ready (30.6.4).
        // Throws:
        //   - future_error if its shared state already has a stored value or
        //     exception, or
        //   - any exception thrown by the constructor selected to copy an
        //     object of R.
        // Error conditions:
        //   - promise_already_satisfied if its shared state already has a
        //     stored value or exception.
        //   - no_state if *this has no shared state.
        void set_value(error_code& ec = throws)
        {
            base_type::set_result(hpx::util::unused, ec);
        }

        // Effects: atomically stores the exception pointer p in the shared
        //          state and makes that state ready (30.6.4).
        // Throws: future_error if its shared state already has a stored value
        //         or exception.
        // Error conditions:
        //   - promise_already_satisfied if its shared state already has a
        //     stored value or exception.
        //   - no_state if *this has no shared state.
        void set_exception(boost::exception_ptr const& e, error_code& ec = throws)
        {
            base_type::set_result(e, ec);
        }
    };

    template <typename R>
    void swap(promise<R>& x, promise<R>& y) BOOST_NOEXCEPT
    {
        x.swap(y);
    }
}}}

#ifdef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
namespace hpx { namespace lcos
{
    // [N3722, 4.1] asks for this...
    template <typename R>
    inline future<R>::future(local::promise<R>& promise)
    {
        promise.get_future().swap(*this);
    }

    template <typename R>
    inline shared_future<R>::shared_future(local::promise<R>& promise)
    {
        shared_future<R>(promise.get_future()).swap(*this);
    }

    // [N3722, 4.1] asks for this...
    template <>
    inline future<void>::future(local::promise<void>& promise)
    {
        promise.get_future().swap(*this);
    }

    template <>
    inline shared_future<void>::shared_future(local::promise<void>& promise)
    {
        shared_future<void>(promise.get_future()).swap(*this);
    }
}}
#endif

#endif
