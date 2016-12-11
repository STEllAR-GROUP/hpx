//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_PROMISE_MAR_01_2012_0121PM)
#define HPX_LCOS_LOCAL_PROMISE_MAR_01_2012_0121PM

#include <hpx/config.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/util/unused.hpp>

#include <boost/exception_ptr.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/utility/swap.hpp>

#include <utility>

namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        template <typename R,
            typename SharedState = lcos::detail::future_data<R>>
        class promise_base
        {
            HPX_MOVABLE_ONLY(promise_base);

            typedef SharedState shared_state_type;
            typedef typename shared_state_type::init_no_addref init_no_addref;

        public:
            promise_base()
              : shared_state_(new shared_state_type(init_no_addref()), false)
              , future_retrieved_(false)
            {}

            promise_base(promise_base&& other) HPX_NOEXCEPT
              : shared_state_(std::move(other.shared_state_))
              , future_retrieved_(other.future_retrieved_)
            {
                other.shared_state_ = nullptr;
                other.future_retrieved_ = false;
            }

            ~promise_base()
            {
                check_abandon_shared_state(
                    "local::detail::promise_base<R>::~promise_base()");
            }

            promise_base& operator=(promise_base&& other) HPX_NOEXCEPT
            {
                if (this != &other)
                {
                    this->check_abandon_shared_state(
                        "local::detail::promise_base<R>::operator=");

                    shared_state_ = std::move(other.shared_state_);
                    future_retrieved_ = other.future_retrieved_;

                    other.shared_state_ = nullptr;
                    other.future_retrieved_ = false;
                }
                return *this;
            }

            void swap(promise_base& other) HPX_NOEXCEPT
            {
                boost::swap(shared_state_, other.shared_state_);
                boost::swap(future_retrieved_, other.future_retrieved_);
            }

            bool valid() const HPX_NOEXCEPT
            {
                return shared_state_ != nullptr;
            }

            future<R> get_future(error_code& ec = throws)
            {
                if (future_retrieved_)
                {
                    HPX_THROWS_IF(ec, future_already_retrieved,
                        "local::detail::promise_base<R>::get_future",
                        "future has already been retrieved from this promise");
                    return future<R>();
                }

                if (shared_state_ == nullptr)
                {
                    HPX_THROWS_IF(ec, no_state,
                        "local::detail::promise_base<R>::get_future",
                        "this promise has no valid shared state");
                    return future<R>();
                }

                future_retrieved_ = true;
                return traits::future_access<future<R> >::create(shared_state_);
            }

            template <typename T>
            void set_value(T&& value, error_code& ec = throws)
            {
                if (shared_state_ == nullptr)
                {
                    HPX_THROWS_IF(ec, no_state,
                        "local::detail::promise_base<R>::set_value",
                        "this promise has no valid shared state");
                    return;
                }

                if (shared_state_->is_ready())
                {
                    HPX_THROWS_IF(ec, promise_already_satisfied,
                        "local::detail::promise_base<R>::set_value",
                        "result has already been stored for this promise");
                    return;
                }

                shared_state_->set_value(std::forward<T>(value), ec);
                if (ec) return;
            }

            template <typename T>
            void set_exception(T&& value, error_code& ec = throws)
            {
                if (shared_state_ == nullptr)
                {
                    HPX_THROWS_IF(ec, no_state,
                        "local::detail::promise_base<R>::set_exception",
                        "this promise has no valid shared state");
                    return;
                }

                if (shared_state_->is_ready())
                {
                    HPX_THROWS_IF(ec, promise_already_satisfied,
                        "local::detail::promise_base<R>::set_exception",
                        "result has already been stored for this promise");
                    return;
                }

                shared_state_->set_exception(std::forward<T>(value), ec);
                if (ec) return;
            }

        protected:
            void check_abandon_shared_state(const char* fun)
            {
                if (shared_state_ != nullptr && future_retrieved_ &&
                    !shared_state_->is_ready())
                {
                    shared_state_->set_error(broken_promise, fun,
                        "abandoning not ready shared state");
                }
            }

            boost::intrusive_ptr<shared_state_type> shared_state_;
            bool future_retrieved_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    class promise : public detail::promise_base<R>
    {
        HPX_MOVABLE_ONLY(promise);

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
        promise(promise&& other) HPX_NOEXCEPT
          : base_type(std::move(other))
        {}

        // Effects: Abandons any shared state
        ~promise()
        {}

        // Effects: Abandons any shared state (30.6.4) and then as if
        //          promise(std::move(other)).swap(*this).
        // Returns: *this.
        promise& operator=(promise&& other) HPX_NOEXCEPT
        {
            base_type::operator=(std::move(other));
            return *this;
        }

        // Effects: Exchanges the shared state of *this and other.
        // Postcondition: *this has the shared state (if any) that other had
        //                prior to the call to swap. other has the shared state
        //                (if any) that *this had prior to the call to swap.
        void swap(promise& other) HPX_NOEXCEPT
        {
            base_type::swap(other);
        }

        // Returns: true only if *this refers to a shared state.
        bool valid() const HPX_NOEXCEPT
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
            base_type::set_value(r, ec);
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
            base_type::set_value(std::move(r), ec);
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
            base_type::set_exception(e, ec);
        }
    };

    template <typename R>
    class promise<R&> : public detail::promise_base<R&>
    {
        HPX_MOVABLE_ONLY(promise);

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
        promise(promise&& other) HPX_NOEXCEPT
          : base_type(std::move(other))
        {}

        // Effects: Abandons any shared state
        ~promise()
        {}

        // Effects: Abandons any shared state (30.6.4) and then as if
        //          promise(std::move(other)).swap(*this).
        // Returns: *this.
        promise& operator=(promise&& other) HPX_NOEXCEPT
        {
            base_type::operator=(std::move(other));
            return *this;
        }

        // Effects: Exchanges the shared state of *this and other.
        // Postcondition: *this has the shared state (if any) that other had
        //                prior to the call to swap. other has the shared state
        //                (if any) that *this had prior to the call to swap.
        void swap(promise& other) HPX_NOEXCEPT
        {
            base_type::swap(other);
        }

        // Returns: true only if *this refers to a shared state.
        bool valid() const HPX_NOEXCEPT
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
            base_type::set_value(r, ec);
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
            base_type::set_exception(e, ec);
        }
    };

    template <>
    class promise<void> : public detail::promise_base<void>
    {
        HPX_MOVABLE_ONLY(promise);

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
        promise(promise&& other) HPX_NOEXCEPT
          : base_type(std::move(other))
        {}

        // Effects: Abandons any shared state
        ~promise()
        {}

        // Effects: Abandons any shared state (30.6.4) and then as if
        //          promise(std::move(other)).swap(*this).
        // Returns: *this.
        promise& operator=(promise&& other) HPX_NOEXCEPT
        {
            base_type::operator=(std::move(other));
            return *this;
        }

        // Effects: Exchanges the shared state of *this and other.
        // Postcondition: *this has the shared state (if any) that other had
        //                prior to the call to swap. other has the shared state
        //                (if any) that *this had prior to the call to swap.
        void swap(promise& other) HPX_NOEXCEPT
        {
            base_type::swap(other);
        }

        // Returns: true only if *this refers to a shared state.
        bool valid() const HPX_NOEXCEPT
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
            base_type::set_value(hpx::util::unused, ec);
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
            base_type::set_exception(e, ec);
        }
    };

    template <typename R>
    void swap(promise<R>& x, promise<R>& y) HPX_NOEXCEPT
    {
        x.swap(y);
    }
}}}

#endif
