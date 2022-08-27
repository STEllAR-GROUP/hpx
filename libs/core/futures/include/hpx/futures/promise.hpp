//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/allocator_support/allocator_deleter.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/type_support/unused.hpp>

#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx {

    namespace detail {

        template <typename R,
            typename SharedState = lcos::detail::future_data<R>>
        class promise_base;

        template <typename R, typename SharedState>
        class promise_base
        {
            using shared_state_type = SharedState;
            using init_no_addref = typename shared_state_type::init_no_addref;

        public:
            promise_base()
              : shared_state_(new shared_state_type(init_no_addref{}), false)
              , future_retrieved_(false)
              , shared_future_retrieved_(false)
            {
            }

            template <typename Allocator>
            promise_base(std::allocator_arg_t, Allocator const& a)
              : shared_state_()
              , future_retrieved_(false)
              , shared_future_retrieved_(false)
            {
                using allocator_shared_state_type =
                    typename traits::detail::shared_state_allocator<SharedState,
                        Allocator>::type;

                using other_allocator =
                    typename std::allocator_traits<Allocator>::
                        template rebind_alloc<allocator_shared_state_type>;
                using traits = std::allocator_traits<other_allocator>;
                using unique_pointer =
                    std::unique_ptr<allocator_shared_state_type,
                        util::allocator_deleter<other_allocator>>;

                other_allocator alloc(a);
                unique_pointer p(traits::allocate(alloc, 1),
                    util::allocator_deleter<other_allocator>{alloc});

                using lcos::detail::in_place;
                traits::construct(
                    alloc, p.get(), init_no_addref{}, in_place{}, alloc);
                shared_state_.reset(p.release(), false);
            }

            promise_base(promise_base&& other) noexcept
              : shared_state_(HPX_MOVE(other.shared_state_))
              , future_retrieved_(other.future_retrieved_)
              , shared_future_retrieved_(other.shared_future_retrieved_)
            {
                other.future_retrieved_ = false;
                other.shared_future_retrieved_ = false;
            }

            ~promise_base()
            {
                check_abandon_shared_state(
                    "detail::promise_base<R>::~promise_base()");
            }

            promise_base& operator=(promise_base&& other) noexcept
            {
                if (this != &other)
                {
                    this->check_abandon_shared_state(
                        "detail::promise_base<R>::operator=");

                    shared_state_ = HPX_MOVE(other.shared_state_);
                    future_retrieved_ = other.future_retrieved_;
                    shared_future_retrieved_ = other.shared_future_retrieved_;

                    other.future_retrieved_ = false;
                    other.shared_future_retrieved_ = false;
                }
                return *this;
            }

            void swap(promise_base& other) noexcept
            {
                std::swap(shared_state_, other.shared_state_);
                std::swap(future_retrieved_, other.future_retrieved_);
                std::swap(
                    shared_future_retrieved_, other.shared_future_retrieved_);
            }

            bool valid() const noexcept
            {
                return shared_state_ != nullptr;
            }

            hpx::future<R> get_future(error_code& ec = throws)
            {
                if (future_retrieved_ || shared_future_retrieved_)
                {
                    HPX_THROWS_IF(ec, future_already_retrieved,
                        "detail::promise_base<R>::get_future",
                        "future or shared future has already been retrieved "
                        "from this promise");
                    return hpx::future<R>();
                }

                if (shared_state_ == nullptr)
                {
                    HPX_THROWS_IF(ec, no_state,
                        "detail::promise_base<R>::get_future",
                        "this promise has no valid shared state");
                    return hpx::future<R>();
                }

                future_retrieved_ = true;
                return traits::future_access<hpx::future<R>>::create(
                    shared_state_);
            }

            hpx::shared_future<R> get_shared_future(error_code& ec = throws)
            {
                if (future_retrieved_)
                {
                    HPX_THROWS_IF(ec, future_already_retrieved,
                        "detail::promise_base<R>::get_shared_future",
                        "future has already been retrieved from this promise");
                    return hpx::shared_future<R>();
                }

                if (shared_state_ == nullptr)
                {
                    HPX_THROWS_IF(ec, no_state,
                        "detail::promise_base<R>::get_shared_future",
                        "this promise has no valid shared state");
                    return hpx::shared_future<R>();
                }

                shared_future_retrieved_ = true;
                return traits::future_access<hpx::shared_future<R>>::create(
                    shared_state_);
            }

            template <typename... Ts>
            std::enable_if_t<std::is_constructible_v<R, Ts&&...> ||
                std::is_void_v<R>>
            set_value(Ts&&... ts)
            {
                if (shared_state_ == nullptr)
                {
                    HPX_THROW_EXCEPTION(no_state,
                        "detail::promise_base<R>::set_value",
                        "this promise has no valid shared state");
                    return;
                }

                if (shared_state_->is_ready())
                {
                    HPX_THROW_EXCEPTION(promise_already_satisfied,
                        "detail::promise_base<R>::set_value",
                        "result has already been stored for this promise");
                    return;
                }

                shared_state_->set_value(HPX_FORWARD(Ts, ts)...);
            }

            template <typename T>
            void set_exception(T&& value)
            {
                if (shared_state_ == nullptr)
                {
                    HPX_THROW_EXCEPTION(no_state,
                        "detail::promise_base<R>::set_exception",
                        "this promise has no valid shared state");
                    return;
                }

                if (shared_state_->is_ready())
                {
                    HPX_THROW_EXCEPTION(promise_already_satisfied,
                        "detail::promise_base<R>::set_exception",
                        "result has already been stored for this promise");
                    return;
                }

                shared_state_->set_exception(HPX_FORWARD(T, value));
            }

        protected:
            void check_abandon_shared_state(const char* fun)
            {
                if (shared_state_ != nullptr &&
                    (future_retrieved_ || shared_future_retrieved_) &&
                    !shared_state_->is_ready())
                {
                    shared_state_->set_error(broken_promise, fun,
                        "abandoning not ready shared state");
                }
            }

            hpx::intrusive_ptr<shared_state_type> shared_state_;
            bool future_retrieved_;
            bool shared_future_retrieved_;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    class promise : public detail::promise_base<R>
    {
        using base_type = detail::promise_base<R>;

    public:
        // Effects: constructs a promise object and a shared state.
        promise() = default;

        // Effects: constructs a promise object and a shared state. The
        // constructor uses the allocator a to allocate the memory for the
        // shared state.
        template <typename Allocator>
        promise(std::allocator_arg_t, Allocator const& a)
          : base_type(std::allocator_arg, a)
        {
        }

        // Effects: constructs a new promise object and transfers ownership of
        //          the shared state of other (if any) to the newly-
        //          constructed object.
        // Postcondition: other has no shared state.
        promise(promise&& other) noexcept = default;

        // Effects: Abandons any shared state
        ~promise() = default;

        // Effects: Abandons any shared state (30.6.4) and then as if
        //          promise(HPX_MOVE(other)).swap(*this).
        // Returns: *this.
        promise& operator=(promise&& other) noexcept = default;

        // Effects: Exchanges the shared state of *this and other.
        // Postcondition: *this has the shared state (if any) that other had
        //                prior to the call to swap. other has the shared state
        //                (if any) that *this had prior to the call to swap.
        void swap(promise& other) noexcept
        {
            base_type::swap(other);
        }

        // Returns: true only if *this refers to a shared state.
        using base_type::valid;

        // Returns: A future<R> object with the same shared state as *this.
        // Throws: future_error if *this has no shared state or if get_future
        //         or get_shared_future has already been called on a promise
        //         with the same shared state as *this.
        // Error conditions:
        //   - future_already_retrieved if get_future or get_shared_future has
        //     already been called on a promise with the same shared state as
        //     *this.
        //   - no_state if *this has no shared state.
        using base_type::get_future;

        // Returns: A shared_future<R> object with the same shared state as *this.
        // Throws: future_error if *this has no shared state or if
        //         get_shared_future has already been called on a promise
        //         with the same shared state as *this.
        // Error conditions:
        //   - future_already_retrieved if get_shared_future has already been
        //     called on a promise with the same shared state as *this.
        //   - no_state if *this has no shared state.
        using base_type::get_shared_future;

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
        void set_value(R const& r)
        {
            base_type::set_value(r);
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
        void set_value(R&& r)
        {
            base_type::set_value(HPX_MOVE(r));
        }

        // Extension (see wg21.link/P0319)
        //
        // Effects: atomically initializes the stored value as if
        //          direct-non-list-initializing an object of type R with the
        //          arguments forward<Args>(args)...) in the shared state and
        //          makes that state ready.
        // Requires:
        //      - std::is_constructible<R, Ts&&...>::value == true
        // Throws:
        //   - future_error if its shared state already has a stored value or
        //     exception, or
        //   - any exception thrown by the constructor selected to move an
        //     object of R.
        // Error conditions:
        //   - promise_already_satisfied if its shared state already has a
        //     stored value or exception.
        //   - no_state if *this has no shared state.
        template <typename... Ts>
        void set_value(Ts&&... ts)
        {
            base_type::set_value(HPX_FORWARD(Ts, ts)...);
        }

        // Effects: atomically stores the exception pointer p in the shared
        //          state and makes that state ready (30.6.4).
        // Throws: future_error if its shared state already has a stored value
        //         or exception.
        // Error conditions:
        //   - promise_already_satisfied if its shared state already has a
        //     stored value or exception.
        //   - no_state if *this has no shared state.
        using base_type::set_exception;
    };

    template <typename R>
    class promise<R&> : public detail::promise_base<R&>
    {
        using base_type = detail::promise_base<R&>;

    public:
        // Effects: constructs a promise object and a shared state.
        promise() = default;

        // Effects: constructs a promise object and a shared state. The
        // constructor uses the allocator a to allocate the memory for the
        // shared state.
        template <typename Allocator>
        promise(std::allocator_arg_t, Allocator const& a)
          : base_type(std::allocator_arg, a)
        {
        }

        // Effects: constructs a new promise object and transfers ownership of
        //          the shared state of other (if any) to the newly-
        //          constructed object.
        // Postcondition: other has no shared state.
        promise(promise&& other) noexcept = default;

        // Effects: Abandons any shared state
        ~promise() = default;

        // Effects: Abandons any shared state (30.6.4) and then as if
        //          promise(HPX_MOVE(other)).swap(*this).
        // Returns: *this.
        promise& operator=(promise&& other) noexcept = default;

        // Effects: Exchanges the shared state of *this and other.
        // Postcondition: *this has the shared state (if any) that other had
        //                prior to the call to swap. other has the shared state
        //                (if any) that *this had prior to the call to swap.
        void swap(promise& other) noexcept
        {
            base_type::swap(other);
        }

        // Returns: true only if *this refers to a shared state.
        using base_type::valid;

        // Returns: A future<R&> object with the same shared state as *this.
        // Throws: future_error if *this has no shared state or if get_future
        //         or get_shared_future has already been called on a promise
        //         with the same shared state as *this.
        // Error conditions:
        //   - future_already_retrieved if get_future or get_shared_future has
        //     already been called on a promise with the same shared state as
        //     *this.
        //   - no_state if *this has no shared state.
        using base_type::get_future;

        // Returns: A shared_future<R&> object with the same shared state as *this.
        // Throws: future_error if *this has no shared state or if
        //         get_shared_future has already been called on a promise
        //         with the same shared state as *this.
        // Error conditions:
        //   - future_already_retrieved if get_shared_future has already been
        //     called on a promise with the same shared state as *this.
        //   - no_state if *this has no shared state.
        using base_type::get_shared_future;

        // Effects: atomically stores the value r in the shared state and makes
        //          that state ready (30.6.4).
        // Throws:
        //   - future_error if its shared state already has a stored value or
        //     exception.
        // Error conditions:
        //   - promise_already_satisfied if its shared state already has a
        //     stored value or exception.
        //   - no_state if *this has no shared state.
        void set_value(R& r)
        {
            base_type::set_value(r);
        }

        // Effects: atomically stores the exception pointer p in the shared
        //          state and makes that state ready (30.6.4).
        // Throws: future_error if its shared state already has a stored value
        //         or exception.
        // Error conditions:
        //   - promise_already_satisfied if its shared state already has a
        //     stored value or exception.
        //   - no_state if *this has no shared state.
        using base_type::set_exception;
    };

    template <>
    class promise<void> : public detail::promise_base<void>
    {
        using base_type = detail::promise_base<void>;

    public:
        // Effects: constructs a promise object and a shared state.
        promise() = default;

        // Effects: constructs a promise object and a shared state. The
        // constructor uses the allocator a to allocate the memory for the
        // shared state.
        template <typename Allocator>
        promise(std::allocator_arg_t, Allocator const& a)
          : base_type(std::allocator_arg, a)
        {
        }

        // Effects: constructs a new promise object and transfers ownership of
        //          the shared state of other (if any) to the newly-
        //          constructed object.
        // Postcondition: other has no shared state.
        promise(promise&& other) noexcept = default;

        // Effects: Abandons any shared state
        ~promise() = default;

        // Effects: Abandons any shared state (30.6.4) and then as if
        //          promise(HPX_MOVE(other)).swap(*this).
        // Returns: *this.
        promise& operator=(promise&& other) noexcept = default;

        // Effects: Exchanges the shared state of *this and other.
        // Postcondition: *this has the shared state (if any) that other had
        //                prior to the call to swap. other has the shared state
        //                (if any) that *this had prior to the call to swap.
        void swap(promise& other) noexcept
        {
            base_type::swap(other);
        }

        // Returns: true only if *this refers to a shared state.
        using base_type::valid;

        // Returns: A future<R> object with the same shared state as *this.
        // Throws: future_error if *this has no shared state or if get_future
        //         or get_shared_future has already been called on a promise
        //         with the same shared state as *this.
        // Error conditions:
        //   - future_already_retrieved if get_future or get_shared_future has
        //     already been called on a promise with the same shared state as
        //     *this.
        //   - no_state if *this has no shared state.
        using base_type::get_future;

        // Returns: A shared_future<R> object with the same shared state as *this.
        // Throws: future_error if *this has no shared state or if
        //         get_shared_future has already been called on a promise
        //         with the same shared state as *this.
        // Error conditions:
        //   - future_already_retrieved if get_shared_future has already been
        //     called on a promise with the same shared state as *this.
        //   - no_state if *this has no shared state.
        using base_type::get_shared_future;

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
        void set_value()
        {
            base_type::set_value(hpx::util::unused);
        }

        // Effects: atomically stores the exception pointer p in the shared
        //          state and makes that state ready (30.6.4).
        // Throws: future_error if its shared state already has a stored value
        //         or exception.
        // Error conditions:
        //   - promise_already_satisfied if its shared state already has a
        //     stored value or exception.
        //   - no_state if *this has no shared state.
        using base_type::set_exception;
    };

    template <typename R>
    void swap(promise<R>& x, promise<R>& y) noexcept
    {
        x.swap(y);
    }
}    // namespace hpx

namespace hpx::lcos::local {

    template <typename R>
    using promise HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::local::promise is deprecated, use hpx::promise instead") =
        hpx::promise<R>;
}

namespace std {

    // Requires: Allocator shall be an allocator (17.6.3.5)
    template <typename R, typename Allocator>
    struct uses_allocator<hpx::promise<R>, Allocator> : std::true_type
    {
    };
}    // namespace std
