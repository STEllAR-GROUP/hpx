//  Copyright (c)2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_THREADING_STOP_TOKEN_HPP
#define HPX_THREADING_STOP_TOKEN_HPP

#include <hpx/config.hpp>
#include <hpx/basic_execution.hpp>
#include <hpx/memory.hpp>
#include <hpx/thread_support.hpp>
#include <hpx/threading/thread.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
// This header holds components that can be used to asynchronously request that
// an operation stops execution in a timely manner, typically because the
// result is no longer required. Such a request is called a stop request.
//
// stop_source, stop_token, and stop_callback implement semantics of shared
// ownership of a stop state. Any stop_source, stop_token, or stop_callback
// that shares ownership of the same stop state is an associated stop_source,
// stop_token, or stop_callback, respectively. The last remaining owner of the
// stop state automatically releases the resources associated with the stop
// state.
//
// A stop_token can be passed to an operation which can either:
//  - actively poll the token to check if there has been a stop request, or
//  - register a callback using the stop_callback class template which will
//      be called in the event that a stop request is made.
// A stop request made via a stop_source will be visible to all associated
// stop_token and stop_source objects. Once a stop request has been made it
// cannot be withdrawn (a subsequent stop request has no effect).
//
// Callbacks registered via a stop_callback object are called when a stop
// request is first made by any associated stop_source object.
//
// Calls to the functions request_stop, stop_requested, and stop_possible
// do not introduce data races. A call to request_stop that returns true
// synchronizes with a call to stop_requested on an associated stop_token
// or stop_source object that returns true. Registration of a callback
// synchronizes with the invocation of that callback.

namespace hpx {

    namespace detail {

        struct stop_state;
        HPX_EXPORT void intrusive_ptr_add_ref(stop_state* p);
        HPX_EXPORT void intrusive_ptr_release(stop_state* p);

        ///////////////////////////////////////////////////////////////////////
        struct stop_callback_base
        {
            stop_callback_base* next_ = nullptr;
            stop_callback_base** prev_ = nullptr;
            bool* is_removed_ = nullptr;

            std::atomic<bool> callback_finished_executing_{false};

            virtual void execute() noexcept = 0;

            void add_this_callback(stop_callback_base*& callbacks);
            bool remove_this_callback();

        protected:
            ~stop_callback_base() = default;
        };

        ///////////////////////////////////////////////////////////////////////
        struct stop_state
        {
        private:
            // bits 0-30 - token ref count (31 bits)
            static constexpr std::uint64_t token_ref_increment = 1ull;
            static constexpr std::uint64_t token_ref_mask = 0x7fffffffull;
            // bit 31 - stop-requested
            static constexpr std::uint64_t stop_requested_flag = 1ull << 31;

            // bits 32-62 - source ref count (31 bits)
            static constexpr std::uint64_t source_ref_increment =
                token_ref_increment << 32;
            static constexpr std::uint64_t source_ref_mask = token_ref_mask
                << 32;
            // bit 63 - locked
            static constexpr std::uint64_t locked_flag = stop_requested_flag
                << 32;

        public:
            stop_state()
              : state_(token_ref_increment)
            {
            }

            bool stop_requested() const noexcept
            {
                return stop_requested(state_.load(std::memory_order_acquire));
            }

            // Returns: false if a stop request was not made and there are no
            //      associated stop_source objects; otherwise, true.
            bool stop_possible() const noexcept
            {
                return stop_possible(state_.load(std::memory_order_acquire));
            }

            HPX_EXPORT bool request_stop() noexcept;

            void add_source_count()
            {
                state_.fetch_add(stop_state::source_ref_increment,
                    std::memory_order_relaxed);
            }

            void remove_source_count()
            {
                state_.fetch_sub(stop_state::source_ref_increment,
                    std::memory_order_acq_rel);
            }

            HPX_EXPORT bool add_callback(stop_callback_base* cb) noexcept;
            HPX_EXPORT void remove_callback(stop_callback_base* cb) noexcept;

        private:
            static bool is_locked(std::uint64_t state) noexcept
            {
                return (state & stop_state::locked_flag) != 0;
            }

            static bool stop_requested(std::uint64_t state) noexcept
            {
                return (state & stop_state::stop_requested_flag) != 0;
            }

            static bool stop_possible(std::uint64_t state) noexcept
            {
                // Stop may happen, if it has already been requested or if
                // there are still interrupt_source instances in existence.
                return stop_requested(state) ||
                    (state & stop_state::source_ref_mask) != 0;
            }

            // Effects: locks the state and atomically sets stop-requested
            //      signal
            //
            // Returns: false if stop was requested; otherwise, true.
            HPX_EXPORT bool lock_and_request_stop() noexcept;

            // Effect: locks the state if no stop was requested and stop is
            //      possible. Also executes callbacks if stop was requested.
            //
            // Returns: false if stop was requested or stop is not possible
            HPX_EXPORT bool lock_if_not_stopped(
                stop_callback_base* cb) noexcept;

        public:
            // Effect: locks the state
            HPX_EXPORT void lock() noexcept;

            void unlock() noexcept
            {
                state_.fetch_sub(locked_flag, std::memory_order_release);
            }

        private:
            friend struct scoped_lock_if_not_stopped;
            friend struct scoped_lock_and_request_stop;

            friend HPX_EXPORT void intrusive_ptr_add_ref(stop_state* p);
            friend HPX_EXPORT void intrusive_ptr_release(stop_state* p);

            std::atomic<std::uint64_t> state_;
            stop_callback_base* callbacks_ = nullptr;
            hpx::thread::id signalling_thread_;
        };

    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    //
    // 32.3.3, class stop_token
    //
    // The class stop_token provides an interface for querying whether a
    // stop request has been made (stop_requested) or can ever be made
    // (stop_possible) using an associated stop_source object (32.3.4). A
    // stop_token can also be passed to a stop_callback (32.3.5) constructor
    // to register a callback to be called when a stop request has been made
    // from an associated stop_source.
    class stop_token
    {
    public:
        // 32.3.3.1 constructors, copy, and assignment

        // Postconditions: stop_possible() is false and stop_requested() is
        //      false. [Note: Because the created stop_token object can never
        //      receive a stop request, no resources are allocated for a stop
        //      state]
        stop_token() noexcept = default;

        stop_token(stop_token const& rhs) noexcept
          : state_(rhs.state_)
        {
        }
        stop_token(stop_token&&) noexcept = default;

        stop_token& operator=(stop_token const& rhs) noexcept
        {
            state_ = rhs.state_;
            return *this;
        }
        stop_token& operator=(stop_token&&) noexcept = default;

        // Effects: Releases ownership of the stop state, if any.
        ~stop_token() = default;

        // Effects: Exchanges the values of *this and rhs.
        void swap(stop_token& s) noexcept
        {
            std::swap(state_, s.state_);
        }

        // 32.3.3.2 stop handling

        // Returns: true if *this has ownership of a stop state that has
        //      received a stop request; otherwise, false.
        HPX_NODISCARD bool stop_requested() const noexcept
        {
            return !!state_ && state_->stop_requested();
        }

        //  Returns: false if:
        //      (2.1) *this does not have ownership of a stop state, or
        //      (2.2) a stop request was not made and there are no associated
        //            stop_source objects; otherwise, true.
        HPX_NODISCARD bool stop_possible() const noexcept
        {
            return !!state_ && state_->stop_possible();
        }

        // 32.3.3.3 Comparisons

        // Returns: true if lhs and rhs have ownership of the same stop state or
        //      if both lhs and rhs do not have ownership of a stop state;
        //      otherwise false.
        HPX_NODISCARD friend bool operator==(
            stop_token const& lhs, stop_token const& rhs) noexcept
        {
            return lhs.state_ == rhs.state_;
        }

        // Returns: !(lhs==rhs).
        HPX_NODISCARD friend bool operator!=(
            stop_token const& lhs, stop_token const& rhs) noexcept
        {
            return !(lhs == rhs);
        }

    private:
        template <typename Callback>
        friend class stop_callback;
        friend class stop_source;

        stop_token(hpx::memory::intrusive_ptr<detail::stop_state> const& state)
          : state_(state)
        {
        }

    private:
        hpx::memory::intrusive_ptr<detail::stop_state> state_;
    };

    ///////////////////////////////////////////////////////////////////////////
    //
    // 32.3.4, class stop_source
    //
    // The class stop_source implements the semantics of making a
    // stop request. A stop request made on a stop_source object is
    // visible to all associated stop_source and stop_token (32.3.3)
    // objects. Once a stop request has been made it cannot be
    // withdrawn (a subsequent stop request has no effect).

    // no-shared-stop-state indicator
    struct nostopstate_t
    {
        explicit nostopstate_t() = default;
    };

#if defined(HPX_HAVE_CXX17_INLINE_VARIABLE)
    inline constexpr nostopstate_t nostopstate{};
#else
    static constexpr nostopstate_t nostopstate{};
#endif

    class stop_source
    {
    public:
        // 32.3.4.1 constructors, copy, and assignment

        // Effects: Initialises *this to have ownership of a new stop state.
        //
        // Postconditions: stop_possible() is true and stop_requested() is false.
        //
        // Throws: std::bad_alloc if memory could not be allocated for the stop
        //      state.
        stop_source()
          : state_(new detail::stop_state, false)
        {
            state_->add_source_count();
        }

        // Postconditions: stop_possible() is false and stop_requested() is
        //      false. [Note: No resources are allocated for the state.]
        explicit stop_source(nostopstate_t) noexcept {}

        stop_source(stop_source const& rhs) noexcept
          : state_(rhs.state_)
        {
            if (state_)
                state_->add_source_count();
        }
        stop_source(stop_source&&) noexcept = default;

        stop_source& operator=(stop_source const& rhs) noexcept
        {
            state_ = rhs.state_;
            if (state_)
                state_->add_source_count();
            return *this;
        }
        stop_source& operator=(stop_source&&) noexcept = default;

        // Effects: Releases ownership of the stop state, if any.
        ~stop_source()
        {
            if (state_)
                state_->remove_source_count();
        }

        // Effects: Exchanges the values of *this and rhs.
        void swap(stop_source& s) noexcept
        {
            std::swap(state_, s.state_);
        }

        // 32.3.4.2 stop handling

        // Returns: stop_token() if stop_possible() is false; otherwise a new
        //      associated stop_token object.
        HPX_NODISCARD stop_token get_token() const noexcept
        {
            if (!stop_possible())
            {
                return stop_token();
            }
            return stop_token(state_);
        }

        // Returns: true if *this has ownership of a stop state; otherwise, false.
        HPX_NODISCARD bool stop_possible() const noexcept
        {
            return !!state_;
        }

        // Returns: true if *this has ownership of a stop state that has
        //      received a stop request; otherwise, false.
        HPX_NODISCARD bool stop_requested() const noexcept
        {
            return !!state_ && state_->stop_requested();
        }

        // Effects: If *this does not have ownership of a stop state, returns
        //      false. Otherwise, atomically determines whether the owned
        //      stop state has received a stop request, and if not, makes a
        //      stop request. The determination and making of the stop
        //      request are an atomic read-modify-write operation (6.9.2.1).
        //      If the request was made, the callbacks registered by
        //      associated stop_callback objects are synchronously called.
        //      If an invocation of a callback exits via an exception then
        //      std::terminate is called (14.6.1). [Note: A stop request
        //      includes notifying all condition variables of type
        //      condition_variable_any temporarily registered during an
        //      interruptible wait (32.6.4.2). end note]
        //
        // Postconditions: stop_possible() is false or stop_requested() is true.
        //
        // Returns: true if this call made a stop request; otherwise false
        bool request_stop() noexcept
        {
            return !!state_ && state_->request_stop();
        }

        // 32.3.4.3 Comparisons

        // Returns: true if lhs and rhs have ownership of the same stop state or
        //      if both lhs and rhs do not have ownership of a stop state;
        //      otherwise false.
        HPX_NODISCARD friend bool operator==(
            stop_source const& lhs, stop_source const& rhs) noexcept
        {
            return lhs.state_ == rhs.state_;
        }

        HPX_NODISCARD friend bool operator!=(
            stop_source const& lhs, stop_source const& rhs) noexcept
        {
            return !(lhs == rhs);
        }

    private:
        hpx::memory::intrusive_ptr<detail::stop_state> state_;
    };

    ///////////////////////////////////////////////////////////////////////////
    //
    // 32.3.5, class stop_callback
    //
    template <typename Callback>
    class HPX_NODISCARD stop_callback : private detail::stop_callback_base
    {
    public:
        using callback_type = Callback;

        // 32.3.5.1 constructors and destructor

        // Constraints: Callback and CB satisfy constructible_from<Callback, CB>.
        //
        // Preconditions: Callback and CB model constructible_from<Callback, CB>.
        //
        // Effects: Initializes callback with std::forward<CB>(cb). If
        //      st.stop_requested() is true, then std::forward<CB>(cb)() is
        //      evaluated in the current thread before the constructor returns.
        //      Otherwise, if st has ownership of a stop state, acquires shared
        //      ownership of that stop state and registers the callback with
        //      that stop state such that std::forward<CB>(cb)() is evaluated by
        //      the first call to request_stop() on an associated stop_source.
        //
        // Remarks: If evaluating std::forward<CB>(cb)() exits via
        //      an exception, then std::terminate is called (14.6.1).
        //
        // Throws: Any exception thrown by the initialization of callback.
        template <typename CB,
            typename Enable = typename std::enable_if<
                std::is_constructible<Callback, CB>::value>::type>
        explicit stop_callback(stop_token const& st, CB&& cb) noexcept(
            std::is_nothrow_constructible<Callback, CB>::value)
          : callback_(std::forward<CB>(cb))
          , state_(st.state_)
        {
            if (state_)
                state_->add_callback(this);
        }

        template <typename CB,
            typename Enable = typename std::enable_if<
                std::is_constructible<Callback, CB>::value>::type>
        explicit stop_callback(stop_token&& st, CB&& cb) noexcept(
            std::is_nothrow_constructible<Callback, CB>::value)
          : callback_(std::forward<CB>(cb))
          , state_(std::move(st.state_))
        {
            if (state_)
                state_->add_callback(this);
        }

        // Effects: Unregisters the callback from the owned stop state, if any.
        //      The destructor does not block waiting for the execution of
        //      another callback registered by an associated stop_callback. If
        //      the callback is concurrently executing on another thread, then
        //      the return from the invocation of callback strongly happens
        //      before (6.9.2.1) callback is destroyed. If callback is executing
        //      on the current thread, then the destructor does not block (3.6)
        //      waiting for the return from the invocation of callback. Releases
        //      ownership of the stop state, if any.
        ~stop_callback()
        {
            if (state_)
                state_->remove_callback(this);
        }

        stop_callback(stop_callback const&) = delete;
        stop_callback(stop_callback&&) = delete;

        stop_callback& operator=(stop_callback const&) = delete;
        stop_callback& operator=(stop_callback&&) = delete;

    private:
        void execute() noexcept override
        {
            callback_();
        }

    private:
        Callback callback_;
        hpx::memory::intrusive_ptr<detail::stop_state> state_;
    };

    ////////////////////////////////////////////////////////////////////////////
    //
    // Mandates: stop_callback is instantiated with an argument for the
    //      template parameter Callback that satisfies both invocable and
    //      destructible.
    //
    // Expects: stop_callback is instantiated with an argument for the
    //      template parameter Callback that models both invocable and
    //      destructible.
    template <typename Callback>
    stop_callback<typename std::decay<Callback>::type> make_stop_callback(
        stop_token const& st, Callback&& cb)
    {
        return stop_callback<typename std::decay<Callback>::type>(
            st, std::forward<Callback>(cb));
    }

    template <typename Callback>
    stop_callback<typename std::decay<Callback>::type> make_stop_callback(
        stop_token&& st, Callback&& cb)
    {
        return stop_callback<typename std::decay<Callback>::type>(
            std::move(st), std::forward<Callback>(cb));
    }

#if defined(HPX_HAVE_CXX17_DEDUCTION_GUIDES)
    template <typename Callback>
    stop_callback(stop_token, Callback) -> stop_callback<Callback>;
#endif

    // 32.3.3.4 Specialized algorithms

    // Effects: Equivalent to: x.swap(y).
    inline void swap(stop_token& lhs, stop_token& rhs) noexcept
    {
        lhs.swap(rhs);
    }

    // 32.3.4.4 Specialized algorithms

    // Effects: Equivalent to: x.swap(y).
    inline void swap(stop_source& lhs, stop_source& rhs) noexcept
    {
        lhs.swap(rhs);
    }
}    // namespace hpx

#endif
