//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/datastructures/detail/dynamic_bitset.hpp>
#include <hpx/datastructures/detail/intrusive_list.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/lcos_local/conditional_trigger.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/synchronization/no_mutex.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/assert_owns_lock.hpp>
#include <hpx/thread_support/unlock_guard.hpp>

#include <cstddef>
#include <mutex>
#include <utility>

namespace hpx::lcos::local {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex = hpx::spinlock>
    struct base_and_gate
    {
    protected:
        using mutex_type = Mutex;

    private:
        struct condition_list_entry : conditional_trigger
        {
            condition_list_entry() = default;

            condition_list_entry* prev = nullptr;
            condition_list_entry* next = nullptr;
        };

        using condition_list_type =
            hpx::detail::intrusive_list<condition_list_entry>;

    public:
        /// This constructor initializes the base_and_gate object from the the
        /// number of participants to synchronize the control flow with.
        explicit base_and_gate(std::size_t count = 0)
          : received_segments_(count)
          , generation_(0)
        {
        }

        base_and_gate(base_and_gate&& rhs) noexcept
          : mtx_()
          , received_segments_(HPX_MOVE(rhs.received_segments_))
          , promise_(HPX_MOVE(rhs.promise_))
          , generation_(rhs.generation_)
          , conditions_(HPX_MOVE(rhs.conditions_))
        {
            rhs.generation_ = std::size_t(-1);
        }

        base_and_gate& operator=(base_and_gate&& rhs) noexcept
        {
            if (this != &rhs)
            {
                std::lock_guard<mutex_type> l(rhs.mtx_);
                mtx_ = mutex_type();
                received_segments_ = HPX_MOVE(rhs.received_segments_);
                promise_ = HPX_MOVE(rhs.promise_);
                generation_ = rhs.generation_;
                rhs.generation_ = std::size_t(-1);
                conditions_ = HPX_MOVE(rhs.conditions_);
            }
            return *this;
        }

    protected:
        bool trigger_conditions(error_code& ec = throws)
        {
            bool triggered = false;
            if (!conditions_.empty())
            {
                error_code rc(throwmode::lightweight);
                condition_list_entry* next = nullptr;
                for (auto* c = conditions_.front(); c != nullptr; c = next)
                {
                    // item me be deleted during processing
                    next = c->next;
                    triggered |= c->set(rc);

                    if (rc && (&ec != &throws))
                    {
                        ec = rc;
                    }
                }
            }
            else
            {
                if (&ec != &throws)
                    ec = make_success_code();
            }
            return triggered;
        }

    protected:
        /// Get a future allowing to wait for the gate to fire
        template <typename OuterLock>
        hpx::future<void> get_future(OuterLock& outer_lock,
            std::size_t count = std::size_t(-1),
            std::size_t* generation_value = nullptr,
            error_code& ec = hpx::throws)
        {
            std::unique_lock<mutex_type> l(mtx_);

            // by default we use as many segments as specified during
            // construction
            if (count == std::size_t(-1))
            {
                count = received_segments_.size();
            }
            HPX_ASSERT(count != 0);

            init_locked(outer_lock, l, count, ec);
            if (!ec)
            {
                HPX_ASSERT(generation_ != std::size_t(-1));
                ++generation_;

                // re-check/trigger condition, if needed
                trigger_conditions(ec);
                if (!ec)
                {
                    if (generation_value)
                        *generation_value = generation_;
                    return promise_.get_future(ec);
                }
            }
            return hpx::future<void>();
        }

    public:
        hpx::future<void> get_future(std::size_t count = std::size_t(-1),
            std::size_t* generation_value = nullptr,
            error_code& ec = hpx::throws)
        {
            hpx::no_mutex mtx;
            std::unique_lock<hpx::no_mutex> lk(mtx);
            return get_future(lk, count, generation_value, ec);
        }

    protected:
        /// Get a shared future allowing to wait for the gate to fire
        template <typename OuterLock>
        hpx::shared_future<void> get_shared_future(OuterLock& outer_lock,
            std::size_t count = std::size_t(-1),
            std::size_t* generation_value = nullptr,
            error_code& ec = hpx::throws)
        {
            std::unique_lock<mutex_type> l(mtx_);

            // by default we use as many segments as specified during
            // construction
            if (count == std::size_t(-1))
            {
                count = received_segments_.size();
            }
            HPX_ASSERT(count != 0);
            HPX_ASSERT(generation_ != std::size_t(-1));

            if (generation_ == 0)
            {
                init_locked(outer_lock, l, count, ec);
                generation_ = 1;
            }

            if (!ec)
            {
                // re-check/trigger condition, if needed
                trigger_conditions(ec);
                if (!ec)
                {
                    if (generation_value)
                    {
                        *generation_value = generation_;
                    }
                    return promise_.get_shared_future(ec);
                }
            }
            return hpx::future<void>().share();
        }

    public:
        hpx::shared_future<void> get_shared_future(
            std::size_t count = std::size_t(-1),
            std::size_t* generation_value = nullptr,
            error_code& ec = hpx::throws)
        {
            hpx::no_mutex mtx;
            std::unique_lock<hpx::no_mutex> lk(mtx);
            return get_shared_future(lk, count, generation_value, ec);
        }

    protected:
        /// Set the data which has to go into the segment \a which.
        template <typename OuterLock>
        bool set(
            std::size_t which, OuterLock outer_lock, error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(outer_lock);

            std::unique_lock<mutex_type> l(mtx_);
            if (which >= received_segments_.size())
            {
                // out of bounds index
                l.unlock();
                outer_lock.unlock();
                HPX_THROWS_IF(ec, hpx::error::bad_parameter,
                    "base_and_gate<>::set",
                    "index is out of range for this base_and_gate");
                return false;
            }
            if (received_segments_.test(which))
            {
                // segment already filled, logic error
                l.unlock();
                outer_lock.unlock();
                HPX_THROWS_IF(ec, hpx::error::bad_parameter,
                    "base_and_gate<>::set",
                    "input with the given index has already been triggered");
                return false;
            }

            if (&ec != &throws)
                ec = make_success_code();

            // set the corresponding bit
            received_segments_.set(which);

            if (received_segments_.count() == received_segments_.size())
            {
                // we have received the last missing segment
                hpx::promise<void> p;
                std::swap(p, promise_);
                received_segments_.reset();    // reset data store

                // Unlock the lock to avoid locking problems when triggering the
                // promise
                l.unlock();
                outer_lock.unlock();
                p.set_value();    // fire event

                return true;
            }

            outer_lock.unlock();
            return false;
        }

    public:
        bool set(std::size_t which, error_code& ec = throws)
        {
            hpx::no_mutex mtx;
            std::unique_lock<hpx::no_mutex> lk(mtx);
            return set(which, HPX_MOVE(lk), ec);
        }

    protected:
        bool test_condition(std::size_t generation_value) noexcept
        {
            return !(generation_value > generation_);
        }

        struct manage_condition
        {
            manage_condition(
                base_and_gate& gate, condition_list_entry& cond) noexcept
              : this_(gate)
              , e_(cond)
            {
                this_.conditions_.push_back(cond);
            }

            ~manage_condition()
            {
                this_.conditions_.erase(&e_);
            }

            template <typename Condition>
            hpx::future<void> get_future(
                Condition&& func, error_code& ec = hpx::throws)
            {
                return e_.get_future(HPX_FORWARD(Condition, func), ec);
            }

            base_and_gate& this_;
            condition_list_entry& e_;
        };

    public:
        /// Wait for the generational counter to reach the requested stage.
        void synchronize(std::size_t generation_value,
            char const* function_name = "base_and_gate<>::synchronize",
            error_code& ec = throws)
        {
            std::unique_lock<mutex_type> l(mtx_);
            synchronize(generation_value, l, function_name, ec);
        }

    protected:
        template <typename Lock>
        void synchronize(std::size_t generation_value, Lock& l,
            char const* function_name = "base_and_gate<>::synchronize",
            error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            if (generation_value < generation_)
            {
                l.unlock();
                HPX_THROWS_IF(ec, hpx::error::invalid_status, function_name,
                    "sequencing error, generational counter too small");
                return;
            }

            // make sure this set operation has not arrived ahead of time
            if (!test_condition(generation_value))
            {
                condition_list_entry c;
                manage_condition cond(*this, c);

                hpx::future<void> f = cond.get_future(hpx::bind_front(
                    &base_and_gate::test_condition, this, generation_value));

                {
                    hpx::unlock_guard<Lock> ul(l);
                    f.get();
                }    // make sure lock gets re-acquired
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

    public:
        template <typename Lock>
        std::size_t next_generation(Lock& l, std::size_t new_generation)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            HPX_ASSERT(generation_ != std::size_t(-1));

            if (new_generation != std::size_t(-1))
            {
                if (new_generation < generation_)
                {
                    l.unlock();
                    HPX_THROW_EXCEPTION(hpx::error::invalid_status,
                        "and_gate::next_generation",
                        "sequencing error, new generational counter value too "
                        "small");
                    return std::size_t(-1);
                }
                generation_ = new_generation;
            }

            std::size_t retval = ++generation_;

            trigger_conditions();    // re-check/trigger condition, if needed

            return retval;
        }

        std::size_t next_generation(
            std::size_t new_generation = std::size_t(-1))
        {
            std::unique_lock<mutex_type> l(mtx_);
            return next_generation(l, new_generation);
        }

        template <typename Lock>
        std::size_t generation(Lock& l) const
        {
            HPX_ASSERT_OWNS_LOCK(l);
            return generation_;
        }

        std::size_t generation() const
        {
            std::lock_guard<mutex_type> l(mtx_);
            return generation(l);
        }

    protected:
        template <typename OuterLock, typename Lock>
        void init_locked(OuterLock& outer_lock, Lock& l, std::size_t count,
            error_code& ec = throws)
        {
            if (0 != received_segments_.count())
            {
                // reset happens while part of the slots are filled
                l.unlock();
                outer_lock.unlock();
                HPX_THROWS_IF(ec, hpx::error::bad_parameter,
                    "base_and_gate<>::init",
                    "initializing this base_and_gate while slots are filled");
                return;
            }

            if (received_segments_.size() != count)
            {
                received_segments_.resize(count);    // resize the bitmap
            }
            received_segments_.reset();    // reset all existing bits

            if (&ec != &throws)
                ec = make_success_code();
        }

    private:
        mutable mutex_type mtx_;
        hpx::detail::dynamic_bitset<> received_segments_;
        hpx::promise<void> promise_;
        std::size_t generation_;
        condition_list_type conditions_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Note: This type is not thread-safe. It has to be protected from
    //       concurrent access by different threads by the code using instances
    //       of this type.
    struct and_gate : public base_and_gate<hpx::no_mutex>
    {
    private:
        using base_type = base_and_gate<hpx::no_mutex>;

    public:
        explicit and_gate(std::size_t count = 0)
          : base_type(count)
        {
        }

        and_gate(and_gate&& rhs) noexcept
          : base_type(HPX_MOVE(static_cast<base_type&>(rhs)))
        {
        }

        and_gate& operator=(and_gate&& rhs) noexcept
        {
            if (this != &rhs)
                static_cast<base_type&>(*this) = HPX_MOVE(rhs);
            return *this;
        }

        template <typename Lock>
        hpx::future<void> get_future(Lock& l,
            std::size_t count = std::size_t(-1),
            std::size_t* generation_value = nullptr,
            error_code& ec = hpx::throws)
        {
            return this->base_type::get_future(l, count, generation_value, ec);
        }

        template <typename Lock>
        hpx::shared_future<void> get_shared_future(Lock& l,
            std::size_t count = std::size_t(-1),
            std::size_t* generation_value = nullptr,
            error_code& ec = hpx::throws)
        {
            return this->base_type::get_shared_future(
                l, count, generation_value, ec);
        }

        template <typename Lock>
        bool set(std::size_t which, Lock l, error_code& ec = throws)
        {
            return this->base_type::set(which, HPX_MOVE(l), ec);
        }

        template <typename Lock>
        void synchronize(std::size_t generation_value, Lock& l,
            char const* function_name = "and_gate::synchronize",
            error_code& ec = throws)
        {
            this->base_type::synchronize(
                generation_value, l, function_name, ec);
        }
    };
}    // namespace hpx::lcos::local
