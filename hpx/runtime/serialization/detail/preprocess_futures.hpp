//  Copyright (c) 2015-2019 Hartmut Kaiser
//  Copyright (c) 2015-2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERIALIZATION_DETAIL_PREPROCESS_FUTURES_HPP)
#define HPX_SERIALIZATION_DETAIL_PREPROCESS_FUTURES_HPP

#include <hpx/assertion.hpp>
#include <hpx/datastructures.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/promise.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos_fwd.hpp>
#include <hpx/serialization/extra_archive_data.hpp>
#include <hpx/serialization/extra_output_data.hpp>

#include <cstddef>
#include <mutex>
#include <type_traits>
#include <utility>

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization { namespace detail {

    // This class allows to register futures during serialization preprocessing
    // to ensure each future is ready before serializing it.
    class preprocess_futures
    {
        using mutex_type = hpx::lcos::local::spinlock;

    public:
        preprocess_futures()
          : mtx_()
          , done_(false)
          , num_futures_(0)
          , triggered_futures_(0)
        {
        }

        preprocess_futures(preprocess_futures&& rhs) noexcept
          : mtx_()
          , done_(rhs.done_)
          , num_futures_(rhs.num_futures_)
          , triggered_futures_(rhs.triggered_futures_)
          , promise_(std::move(rhs.promise_))
        {
            rhs.done_ = true;
            rhs.num_futures_ = 0;
            rhs.triggered_futures_ = 0;
        }

        preprocess_futures& operator=(preprocess_futures&& rhs) noexcept
        {
            done_ = rhs.done_;
            num_futures_ = rhs.num_futures_;
            triggered_futures_ = rhs.triggered_futures_;
            promise_ = std::move(rhs.promise_);

            rhs.done_ = true;
            rhs.num_futures_ = 0;
            rhs.triggered_futures_ = 0;

            return *this;
        }

        void trigger()
        {
            // hpx::lcos::local::promise<>::set_value() might need to acquire
            // a lock, as such, we check the our triggering condition inside a
            // critical section and trigger the promise outside of it.
            bool set_value = false;

            {
                std::lock_guard<mutex_type> l(mtx_);
                ++triggered_futures_;
                set_value = (done_ && num_futures_ == triggered_futures_);
            }

            if (set_value)
            {
                promise_.set_value();
            }
        }

        void await_future(
            hpx::lcos::detail::future_data_refcnt_base& future_data)
        {
            {
                std::lock_guard<mutex_type> l(mtx_);
                ++num_futures_;
            }

            future_data.set_on_completed([this]() { this->trigger(); });
        }

        void reset()
        {
            done_ = false;
            num_futures_ = 0;
            triggered_futures_ = 0;
            promise_ = hpx::lcos::local::promise<void>();
        }

        bool has_futures()
        {
            if (num_futures_ == 0)
            {
                promise_.set_value();
            }
            return num_futures_ > 0;
        }

        template <typename F>
        void operator()(F f)
        {
            bool set_promise = false;

            {
                std::lock_guard<mutex_type> l(mtx_);
                done_ = true;
                if (num_futures_ == triggered_futures_)
                    set_promise = true;
            }

            hpx::future<void> fut = promise_.get_future();

            if (set_promise)
                promise_.set_value();

            // we don't call f directly to avoid possible stack overflow.
            auto shared_state_ =
                hpx::traits::future_access<hpx::future<void>>::get_shared_state(
                    fut);
            shared_state_->set_on_completed(std::move(f));
        }

        // We add this solely for the purpose of making moveonly_any compile.
        // Comparing instances of this type does not make any sense,
        // conceptually.
        friend bool operator==(
            preprocess_futures const&, preprocess_futures const&)
        {
            HPX_ASSERT(false);    // shouldn't ever be called
            return false;
        }

    private:
        mutex_type mtx_;
        bool done_;
        std::size_t num_futures_;
        std::size_t triggered_futures_;
        hpx::lcos::local::promise<void> promise_;
    };
}}}    // namespace hpx::serialization::detail

namespace hpx { namespace serialization {

    // serialization support for gid_type (handles credit-splitting)
    constexpr std::size_t extra_output_handle_futures = 2;

    template <>
    inline util::moveonly_any_nonser
    init_extra_output_data_item<extra_output_handle_futures>()
    {
        return util::moveonly_any_nonser{detail::preprocess_futures{}};
    }

    template <>
    inline void reset_extra_output_data_item<extra_output_handle_futures>(
        extra_archive_data_type& data)
    {
        util::any_cast<detail::preprocess_futures&>(
            data[extra_output_handle_futures]).reset();
    }
}}    // namespace hpx::serialization

#endif
