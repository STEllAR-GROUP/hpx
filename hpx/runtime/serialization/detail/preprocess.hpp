//  Copyright (c) 2015 Hartmut Kaiser
//  Copyright (c) 2015-2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERIALIZATION_DETAIL_PREPROCESS_HPP)
#define HPX_SERIALIZATION_DETAIL_PREPROCESS_HPP

// This 'container' is used to gather the required archive size for a given
// type before it is serialized. In addition, it allows to register futures
// to ensure each future is ready before serializing it.
#include <hpx/lcos_fwd.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/serialization/binary_filter.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/promise.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/traits/serialization_access_data.hpp>

#include <cstddef>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>

namespace hpx { namespace serialization { namespace detail
{
    class preprocess
    {
        typedef hpx::lcos::local::spinlock mutex_type;

    public:
        typedef std::map<const naming::gid_type*, naming::gid_type> split_gids_map;
        preprocess()
          : size_(0)
          , done_(false)
          , num_futures_(0)
          , triggered_futures_(0)
        {}

        std::size_t size() const { return size_; }
        void resize(std::size_t size) { size_ = size; }

        void trigger()
        {
            // hpx::lcos::local::promise<void>::set_value() might need to acquire
            // a lock, as such, we check the our triggering condition inside a
            // critical section and trigger the promise outside of it.
            bool set_value = false;
            {
                std::lock_guard<mutex_type> l(mtx_);
                ++triggered_futures_;
                set_value = (done_ && num_futures_ == triggered_futures_);
            }
            if(set_value)
            {
                promise_.set_value();
            }
        }

        void await_future(hpx::lcos::detail::future_data_refcnt_base & future_data)
        {
            {
                std::lock_guard<mutex_type> l(mtx_);
                ++num_futures_;
            }
            future_data.set_on_completed(
                [this]()
                {
                    this->trigger();
                }
            );
        }

        void add_gid(
            naming::gid_type const & gid,
            naming::gid_type const & split_gid)
        {
            std::lock_guard<mutex_type> l(mtx_);
            HPX_ASSERT(split_gids_[&gid] == naming::invalid_gid);
            split_gids_[&gid] = split_gid;
        }

        bool has_gid(naming::gid_type const & gid)
        {
            std::lock_guard<mutex_type> l(mtx_);
            return split_gids_.find(&gid) != split_gids_.end();
        }

        void reset()
        {
            size_ = 0;
            done_ = false;
            num_futures_ = 0;
            triggered_futures_ = 0;
            promise_ = hpx::lcos::local::promise<void>();
        }

        bool has_futures()
        {
            if(num_futures_ == 0)
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

            if (set_promise)
                promise_.set_value();

            hpx::future<void> fut = promise_.get_future();
            // we don't call f directly to avoid possible stack overflow.
            auto shared_state_ = hpx::traits::future_access<hpx::future<void> >::
                get_shared_state(fut);
            shared_state_->set_on_completed(std::move(f));
        }

        split_gids_map split_gids_;

    private:
        std::size_t size_;
        mutex_type mtx_;
        bool done_;
        std::size_t num_futures_;
        std::size_t triggered_futures_;

        hpx::lcos::local::promise<void> promise_;
    };
}}}

namespace hpx { namespace traits
{
    template <>
    struct serialization_access_data<serialization::detail::preprocess>
      : default_serialization_access_data<serialization::detail::preprocess>
    {
        typedef std::true_type preprocessing_only;

        HPX_CONSTEXPR static bool is_preprocessing() { return true; }

        static std::size_t size(serialization::detail::preprocess const& cont)
        {
            return cont.size();
        }

        static void resize(serialization::detail::preprocess& cont,
            std::size_t count)
        {
            return cont.resize(cont.size() + count);
        }

        // functions related to output operations
        static void await_future(
            serialization::detail::preprocess& cont
          , hpx::lcos::detail::future_data_refcnt_base & future_data)
        {
            cont.await_future(future_data);
        }

        static void add_gid(serialization::detail::preprocess& cont,
                naming::gid_type const & gid,
                naming::gid_type const & split_gid)
        {
            cont.add_gid(gid, split_gid);
        }

        static bool has_gid(serialization::detail::preprocess& cont,
            naming::gid_type const& gid)
        {
            return cont.has_gid(gid);
        }

        static void reset(serialization::detail::preprocess& cont)
        {
            cont.reset();
        }
    };
}}

#endif
