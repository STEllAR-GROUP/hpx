//  Copyright (c) 2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERIALIZATION_FUTURE_AWAIT_CONTAINER_HPP)
#define HPX_SERIALIZATION_FUTURE_AWAIT_CONTAINER_HPP

// This 'container' is used to gather futures that need to become
// ready before the actual serialization process can be started

#include <hpx/lcos/future.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/util/unwrapped.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/thread/locks.hpp>

#include <vector>

namespace hpx
{
    bool is_starting();
}

namespace hpx { namespace serialization { namespace detail
{
    template <typename Container>
    struct access_data;

    class future_await_container
        : public boost::enable_shared_from_this<future_await_container>
    {
        typedef hpx::lcos::local::spinlock mutex_type;
        typedef std::list<naming::gid_type> new_gids_type;
        typedef std::map<naming::gid_type, new_gids_type> new_gids_map;
    public:
        future_await_container()
          : done_(false)
          , num_futures_(0)
          , triggered_futures_(0)
        {}

        std::size_t size() const { return 0; }
        void resize(std::size_t size) { }

        void trigger()
        {
            // hpx::lcos::local::promise<void>::set_value() might need to acquire
            // a lock, as such, we check the our triggering condition inside a
            // critical section and trigger the promise outside of it.
            bool set_value = false;
            {
                boost::lock_guard<mutex_type> l(mtx_);
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
                boost::lock_guard<mutex_type> l(mtx_);
                ++num_futures_;
            }
            future_data.set_on_completed(
                [this]()
                {
                    trigger();
                }
            );
        }

        void add_gid(
            naming::gid_type const & gid,
            naming::gid_type const & splitted_gid)
        {
            boost::lock_guard<mutex_type> l(mtx_);
            new_gids_[gid].push_back(splitted_gid);
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
            if(num_futures_ == 0)
            {
                promise_.set_value();
            }
            return num_futures_ > 0;
        }

        template <typename F>
        void operator()(F f)
        {
            {
                boost::lock_guard<mutex_type> l(mtx_);
                done_ = true;
                if(num_futures_ == triggered_futures_)
                {
                    promise_.set_value();
                }
            }

            hpx::dataflow(//hpx::launch::sync,
                util::unwrapped(std::move(f))
              , promise_.get_future());
        }

        new_gids_map new_gids_;

    private:
        mutex_type mtx_;
        bool done_;
        std::size_t num_futures_;
        std::size_t triggered_futures_;

        hpx::lcos::local::promise<void> promise_;
    };

    template <>
    struct access_data<future_await_container>
    {
        static bool is_saving() { return false; }
        static bool is_future_awaiting() { return true; }

        static void await_future(
            future_await_container& cont
          , hpx::lcos::detail::future_data_refcnt_base & future_data)
        {
            cont.await_future(future_data);
        }

        static void add_gid(future_await_container& cont,
                naming::gid_type const & gid,
                naming::gid_type const & splitted_gid)
        {
            cont.add_gid(gid, splitted_gid);
        }

        static void
        write(future_await_container& cont, std::size_t count,
            std::size_t current, void const* address)
        {
        }

        static bool
        flush(binary_filter* filter, future_await_container& cont,
            std::size_t current, std::size_t size, std::size_t written)
        {
            return true;
        }
    };
}}}

#endif
