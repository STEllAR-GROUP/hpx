//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_CHANNEL_AUG_19_2015_0513PM)
#define HPX_LCOS_LOCAL_CHANNEL_AUG_19_2015_0513PM

#include <hpx/config.hpp>
#include <hpx/lcos/local/gate.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/move.hpp>

#include <boost/thread/locks.hpp>

namespace hpx { namespace lcos { namespace local
{
    template <typename T>
    class channel
    {
    private:
        typedef lcos::local::spinlock mutex_type;

        T on_get_ready(future<void> && f)
        {
            f.get();       // propagate any exceptions

            boost::unique_lock<mutex_type> l(mtx_);
            return std::move(data_);
        }

    public:
        channel()
          : generation_(0)
        {}

        hpx::future<T> get_future()
        {
            boost::unique_lock<mutex_type> l(mtx_);

            using util::placeholders::_1;
            future<void> f = gate_.get_future();

            if (!f.is_ready())
            {
                l.unlock();
                return f.then(util::bind(&channel::on_get_ready, this, _1));
            }

            return make_ready_future(std::move(data_));
        }

        void set_value(T const& t)
        {
            boost::unique_lock<mutex_type> l(mtx_);
            gate_.synchronize(++generation_, l);
            data_ = t;
            gate_.set(l);
        }

        void set_value(T && t)
        {
            boost::unique_lock<mutex_type> l(mtx_);
            gate_.synchronize(++generation_, l);
            data_ = std::move(t);
            gate_.set(l);
        }

    private:
        mutex_type mtx_;
        std::size_t generation_;
        lcos::local::gate gate_;
        T data_;
    };
}}}

#endif

