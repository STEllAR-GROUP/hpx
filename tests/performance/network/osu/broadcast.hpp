//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/util/bind_action.hpp>
#include <utility>

#define HPX_DEFINE_COMPONENT_BROADCAST(NAME, TYPE)                              \
    void BOOST_PP_CAT(NAME, _)(TYPE const & value)                              \
    {                                                                           \
        NAME .set(hpx::util::any(value));                                       \
    }                                                                           \
    HPX_DEFINE_COMPONENT_ACTION(broadcast_component, BOOST_PP_CAT(NAME, _));    \
                                                                                \
    ::hpx::lcos::broadcast<BOOST_PP_CAT(BOOST_PP_CAT(NAME, _), _action)> NAME;  \
/**/

namespace hpx { namespace lcos
{
    namespace detail
    {
        void broadcast_impl(std::vector<hpx::id_type> ids,
            hpx::util::function<void(hpx::id_type)> fun, std::size_t fan_out = 2);
        HPX_DEFINE_PLAIN_ACTION(broadcast_impl, broadcast_impl_action);
    }

    template <typename Action>
    struct broadcast
    {
        broadcast()
          : ready_future(ready_promise.get_future())
          , bcast_future(hpx::lcos::make_ready_future())
        {}

        explicit broadcast(hpx::id_type id, std::size_t fan_out = 2)
          : this_id(id)
          , ready_future(ready_promise.get_future())
          , bcast_future(hpx::lcos::make_ready_future())
        {}

        template <typename A0>
        A0 when_src(A0 const & a0)
        {
            return a0;
        }

        template <typename A0>
        A0 when_dst()
        {
            return recv_value.cast<A0>();
        }

        template <typename A0>
        hpx::future<A0> operator()(std::vector<hpx::id_type> const & ids,
            std::size_t src, A0 const & a0)
        {
            hpx::wait_all(bcast_future);
            if(ids[src] == this_id)
            {
                std::vector<hpx::id_type> bcast_ids;
                bcast_ids.reserve(ids.size()-1);
                for (hpx::id_type const& id : ids)
                {
                    if(id == this_id) continue;
                    bcast_ids.push_back(id);
                }

                if(bcast_ids.size() > 0)
                {
                    hpx::id_type locality =
                        hpx::naming::get_locality_from_id(bcast_ids[0]);
                    Action act;

                    bcast_future =
                        hpx::async<detail::broadcast_impl_action>(
                            locality
                          , std::move(bcast_ids)
                          , hpx::util::bind(act, hpx::util::placeholders::_1, a0)
                          , fan_out
                        );
                }
                return hpx::lcos::make_ready_future(a0);
            }
            else
            {
                {
                    mutex_type::scoped_lock lk(mtx);
                    bcast_future = bcast_gate.get_future(1);

                    ready_promise.set_value();
                    ready_promise = hpx::lcos::local::promise<void>();
                }

                return bcast_future.then(hpx::util::bind(&broadcast::when_dst<A0>,
                    this));
            }
        }

        void set(hpx::util::any const & v)
        {
            hpx::wait_all(ready_future);
            {
                mutex_type::scoped_lock lk(mtx);
                recv_value = v;
                bcast_gate.set(0);
                ready_future = ready_promise.get_future();
            }
        }

        typedef hpx::lcos::local::spinlock mutex_type;
        mutex_type mtx;

        hpx::id_type this_id;
        std::size_t fan_out;

        hpx::lcos::local::and_gate bcast_gate;
        hpx::lcos::local::promise<void> ready_promise;
        hpx::future<void> ready_future;
        hpx::util::any recv_value;
        hpx::future<void> bcast_future;
    };

    namespace detail
    {
        void broadcast_impl(std::vector<hpx::id_type> ids,
            hpx::util::function<void(hpx::id_type)> fun, std::size_t fan_out)
        {
            // Call some action for the fan_out first ids here ...
            std::vector<hpx::future<void> > broadcast_futures;
            broadcast_futures.reserve((std::min)(ids.size(), fan_out));
            for(std::size_t i = 0; i < (std::min)(fan_out, ids.size()); ++i)
            {
                broadcast_futures.push_back(
                    hpx::async(fun, ids[i])
                );
            }

            if(ids.size() > fan_out)
            {
                typedef std::vector<hpx::id_type>::const_iterator iterator;
                iterator begin = ids.cbegin() + fan_out;

                for(std::size_t i = 0; i < fan_out; ++i)
                {
                    std::size_t next_dist = (ids.size() - fan_out)/fan_out + 1;
                    iterator end
                        = ((i == fan_out-1) || ((std::distance(ids.cbegin() +
                            fan_out, begin) + next_dist) >= ids.size()))
                        ? ids.cend()
                        : begin + next_dist;

                    std::vector<hpx::id_type> next(begin, end);
                    if(next.size() > 0)
                    {
                        hpx::id_type dst = hpx::naming::get_locality_from_id(next[0]);

                        broadcast_futures.push_back(
                            hpx::async<broadcast_impl_action>(dst, std::move(next),
                                fun, fan_out)
                        );
                        /*
                        hpx::apply<broadcast_impl_action>(dst, std::move(next),
                        fun, fan_out);
                        */
                    }

                    if(end == ids.cend()) break;

                    begin = end;
                }
            }


            if(broadcast_futures.size() > 0)
            {
                hpx::wait_all(broadcast_futures);
            }
        }
    }
}}
