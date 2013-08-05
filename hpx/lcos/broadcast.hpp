//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_BROADCAST_HPP
#define HPX_LCOS_BROADCAST_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/naming/name.hpp>

#include <vector>

namespace hpx { namespace lcos {
    namespace impl
    {
        template <typename Action>
        struct broadcast_result
        {
            typedef
                typename traits::promise_local_result<
                    typename hpx::actions::extract_action<Action>::remote_result_type
                >::type
                action_result;
            typedef
                typename boost::mpl::if_<
                    boost::is_same<void, action_result>
                  , void
                  , std::vector<action_result>
                >::type
                type;
        };

        template <typename Action>
        struct make_broadcast_action;

        template <typename Action>
        typename broadcast_result<Action>::type
        broadcast_impl(Action const & act, std::vector<hpx::id_type> const & ids, boost::mpl::true_)
        {
            if(ids.size() == 0) return;
        }
        
        template <typename Action>
        typename broadcast_result<Action>::type
        broadcast_impl(Action const& act, std::vector<hpx::id_type> const & ids, boost::mpl::false_)
        {
            typedef
                typename broadcast_result<Action>::type
                result_type;

            if(ids.size() == 0) return result_type();

            hpx::id_type this_id = ids[0];
            
            result_type res(ids.size(), -1);
            if(ids.size() == 1)
            {
                res[0] = boost::move(act(ids[0]));
                return boost::move(res);
            }
            
            if(ids.size() == 2)
            {
                hpx::future<
                    typename traits::promise_local_result<
                        typename hpx::actions::extract_action<Action>::remote_result_type
                    >::type
                > f = hpx::async(act, ids[1]);
                res[0] = boost::move(act(ids[0]));
                res[1] = f.move();
                return boost::move(res);
            }

            std::vector<hpx::future<result_type> > broadcast_futures;
            broadcast_futures.reserve(2);
            std::size_t half = (ids.size() / 2) + 1;
            if(half == 1) half = 2;
            std::vector<hpx::id_type> ids_first(ids.begin() + 1, ids.begin() + half);
            std::vector<hpx::id_type> ids_second(ids.begin() + half, ids.end());

            typedef
                typename impl::make_broadcast_action<Action>::type
                broadcast_impl_action;

            hpx::id_type id = hpx::naming::get_locality_from_id(ids_first[0]);
            broadcast_futures.push_back(
                hpx::async<broadcast_impl_action>(
                    id,
                    act,
                    boost::move(ids_first),
                    boost::integral_constant<bool, false>()
                )
            );

            if(ids_second.size() > 0)
            {
                hpx::id_type id = hpx::naming::get_locality_from_id(ids_second[0]);
                broadcast_futures.push_back(
                    hpx::async<broadcast_impl_action>(
                        id,
                        act,
                        boost::move(ids_second),
                        boost::integral_constant<bool, false>()
                    )
                );
            }

            res[0] = boost::move(act(this_id));

            while(!broadcast_futures.empty())
            {
                HPX_STD_TUPLE<int, hpx::future<result_type> >
                    f_res = hpx::wait_any(broadcast_futures);
                int part = HPX_STD_GET(0, f_res);
                result_type tmp(boost::move(HPX_STD_GET(1, f_res).move()));
                if(part == 0)
                {
                    std::copy(tmp.begin(), tmp.end(), res.begin() + 1);
                }
                else
                {
                    std::copy(tmp.begin(), tmp.end(), res.begin() + half);
                }
                broadcast_futures.erase(broadcast_futures.begin() + part);
            }

            return boost::move(res);
        }


        template <typename Action, typename IsVoid>
        typename broadcast_result<Action>::type
        broadcast(Action const & act, std::vector<hpx::id_type> const & ids, IsVoid)
        {
            return broadcast_impl(act, ids, IsVoid());
        }

        template <typename Action>
        struct make_broadcast_action
        {
            typedef
                typename traits::promise_local_result<
                    typename hpx::actions::extract_action<Action>::remote_result_type
                >::type
                action_result;

            typedef
                typename HPX_MAKE_ACTION_TPL(
                    (broadcast<Action, typename boost::is_same<void, action_result>::type>)
                )::type
                type;
        };
    }

    template <typename Action>
    hpx::future<
        typename impl::broadcast_result<Action>::type
    > broadcast(std::vector<hpx::id_type> const & ids)
    {
        hpx::id_type dest = hpx::naming::get_locality_from_id(ids[0]);

        typedef
            typename impl::make_broadcast_action<Action>::type
            broadcast_impl_action;
        typedef
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::remote_result_type
            >::type
            action_result;
        return
            hpx::async<broadcast_impl_action>(
                dest,
                Action(),
                boost::move(ids),
                typename boost::is_same<void, action_result>::type()
            );
    }

}}

#define HPX_REGISTER_BROADCAST_ACTION_DECLARATION(Action)                       \
    HPX_REGISTER_ACTION_DECLARATION(                                            \
        ::hpx::lcos::impl::make_broadcast_action<Action>::type                  \
      , BOOST_PP_CAT(broadcast_, Action)                                        \
    )                                                                           \
/**/

#define HPX_REGISTER_BROADCAST_ACTION(Action)                                   \
    HPX_REGISTER_PLAIN_ACTION(                                                  \
        ::hpx::lcos::impl::make_broadcast_action<Action>::type                  \
      , BOOST_PP_CAT(broadcast_, Action)                                        \
    )                                                                           \
/**/


#endif
