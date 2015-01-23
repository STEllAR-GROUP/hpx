//  Copyright (c) 2011-2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_SERVER_DETAIL_APPLY_HELPER_HPP
#define HPX_LCOS_DATAFLOW_SERVER_DETAIL_APPLY_HELPER_HPP

#include <hpx/util/detail/pack.hpp>

#include <boost/fusion/include/at_c.hpp>

namespace hpx { namespace lcos { namespace server { namespace detail
{
    template <
        std::size_t N, typename Action
      , typename Is = typename util::detail::make_index_pack<N>::type
    >
    struct apply_helper;

    template <std::size_t N, typename Action, std::size_t ...Is>
    struct apply_helper<N, Action, util::detail::pack_c<std::size_t, Is...> >
    {
        template <typename Vector>
        void operator()(
            naming::id_type const & cont
          , naming::id_type const & id
          , Vector & args
        ) const
        {
            LLCO_(info)
                << "dataflow apply action "
                << hpx::actions::detail::get_action_name<Action>();
            hpx::apply_c<Action>(
                cont
              , id
              , std::move(boost::fusion::at_c<Is>(args))...
            );
        }
    };
}}}}

#endif
