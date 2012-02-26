//  Copyright (c) 2011-2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_LCOS_DATAFLOW_SERVER_DETAIL_APPLY_HELPER_HPP
#define HPX_LCOS_DATAFLOW_SERVER_DETAIL_APPLY_HELPER_HPP

#include <boost/fusion/include/at_c.hpp>

namespace hpx { namespace lcos { namespace server { namespace detail
{
    template <int N, typename Action>
    struct apply_helper;

    template <typename Action>
    struct apply_helper<0, Action>
    {
        template <typename Vector>
        void operator()(
            naming::id_type const & cont
          , naming::id_type const & id
          , Vector const &
        ) const
        {
            hpx::applier::apply_c<Action>(cont, id);
        }
    };
#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            1                                                                   \
          , HPX_ACTION_ARGUMENT_LIMIT                                           \
          , <hpx/components/dataflow/server/detail/apply_helper.hpp>            \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

}}}}


#endif

#else // BOOST_PP_IS_ITERATING
#define N BOOST_PP_ITERATION()
#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
    boost::move(boost::fusion::at_c<N>(args))                                   \
    /**/

    template <typename Action>
    struct apply_helper<N, Action>
    {
        template <typename Vector>
        void operator()(
            naming::id_type const & cont
          , naming::id_type const & id
          , Vector const & args
        ) const
        {
            LLCO_(info)
                << "dataflow apply action "
                << hpx::actions::detail::get_action_name<Action>();
            hpx::applier::apply_c<Action>(
                cont
              , id
              , BOOST_PP_ENUM(N, HPX_LCOS_DATAFLOW_M0, _)
            );
        }
    };
    /**/

#undef HPX_LCOS_DATAFLOW_M0
#undef N

#endif
