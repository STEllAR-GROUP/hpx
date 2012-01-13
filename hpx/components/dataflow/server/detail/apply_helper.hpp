//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_SERVER_DETAIL_APPLY_HELPER_HPP
#define HPX_LCOS_DATAFLOW_SERVER_DETAIL_APPLY_HELPER_HPP

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

#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
                boost::fusion::at_c<N>(args)                                    \
    /**/
#define HPX_LCOS_DATAFLOW_M1(Z, N, D)                                           \
    template <typename Action>                                                  \
    struct apply_helper<N, Action>                                              \
    {                                                                           \
        template <typename Vector>                                              \
        void operator()(                                                        \
            naming::id_type const & cont                                        \
          , naming::id_type const & id                                          \
          , Vector const & args                                                 \
        ) const                                                                 \
        {                                                                       \
            LLCO_(info)                                                         \
                << "dataflow apply action "                                     \
                << hpx::actions::detail::get_action_name<Action>();             \
            hpx::applier::apply_c<Action>(                                      \
                cont                                                            \
              , id                                                              \
              , BOOST_PP_ENUM(N, HPX_LCOS_DATAFLOW_M0, _)                       \
            );                                                                  \
        }                                                                       \
    };                                                                          \
    /**/

BOOST_PP_REPEAT_FROM_TO(1, HPX_ACTION_ARGUMENT_LIMIT, HPX_LCOS_DATAFLOW_M1, _)

#undef HPX_LCOS_DATAFLOW_M0
#undef HPX_LCOS_DATAFLOW_M1

}}}}


#endif
