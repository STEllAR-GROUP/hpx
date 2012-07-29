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
            hpx::apply_c<Action>(cont, id);
        }
    };

#if !defined(HPX_DONT_USE_PREPROCESSED_FILES)
#  include <hpx/components/dataflow/server/detail/preprocessed/apply_helper.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/apply_helper_" HPX_LIMIT_STR ".hpp")
#endif

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

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_DONT_USE_PREPROCESSED_FILES)

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
          , Vector & args
        ) const
        {
            LLCO_(info)
                << "dataflow apply action "
                << hpx::actions::detail::get_action_name<Action>();
            hpx::apply_c<Action>(
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
