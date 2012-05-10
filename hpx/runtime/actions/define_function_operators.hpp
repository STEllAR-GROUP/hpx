//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_ACTIONS_DEFINE_FUNCTION_OPERATORS_MAY_10_2012_0242PM)
#define HPX_RUNTIME_ACTIONS_DEFINE_FUNCTION_OPERATORS_MAY_10_2012_0242PM

#include <hpx/lcos/async.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/comma_if.hpp>

#define HPX_FWD_ARGS(z, n, _)                                                 \
        BOOST_PP_COMMA_IF(n)                                                  \
            BOOST_FWD_REF(BOOST_PP_CAT(Arg, n)) BOOST_PP_CAT(arg, n)          \
    /**/
#define HPX_FORWARD_ARGS(z, n, _)                                             \
        BOOST_PP_COMMA_IF(n)                                                  \
            boost::forward<BOOST_PP_CAT(Arg, n)>(BOOST_PP_CAT(arg, n))        \
    /**/

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (0, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/define_function_operators.hpp"))                     \
    /**/

#include BOOST_PP_ITERATE()

#undef HPX_FWD_ARGS
#undef HPX_FORWARD_ARGS

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived,
        threads::thread_priority Priority>
    template <typename IdType
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    typename boost::enable_if<
        boost::mpl::and_<
            boost::mpl::bool_<
                boost::fusion::result_of::size<arguments_type>::value == N>,
            boost::is_same<IdType, naming::id_type> >,
        Result
    >::type
    action<Component, Action, Result, Arguments, Derived, Priority>::
        operator()(IdType const& id
          BOOST_PP_COMMA_IF(N) BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _)) const
    {
        return hpx::async(*this, id
          BOOST_PP_COMMA_IF(N) BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _))
    }
}}

#undef N

#endif // !BOOST_PP_IS_ITERATING
