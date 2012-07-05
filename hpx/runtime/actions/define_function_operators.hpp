//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_ACTIONS_FUNCTION_OPERATORS_MAY_09_2012_0420PM)
#define HPX_RUNTIME_ACTIONS_FUNCTION_OPERATORS_MAY_09_2012_0420PM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/enum_params.hpp>

#define HPX_MOVE_ARGS(z, n, _)                                                \
        BOOST_PP_COMMA_IF(n) boost::move(BOOST_PP_CAT(arg, n))                \
    /**/

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/define_function_operators.hpp"))                     \
    /**/

#include BOOST_PP_ITERATE()

#undef HPX_MOVE_ARGS

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

    ///////////////////////////////////////////////////////////////////////////
    template <typename IdType, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    typename boost::enable_if<
        boost::mpl::and_<
            boost::mpl::bool_<
                boost::fusion::result_of::size<arguments_type>::value == N>,
            boost::is_same<IdType, naming::id_type> >,
        typename traits::promise_local_result<Result>::type
    >::type
    operator()(IdType const& id, BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, arg),
        error_code& ec = throws) const
    {
        return hpx::async(*this, id
          BOOST_PP_COMMA_IF(N) BOOST_PP_REPEAT(N, HPX_MOVE_ARGS, _)).get(ec);
    }

#undef N

#endif // !BOOST_PP_IS_ITERATING
