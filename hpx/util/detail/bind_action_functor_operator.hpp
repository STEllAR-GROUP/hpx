
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define NNN BOOST_PP_FRAME_ITERATION(2)

    template <BOOST_PP_ENUM_PARAMS(NNN, typename A)>
    result_type operator()(HPX_ENUM_FWD_ARGS(NNN, A, a))
    {
        typedef
            BOOST_PP_CAT(hpx::util::tuple, NNN)<
                BOOST_PP_ENUM(NNN, HPX_UTIL_BIND_REFERENCE, A)
            >
            env_type;
        env_type env(HPX_ENUM_FORWARD_ARGS(NNN, A, a));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          BOOST_PP_COMMA_IF(BOOST_PP_DEC(NN))
                BOOST_PP_ENUM_SHIFTED(NN, HPX_UTIL_BIND_EVAL, _)).get();
    }
    template <BOOST_PP_ENUM_PARAMS(NNN, typename A)>
    result_type operator()(HPX_ENUM_FWD_ARGS(NNN, A, a)) const
    {
        typedef
            BOOST_PP_CAT(hpx::util::tuple, NNN)<
                BOOST_PP_ENUM(NNN, HPX_UTIL_BIND_REFERENCE, A)
            >
            env_type;
        env_type env(HPX_ENUM_FORWARD_ARGS(NNN, A, a));
        return hpx::async<Action>(
            hpx::util::detail::eval(env, arg0)
          BOOST_PP_COMMA_IF(BOOST_PP_DEC(NN))
                BOOST_PP_ENUM_SHIFTED(NN, HPX_UTIL_BIND_EVAL, _)).get();
    }

#undef NNN
