//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_COMPONENTS_SERVER_GENERIC_COMPONENT_IMPLEMENTTAION_OCT_12_2008_0845PM)
#define HPX_COMPONENTS_SERVER_GENERIC_COMPONENT_IMPLEMENTTAION_OCT_12_2008_0845PM

#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/comma_if.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/components/server/generic_component_implementation.hpp"))    \
    /**/
    
#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()
#define HPX_EVAL_ARGUMENT(z, n, data) BOOST_PP_COMMA_IF(n) boost::fusion::at_c<n>(p)

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Result, 
        BOOST_PP_ENUM_PARAMS(N, typename T),
        Result (*F)(threads::thread_self&, applier::applier&,
            BOOST_PP_ENUM_PARAMS(N, T))
    >
    class BOOST_PP_CAT(generic_component, N)
      : public simple_component_base<
            BOOST_PP_CAT(generic_component, N)<Result, BOOST_PP_ENUM_PARAMS(N, T), F> 
        >
    {
    private:
        typedef simple_component_base<
            BOOST_PP_CAT(generic_component, N)<Result, BOOST_PP_ENUM_PARAMS(N, T), F> 
        > base_type;

    public:
        typedef Result result_type;
        typedef typename 
            boost::fusion::result_of::as_vector<
                boost::mpl::vector<BOOST_PP_ENUM_PARAMS(N, T)> 
            >::type
        parameter_block_type;

        // parcel action code: the action to be performed 
        enum actions
        {
            generic_component_action = N
        };

        BOOST_PP_CAT(generic_component, N)(applier::applier& appl)
          : base_type(appl)
        {}

        threads::thread_state
        eval (threads::thread_self& self, applier::applier& appl, Result* r,
            parameter_block_type const& p) 
        {
            if (NULL != r) {
                *r = F(self, appl, BOOST_PP_REPEAT(N, HPX_EVAL_ARGUMENT, _));
            }
            else {
                F(self, appl, BOOST_PP_REPEAT(N, HPX_EVAL_ARGUMENT, _));
            }
            return threads::terminated;
        }

        typedef hpx::actions::result_action1<
            BOOST_PP_CAT(generic_component, N), Result, 
            generic_component_action, parameter_block_type const&, 
            &BOOST_PP_CAT(generic_component, N)::eval
        > eval_action;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        BOOST_PP_ENUM_PARAMS(N, typename T),
        void (*F)(threads::thread_self&, applier::applier&,
            BOOST_PP_ENUM_PARAMS(N, T))
    >
    class BOOST_PP_CAT(generic_component, N)<void, BOOST_PP_ENUM_PARAMS(N, T), F>
      : public simple_component_base<
            BOOST_PP_CAT(generic_component, N)<void, BOOST_PP_ENUM_PARAMS(N, T), F> 
        >
    {
    private:
        typedef simple_component_base<
            BOOST_PP_CAT(generic_component, N)<void, BOOST_PP_ENUM_PARAMS(N, T), F> 
        > base_type;

    public:
        typedef void result_type;
        typedef typename 
            boost::fusion::result_of::as_vector<
                boost::mpl::vector<BOOST_PP_ENUM_PARAMS(N, T)> 
            >::type
        parameter_block_type;

        // parcel action code: the action to be performed 
        enum actions
        {
            generic_component_action = N
        };

        BOOST_PP_CAT(generic_component, N)(applier::applier& appl)
          : base_type(appl)
        {}

        threads::thread_state 
        eval (threads::thread_self& self, applier::applier& appl,
            parameter_block_type const& p) 
        {
            F(self, appl, BOOST_PP_REPEAT(N, HPX_EVAL_ARGUMENT, _));
            return threads::terminated;
        }

        typedef hpx::actions::action1<
            BOOST_PP_CAT(generic_component, N), 
            generic_component_action, parameter_block_type const&, 
            &BOOST_PP_CAT(generic_component, N)::eval
        > eval_action;
    };

#undef HPX_EVAL_ARGUMENT
#undef N

#endif
