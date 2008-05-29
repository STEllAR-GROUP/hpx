//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_COMPONENTS_ACTION_IMPLEMENTATIONS_MAY_20_2008_1104AM)
#define HPX_COMPONENTS_ACTION_IMPLEMENTATIONS_MAY_20_2008_1104AM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (3, HPX_ACTION_LIMIT,                                                 \
    "hpx/components/action_implementations.hpp"))                             \
    /**/
    
#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()
#define HPX_ATION_ARGUMENT(z, n, data) BOOST_PP_COMMA_IF(n) this->get<n>()

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version
    template <
        typename Component, int Action, BOOST_PP_ENUM_PARAMS(N, typename Arg),
        bool (Component::*F)(hpx::threadmanager::px_thread_self&, BOOST_PP_ENUM_PARAMS(N, Arg)) 
    >
    class BOOST_PP_CAT(action, N)
      : public action<
            Component, Action, 
            boost::fusion::vector<BOOST_PP_ENUM_PARAMS(N, Arg)> 
        >
    {
    private:
        typedef action<
            Component, Action, 
            boost::fusion::vector<BOOST_PP_ENUM_PARAMS(N, Arg)> 
        > base_type;
        
    public:
        BOOST_PP_CAT(action, N)() 
        {}

        // construct an action from its arguments
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(action, N)(BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg)) 
          : base_type(BOOST_PP_ENUM_PARAMS(N, arg)) 
        {}
        
    private:
        boost::function<bool (hpx::threadmanager::px_thread_self&)>
            get_thread_function(naming::address::address_type lva) const
        {
            return boost::bind(F, reinterpret_cast<Component*>(lva), _1,
                BOOST_PP_REPEAT(N, HPX_ATION_ARGUMENT, _));
        }

    private:
        // serialization support    
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<action>(*this);
        }
    };

#undef HPX_ATION_ARGUMENT
#undef N

#endif
