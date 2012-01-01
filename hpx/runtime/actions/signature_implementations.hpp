////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_276195A7_AC33_470E_A638_A30E29A75BCF)
#define HPX_276195A7_AC33_470E_A638_A30E29A75BCF

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/signature_implementations.hpp"))                     \
    /**/

#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()
#define HPX_PARAM_TYPES(z, n, data)                                           \
        BOOST_PP_COMMA_IF(n)                                                  \
        BOOST_PP_CAT(data, n) const&                                          \
        BOOST_PP_CAT(BOOST_PP_CAT(data, n), _)                                \
    /**/

template <typename Result, BOOST_PP_ENUM_PARAMS(N, typename T)>
struct signature<
    Result, boost::fusion::vector<BOOST_PP_ENUM_PARAMS(N, T)>
> : base_action
{
    typedef boost::fusion::vector<BOOST_PP_ENUM_PARAMS(N, T)> arguments_type;
    typedef Result result_type;

    virtual result_type execute_function(
        naming::address::address_type lva,
        BOOST_PP_REPEAT(N, HPX_PARAM_TYPES, T)
    ) const = 0;

    virtual HPX_STD_FUNCTION<threads::thread_function_type>
    get_thread_function(naming::address::address_type lva,
        arguments_type const& args) const = 0;

    virtual HPX_STD_FUNCTION<threads::thread_function_type>
    get_thread_function(continuation_type& cont,
        naming::address::address_type lva,
        arguments_type const& args) const = 0;

    virtual threads::thread_init_data&
    get_thread_init_data(naming::address::address_type lva,
        threads::thread_init_data& data,
        arguments_type const& args) = 0;

    virtual threads::thread_init_data&
    get_thread_init_data(continuation_type& cont,
        naming::address::address_type lva,
        threads::thread_init_data& data,
        arguments_type const& args) = 0;

    /// serialization support
    static void register_base()
    {
        using namespace boost::serialization;
        void_cast_register<signature, base_action>();
    }
};

#undef HPX_PARAM_TYPES
#undef N

#endif

