//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_COMPONENTS_NEW_OCT_10_2012_1256PM)
#define HPX_RUNTIME_COMPONENTS_NEW_OCT_10_2012_1256PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/util/move.hpp>
#include <hpx/traits/is_component.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/enum_params.hpp>

namespace hpx { namespace components
{
    template <typename Component>
    inline typename boost::enable_if<
        traits::is_component<Component>, 
        lcos::future<naming::id_type, naming::gid_type>
    >::type
    new_(id_type const& locality)
    {
        return components::stub_base<Component>::create_async(locality);
    }

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/components/preprocessed/new.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/new_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
     "hpx/runtime/components/new.hpp"))                                       \
/**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

}}

#endif // HPX_NEW_OCT_10_2012_1256PM

#else  // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

    template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline typename boost::enable_if<
        traits::is_component<Component>, 
        lcos::future<naming::id_type, naming::gid_type>
    >::type
    new_(id_type const& locality, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return components::stub_base<Component>::create_async(locality,
            HPX_ENUM_FORWARD_ARGS(N , Arg, arg));
    }

#undef N

#endif
