//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file new.hpp

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
#if defined(DOXYGEN)
    /// \brief Create a new instance of the given Component type on the
    ///        specified locality.
    ///
    /// This function creates a new instance of the given Component type
    /// on the specified locality and returns a future object for the
    /// global address which can be used to reference the new component
    /// instance.
    ///
    /// \param locality  [in] The global address of the locality where the
    ///                  new instance should be created on.
    /// \param argN      [in] Any number of arbitrary arguments (passed by
    ///                  value, by const reference or by rvalue reference)
    ///                  which will be forwarded to the constructor of
    ///                  the created component instance.
    ///
    /// \returns The function returns an \a hpx::future object instance
    ///          which can be used to retrieve the global address of the
    ///          newly created component.
    template <typename Component, typename ArgN, ...>
    hpx::future<hpx::id_type>
    new_(hpx::id_type const& locality, Arg0 argN, ...);
#else
    template <typename Component>
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type
    new_(id_type const& locality)
    {
        return components::stub_base<Component>::create_async(locality);
    }
#endif

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

namespace hpx
{
    using hpx::components::new_;
}

#endif // HPX_NEW_OCT_10_2012_1256PM

#else  // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

    template <typename Component, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type
    new_(id_type const& locality, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return components::stub_base<Component>::create_async(locality,
            HPX_ENUM_FORWARD_ARGS(N , Arg, arg));
    }

#undef N

#endif
