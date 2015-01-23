//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file new.hpp

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

    /// \brief Create a new instance of the given Component type on the
    ///        co-located with the specified object.
    ///
    /// This function creates a new instance of the given Component type
    /// on the specified locality the given object is currently located on
    /// and returns a future object for the global address which can be used
    /// to reference the new component instance.
    ///
    /// \param id        [in] The global address of an object defining the
    ///                  locality where the new instance should be created on.
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
    new_colocated(hpx::id_type const& id, Arg0 argN, ...);
#else
    template <typename Component, typename ...Ts>
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type
    new_(id_type const& locality, Ts&&... vs)
    {
        return components::stub_base<Component>::create_async(locality,
            std::forward<Ts>(vs)...);
    }

    template <typename Component, typename ...Ts>
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type
    new_colocated(id_type const& id, Ts&&... vs)
    {
        return components::stub_base<Component>::create_colocated_async(id,
            std::forward<Ts>(vs)...);
    }
#endif // !defined(DOXYGEN)
}}

namespace hpx
{
    using hpx::components::new_;
    using hpx::components::new_colocated;
}

#endif // HPX_NEW_OCT_10_2012_1256PM
