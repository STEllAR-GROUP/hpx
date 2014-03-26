//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file copy_component.hpp

#if !defined(HPX_RUNTIME_COMPONENTS_COPY_COMPONENT_SEP_20_2013_0828PM)
#define HPX_RUNTIME_COMPONENTS_COPY_COMPONENT_SEP_20_2013_0828PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/server/copy_component.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/async_colocated.hpp>
#include <hpx/traits/is_component.hpp>

#include <boost/utility/enable_if.hpp>

namespace hpx { namespace components
{
    /// \brief Copy given component to the specified target locality
    ///
    /// The function \a copy<Component> will create a copy of the component
    /// referenced by \a to_copy on the locality specified with
    /// \a target_locality. It returns a future referring to the newly created
    /// component instance.
    ///
    /// \param to_copy         [in] The global id of the component to copy
    /// \param target_locality [in, optional] The locality where the copy 
    ///                        should be created (default is same locality 
    ///                        as source).
    ///
    /// \tparam  The only template argument specifies the component type to 
    ///          create.
    ///
    /// \returns A future representing the global id of the newly (copied)
    ///          component instance.
    ///
    /// \note If the second argument is omitted (or is invalid_id) the
    ///       new component instance is created on the locality of the
    ///       component instance which is to be copied.
    ///
    template <typename Component>
#if defined(DOXYGEN)
    future<naming::id_type>
#else
    inline typename boost::enable_if<
        traits::is_component<Component>, future<naming::id_type>
    >::type 
#endif
    copy(naming::id_type const& to_copy,
        naming::id_type const& target_locality = naming::invalid_id)
    {
        typedef server::copy_component_action<Component> action_type;
        return async_colocated<action_type>(to_copy, to_copy, target_locality);
    }
}}

#endif
