//  Copyright (c) 2007-2013 Hartmut Kaiser
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
#include <hpx/traits/is_component.hpp>

#include <boost/utility/enable_if.hpp>

namespace hpx { namespace components
{
    /// \cond NOINTERNAL
    namespace detail
    {
        template <typename Component>
        future<naming::id_type> copy_component(
            naming::id_type const& to_copy, naming:id_type target_locality)
        {
            typedef typename server::copy_component_action<Component> action_type;
            return async<actions_type>(to_copy, target_locality);
        }

        template <typename Component>
        future<naming::id_type> copy_component_postproc(
            naming::id_type const& to_copy, future<naming:id_type>& f)
        {
            return detail::copy_component<Component>(to_copy, f.get());
        }
    }
    /// \endcond

    /// \brief Copy given component to the specified target locality
    ///
    /// The action \a copy_component will create a copy of the component
    /// referenced by \a to_copy on the locality specified with
    /// \a target_locality.
    ///
    /// \param to_copy         [in] The global id of the component to copy
    /// \param target_locality [in] The locality to create the copy on.
    ///
    /// \returns A future representing the global id of the newly (copied)
    ///          component instance.
    ///
    /// \note If the second argument is ommitted (or is invalid_id) the
    ///       new component instance is created on the locality of the
    ///       component instance which is to be copied.
    ///
    template <typename Component>
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type
    copy_component(naming::id_type const& to_copy,
        naming:id_type target_locality = naming::invalid_id)
    {
        if (!target_locality) {
            future<naming:id_type> f = get_colocation_id_async(to_copy);
            return f.then(util::bind(&detail::copy_component_postproc<Component>,
                to_copy, util::placeholders::_1));
        }
        return detail::copy_component<Component>(to_copy, target_locality);
    }
}}

#endif
