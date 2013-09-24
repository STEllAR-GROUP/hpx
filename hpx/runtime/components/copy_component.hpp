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
        // This creates the new copy of the component on the same locality 
        // as the source.
        // This will be called when f is ready().
        template <typename Component>
        naming::id_type copy_same_locality(naming::id_type const& to_copy,
            future<naming::id_type>& f)
        {
            typedef typename server::copy_component_action<Component>
                action_type;

            naming::id_type component_locality(f.move());
            return async<action_type>(component_locality, to_copy,
                component_locality).move();
        }

        // This creates the new copy of the component on the same locality 
        // as the source.
        // This will be called when f is ready().
        template <typename Component>
        naming::id_type copy_any_locality(naming::id_type const& to_copy,
            naming::id_type const& target_locality,
            future<naming::id_type>& f)
        {
            typedef typename server::copy_component_action<Component>
                action_type;
            return async<action_type>(f.move(), to_copy, target_locality).move();
        }
    }
    /// \endcond

    /// \brief Copy given component to the specified target locality
    ///
    /// The function \a copy<Component> will create a copy of the component
    /// referenced by \a to_copy on the locality specified with
    /// \a target_locality. It returns a future referring to the newly created
    /// component instance.
    ///
    /// \param to_copy         [in] The global id of the component to copy
    /// \param target_locality [in, optional] The locality to create the copy 
    ///                        to (default is same locality as source).
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
    inline typename boost::enable_if<
        traits::is_component<Component>, lcos::future<naming::id_type>
    >::type 
    copy(naming::id_type const& to_copy,
        naming::id_type const& target_locality = naming::invalid_id)
    {
        future<naming::id_type> f = get_colocation_id(to_copy);

        // if the new component should be created on the same locality as 
        // the source component, we simply invoke the copy operation there.
        if (!target_locality) {
            return f.then(util::bind(
                &detail::copy_same_locality<Component>,
                to_copy, util::placeholders::_1));
        }

        // at this point it is not clear what locality the new component
        // should be created on
        return f.then(util::bind(
            &detail::copy_any_locality<Component>,
            to_copy, target_locality, util::placeholders::_1));
    }
}}

#endif
