//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file get_ptr.hpp

#if !defined(HPX_RUNTIME_GET_PTR_SEP_18_2013_0622PM)
#define HPX_RUNTIME_GET_PTR_SEP_18_2013_0622PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/get_lva.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>

#include <boost/shared_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    /// \cond NOINTERNAL
    namespace detail
    {
        struct get_ptr_deleter
        {
            get_ptr_deleter(naming::id_type const& id) : id_(id) {}

            template <typename Component>
            void operator()(Component*)
            {
                id_ = naming::invalid_id;       // release component
            }

            naming::id_type id_;                // hold component alive
        };

        template <typename Component>
        boost::shared_ptr<Component>
        get_ptr_postproc(future<naming::address>& f, naming::id_type const& id)
        {
            naming::address addr = f.get();

            if (get_locality() != addr.locality_)
            {
                HPX_THROW_EXCEPTION(bad_parameter, "hpx::get_ptr",
                    "the given component id does not belong to a local object");
                return boost::shared_ptr<Component>();
            }

            if (!components::types_are_compatible(addr.type_,
                    components::get_component_type<Component>()))
            {
                HPX_THROW_EXCEPTION(bad_component_type, "hpx::get_ptr",
                    "requested component type does not match the given component id");
                return boost::shared_ptr<Component>();
            }

            Component* p = get_lva<Component>::call(addr.address_);
            return boost::shared_ptr<Component>(p, detail::get_ptr_deleter(id));
        }
    }
    /// \endcond

    /// \brief Returns a future referring to a the pointer to the underlying memory of a component
    ///
    /// The function hpx::get_ptr_async can be used to extract a future
    /// referring to the pointer to the underlying memory of a given component.
    ///
    /// \param id  [in] The global id of the component for which the pointer
    ///            to the underlying memory should be retrieved.
    ///
    /// \tparam    The only template parameter has to be the type of the
    ///            server side component.
    ///
    /// \returns   This function returns a future representing the pointer to
    ///            the underlying memory for the component instance with the
    ///            given \a id.
    ///
    /// \note      This function will successfully return the requested result
    ///            only if the given component is currently located on the the
    ///            calling locality. Otherwise the function will raise an
    ///            error.
    ///
    template <typename Component>
    hpx::future<boost::shared_ptr<Component> >
    get_ptr_async(naming::id_type const& id)
    {
        using util::placeholders::_1;
        future<naming::address> f = agas::resolve_async(id);
        return f.then(util::bind(&detail::get_ptr_postproc<Component>, _1, id));
    }
    
    /// \brief Returns the pointer to the underlying memory of a component
    ///
    /// The function hpx::get_ptr can be used to extract the pointer to the
    /// underlying memory of a given component.
    ///
    /// \param id  [in] The global id of the component for which the pointer
    ///            to the underlying memory should be retrieved.
    /// \param ec  [in,out] this represents the error status on exit, if this
    ///            is pre-initialized to \a hpx#throws the function will throw
    ///            on error instead.
    ///
    /// \tparam    The only template parameter has to be the type of the
    ///            server side component.
    ///
    /// \returns   This function returns the pointer to the underlying memory
    ///            for the component instance with the given \a id.
    ///
    /// \note      This function will successfully return the requested result
    ///            only if the given component is currently located on the the
    ///            requesting locality. Otherwise the function will raise and
    ///            error.
    ///
    /// \note      As long as \a ec is not pre-initialized to \a hpx::throws this
    ///            function doesn't throw but returns the result code using the
    ///            parameter \a ec. Otherwise it throws an instance of
    ///            hpx::exception.
    ///
    template <typename Component>
    boost::shared_ptr<Component>
    get_ptr(naming::id_type const& id, error_code& ec = throws)
    {
        hpx::future<boost::shared_ptr<Component> > ptr =
            get_ptr_async<Component>(id);
        return ptr.get(ec);
    }
}

#endif
