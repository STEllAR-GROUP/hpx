//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file get_ptr.hpp

#if !defined(HPX_RUNTIME_GET_PTR_SEP_18_2013_0622PM)
#define HPX_RUNTIME_GET_PTR_SEP_18_2013_0622PM

#include <hpx/config.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/runtime/get_lva.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>

#include <memory>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    /// \cond NOINTERNAL
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        struct get_ptr_deleter
        {
            get_ptr_deleter(naming::id_type const& id) : id_(id) {}

            template <typename Component>
            void operator()(Component* p)
            {
                id_ = naming::invalid_id;       // release component
                p->unpin();
            }

            naming::id_type id_;                // holds component alive
        };

        struct get_ptr_for_migration_deleter
        {
            get_ptr_for_migration_deleter(naming::id_type const& id)
              : id_(id)
            {}

            template <typename Component>
            void operator()(Component* p)
            {
                bool was_migrated = p->pin_count() == ~0x0u;
                p->unpin();

                HPX_ASSERT(was_migrated);
                if (was_migrated)
                {
                    using components::stubs::runtime_support;
                    agas::gva g (hpx::get_locality(),
                        components::get_component_type<Component>(), 1, p);
                    runtime_support::free_component_locally(g, id_.get_gid());
                }
                id_ = naming::invalid_id;       // release credits
            }

            naming::id_type id_;                // holds component alive
        };

        template <typename Component, typename Deleter>
        std::shared_ptr<Component>
        get_ptr_postproc_helper(naming::address const& addr,
            naming::id_type const& id)
        {
            if (get_locality() != addr.locality_)
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "hpx::get_ptr_postproc<Component, Deleter>",
                    "the given component id does not belong to a local object");
                return std::shared_ptr<Component>();
            }

            if (!traits::component_type_is_compatible<Component>::call(addr))
            {
                HPX_THROW_EXCEPTION(bad_component_type,
                    "hpx::get_ptr_postproc<Component, Deleter>",
                    "requested component type does not match the given component id");
                return std::shared_ptr<Component>();
            }

            Component* p = get_lva<Component>::call(addr.address_);
            std::shared_ptr<Component> ptr(p, Deleter(id));

            ptr->pin();     // the shared_ptr pins the component
            return ptr;
        }

        template <typename Component, typename Deleter>
        std::shared_ptr<Component>
        get_ptr_postproc(hpx::future<naming::address> f,
            naming::id_type const& id)
        {
            return get_ptr_postproc_helper<Component, Deleter>(f.get(), id);
        }

        ///////////////////////////////////////////////////////////////////////
        // This is similar to get_ptr<> below, except that the shared_ptr will
        // delete the local instance when it goes out of scope.
        template <typename Component>
        std::shared_ptr<Component>
        get_ptr_for_migration(naming::address const& addr,
            naming::id_type const& id)
        {
            return get_ptr_postproc_helper<
                    Component, get_ptr_for_migration_deleter
                >(addr, id);
        }
    }
    /// \endcond

    /// \brief Returns a future referring to a the pointer to the
    ///  underlying memory of a component
    ///
    /// The function hpx::get_ptr can be used to extract a future
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
    hpx::future<std::shared_ptr<Component> >
    get_ptr(naming::id_type const& id)
    {
        using util::placeholders::_1;
        hpx::future<naming::address> f = agas::resolve(id);
        return f.then(util::bind(
            &detail::get_ptr_postproc<Component, detail::get_ptr_deleter>,
            _1, id));
    }

    /// \brief Returns the pointer to the underlying memory of a component
    ///
    /// The function hpx::get_ptr_sync can be used to extract the pointer to
    /// the underlying memory of a given component.
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
    std::shared_ptr<Component>
    get_ptr_sync(naming::id_type const& id, error_code& ec = throws)
    {
        hpx::future<std::shared_ptr<Component> > ptr =
            get_ptr<Component>(id);
        return ptr.get(ec);
    }
}

#endif
