//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011-2017 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/assert.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/traits/is_component.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming/credit_handling.hpp>
#include <hpx/runtime/components/server/component_heap.hpp>
#include <hpx/runtime/components/server/create_component_fwd.hpp>
#include <hpx/runtime/components/server/wrapper_heap.hpp>
#include <hpx/runtime/components/server/wrapper_heap_list.hpp>
#include <hpx/runtime/components_fwd.hpp>
#include <hpx/traits/managed_component_policies.hpp>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    template <typename Component, typename Derived>
    class managed_component;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail_adl_barrier
    {
        template <typename BackPtrTag>
        struct init;

        template <>
        struct init<traits::construct_with_back_ptr>
        {
            template <typename Component, typename Managed>
            static constexpr void call(
                Component* /* component */, Managed* /* this_ */)
            {
            }

            template <typename Component, typename Managed, typename ...Ts>
            static void call_new(Component*& component, Managed* this_, Ts&&... vs)
            {
                typedef typename Managed::wrapped_type wrapped_type;
                component = new wrapped_type(this_, std::forward<Ts>(vs)...);
            }
        };

        template <>
        struct init<traits::construct_without_back_ptr>
        {
            template <typename Component, typename Managed>
            static void call(Component* component, Managed* this_)
            {
                component->set_back_ptr(this_);
            }

            template <typename Component, typename Managed, typename ...Ts>
            static void call_new(Component*& component, Managed* this_, Ts&&... vs)
            {
                typedef typename Managed::wrapped_type wrapped_type;
                component = new wrapped_type(std::forward<Ts>(vs)...);
                component->set_back_ptr(this_);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // This is used by the component implementation to decide whether to
        // delete the managed_component instance it depends on.
        template <typename DtorTag>
        struct destroy_backptr;

        template <>
        struct destroy_backptr<traits::managed_object_is_lifetime_controlled>
        {
            template <typename BackPtr>
            static void call(BackPtr* back_ptr)
            {
                // The managed_component's controls the lifetime of the
                // component implementation.
                back_ptr->finalize();
                back_ptr->~BackPtr();
                component_heap<typename BackPtr::wrapped_type>().free(back_ptr);
            }
        };

        template <>
        struct destroy_backptr<traits::managed_object_controls_lifetime>
        {
            template <typename BackPtr>
            static constexpr void call(BackPtr*)
            {
                // The managed_component's lifetime is controlled by the
                // component implementation. Do nothing.
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // This is used by the managed_component to decide whether to
        // delete the component implementation depending on it.
        template <typename DtorTag>
        struct manage_lifetime;

        template <>
        struct manage_lifetime<traits::managed_object_is_lifetime_controlled>
        {
            template <typename Component>
            static constexpr void call(Component*)
            {
                // The managed_component's lifetime is controlled by the
                // component implementation. Do nothing.
            }

            template <typename Component>
            static void addref(Component* component)
            {
                intrusive_ptr_add_ref(component);
            }

            template <typename Component>
            static void release(Component* component)
            {
                intrusive_ptr_release(component);
            }
        };

        template <>
        struct manage_lifetime<traits::managed_object_controls_lifetime>
        {
            template <typename Component>
            static void call(Component* component)
            {
                // The managed_component controls the lifetime of the
                // component implementation.
                component->finalize();
                delete component;
            }

            template <typename Component>
            static constexpr void addref(Component*)
            {
            }

            template <typename Component>
            static constexpr void release(Component*)
            {
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct base_managed_component
          : public traits::detail::managed_component_tag
        {
            /// \brief finalize() will be called just before the instance gets
            ///        destructed
            static constexpr void finalize() {}

#if defined(HPX_DISABLE_ASSERTS) || defined(BOOST_DISABLE_ASSERTS) || defined(NDEBUG)
            static constexpr void mark_as_migrated()
            {
            }
            static constexpr void on_migrated()
            {
            }
#else
            static void mark_as_migrated()
            {
                // If this assertion is triggered then this component instance is
                // being migrated even if the component type has not been enabled
                // to support migration.
                HPX_ASSERT(false);
            }

            static void on_migrated()
            {
                // If this assertion is triggered then this component instance is being
                // migrated even if the component type has not been enabled to support
                // migration.
                HPX_ASSERT(false);
            }
#endif
        };
    }

    template <typename Component, typename Wrapper,
        typename CtorPolicy, typename DtorPolicy>
    class managed_component_base : public detail::base_managed_component
    {
    public:
        HPX_NON_COPYABLE(managed_component_base);

    public:
        typedef typename std::conditional<
            std::is_same<Component, detail::this_type>::value,
            managed_component_base, Component
        >::type this_component_type;

        typedef this_component_type wrapped_type;

        typedef void has_managed_component_base;
        typedef CtorPolicy ctor_policy;
        typedef DtorPolicy dtor_policy;

        // make sure that we have a back_ptr whenever we need to control the
        // lifetime of the managed_component
        static_assert((
            std::is_same<ctor_policy, traits::construct_without_back_ptr>::value ||
            std::is_same<dtor_policy,
            traits::managed_object_controls_lifetime>::value),
            "std::is_same<ctor_policy, traits::construct_without_back_ptr>::value || "
            "std::is_same<dtor_policy, "
            "traits::managed_object_controls_lifetime>::value");

        managed_component_base()
          : back_ptr_(nullptr)
        {}

        explicit managed_component_base(
            managed_component<Component, Wrapper>* back_ptr)
          : back_ptr_(back_ptr)
        {
            HPX_ASSERT(back_ptr);
        }

        // The implementation of the component is responsible for deleting the
        // actual managed component object
        ~managed_component_base()
        {
            detail_adl_barrier::destroy_backptr<dtor_policy>::call(back_ptr_);
        }

        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this
        // component
        typedef managed_component<Component, Wrapper> wrapping_type;
        typedef Component base_type_holder;

        naming::id_type get_unmanaged_id() const;
        naming::id_type get_id() const;

    protected:
        naming::gid_type get_base_gid() const;

    protected:
        template <typename>
        friend struct detail_adl_barrier::init;

        void set_back_ptr(components::managed_component<Component, Wrapper>* bp)
        {
            HPX_ASSERT(nullptr == back_ptr_);
            HPX_ASSERT(bp);
            back_ptr_ = bp;
        }

    private:
        managed_component<Component, Wrapper>* back_ptr_;
    };

    // reference counting
    template <typename Component, typename Derived>
    void intrusive_ptr_add_ref(managed_component<Component, Derived>* p)
    {
        detail_adl_barrier::manage_lifetime<
            typename traits::managed_component_dtor_policy<Component>::type
        >::addref(p->component_);
    }
    template <typename Component, typename Derived>
    void intrusive_ptr_release(managed_component<Component, Derived>* p)
    {
        detail_adl_barrier::manage_lifetime<
            typename traits::managed_component_dtor_policy<Component>::type
        >::release(p->component_);
    }

    namespace detail
    {
        template <typename T>
        class fixed_wrapper_heap;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The managed_component template is used as a indirection layer
    /// for components allowing to gracefully handle the access to non-existing
    /// components.
    ///
    /// Additionally it provides memory management capabilities for the
    /// wrapping instances, and it integrates the memory management with the
    /// AGAS service. Every instance of a managed_component gets assigned
    /// a global id.
    /// The provided memory management allocates the managed_component
    /// instances from a special heap, ensuring fast allocation and avoids a
    /// full network round trip to the AGAS service for each of the allocated
    /// instances.
    ///
    /// \tparam Component
    /// \tparam Derived
    ///
    template <typename Component, typename Derived>
    class managed_component
    {
    public:
        HPX_NON_COPYABLE(managed_component);

    public:
        typedef typename std::conditional<
                std::is_same<Derived, detail::this_type>::value,
                managed_component, Derived
            >::type derived_type;

        typedef Component wrapped_type;
        typedef Component type_holder;
        typedef typename Component::base_type_holder base_type_holder;

        typedef detail::wrapper_heap_list<
            detail::fixed_wrapper_heap<derived_type> > heap_type;
        typedef derived_type value_type;

        /// \brief Construct a managed_component instance holding a
        ///        wrapped instance. This constructor takes ownership of the
        ///        passed pointer.
        ///
        /// \param c    [in] The pointer to the wrapped instance. The
        ///             managed_component takes ownership of this pointer.
        explicit managed_component(Component* comp)
          : component_(comp)
        {
            detail_adl_barrier::init<
                typename traits::managed_component_ctor_policy<Component>::type
            >::call(component_, this);
            intrusive_ptr_add_ref(this);
        }

    public:
        /// \brief Construct a managed_component instance holding a new wrapped
        ///        instance
        template <typename ...Ts>
        managed_component(Ts&&... vs)
          : component_(nullptr)
        {
            detail_adl_barrier::init<
                typename traits::managed_component_ctor_policy<Component>::type
            >::call_new(component_, this, std::forward<Ts>(vs)...);
            intrusive_ptr_add_ref(this);
        }

    public:
        /// \brief The destructor releases any wrapped instances
        ~managed_component()
        {
            intrusive_ptr_release(this);
            detail_adl_barrier::manage_lifetime<
                typename traits::managed_component_dtor_policy<Component>::type
            >::call(component_);
        }

        /// \brief finalize() will be called just before the instance gets
        ///        destructed
        ///
        static constexpr void finalize() {}

        /// \brief Return a pointer to the wrapped instance
        /// \note  Caller must check validity of returned pointer
        Component* get()
        {
            return component_;
        }
        Component const* get() const
        {
            return component_;
        }

        Component* get_checked()
        {
            if (!component_) {
                std::ostringstream strm;
                strm << "component is nullptr ("
                     << components::get_component_type_name(
                        components::get_component_type<wrapped_type>())
                     << ") gid(" << get_base_gid() << ")";
                HPX_THROW_EXCEPTION(invalid_status,
                    "managed_component<Component, Derived>::get_checked",
                    strm.str());
            }
            return get();
        }

        Component const* get_checked() const
        {
            if (!component_) {
                std::ostringstream strm;
                strm << "component is nullptr ("
                     << components::get_component_type_name(
                        components::get_component_type<wrapped_type>())
                     << ") gid(" << get_base_gid() << ")";
                HPX_THROW_EXCEPTION(invalid_status,
                    "managed_component<Component, Derived>::get_checked",
                    strm.str());
            }
            return get();
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        // The managed_component behaves just like the wrapped object
        Component* operator-> ()
        {
            return get_checked();
        }

        Component const* operator-> () const
        {
            return get_checked();
        }

        ///////////////////////////////////////////////////////////////////////
        Component& operator* ()
        {
            return *get_checked();
        }

        Component const& operator* () const
        {
            return *get_checked();
        }

        ///////////////////////////////////////////////////////////////////////
        /// \brief Return the global id of this \a future instance
        naming::id_type get_unmanaged_id() const
        {
            return naming::id_type(get_base_gid(), naming::id_type::unmanaged);
        }

    private:
#if !defined(__NVCC__) && !defined(__CUDACC__)
        // declare friends which are allowed to access get_base_gid()
        friend Component;

        template <typename Component_, typename Wrapper_,
            typename CtorPolicy, typename DtorPolicy>
        friend class managed_component_base;

        template <typename Component_, typename...Ts>
        friend naming::gid_type server::create(Ts&&... ts);

        template <typename Component_, typename...Ts>
        friend naming::gid_type server::create_migrated(
            naming::gid_type const& gid, void** p, Ts&&...ts);

        template <typename Component_, typename...Ts>
        friend std::vector<naming::gid_type> bulk_create(std::size_t count, Ts&&...ts);
#else
    public:
#endif

        naming::gid_type get_base_gid(
            naming::gid_type const& assign_gid = naming::invalid_gid) const
        {
            if (assign_gid)
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    "managed_component::get_base_gid",
                    "managed_components must be assigned new gids on creation");
                return naming::invalid_gid;
            }
            return component_heap<managed_component>().
                get_gid(const_cast<managed_component*>(this));
        }

    public:
        // reference counting
        template<typename C, typename D>
        friend void intrusive_ptr_add_ref(managed_component<C, D>* p);

        template<typename C, typename D>
        friend void intrusive_ptr_release(managed_component<C, D>* p);

    protected:
        Component* component_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Wrapper,
        typename CtorPolicy, typename DtorPolicy>
    inline naming::id_type
    managed_component_base<Component, Wrapper, CtorPolicy, DtorPolicy>::
        get_unmanaged_id() const
    {
        HPX_ASSERT(back_ptr_);
        return back_ptr_->get_unmanaged_id();
    }

    template <typename Component, typename Wrapper,
        typename CtorPolicy, typename DtorPolicy>
    inline naming::id_type
    managed_component_base<Component, Wrapper, CtorPolicy, DtorPolicy>::
        get_id() const
    {
        // all credits should have been taken already
        naming::gid_type gid =
            static_cast<Component const&>(*this).get_base_gid();

        // The underlying heap will always give us a full set of credits, but
        // those are valid for the first invocation of get_base_gid() only.
        // We have to get rid of those credits and properly replenish those.
        naming::detail::strip_credits_from_gid(gid);

        // any invocation causes the credits to be replenished
        naming::detail::replenish_credits(gid);
        return naming::id_type(gid, naming::id_type::managed);
    }

    template <typename Component, typename Wrapper,
        typename CtorPolicy, typename DtorPolicy>
    inline naming::gid_type
    managed_component_base<Component, Wrapper, CtorPolicy, DtorPolicy>::
        get_base_gid() const
    {
        HPX_ASSERT(back_ptr_);
        return back_ptr_->get_base_gid();
    }
}}



