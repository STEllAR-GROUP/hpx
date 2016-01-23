//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SIMPLE_COMPONENT_BASE_JUL_18_2008_0948PM)
#define HPX_COMPONENTS_SIMPLE_COMPONENT_BASE_JUL_18_2008_0948PM

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/traits/is_component.hpp>
#include <hpx/runtime/applier_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/create_component_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/applier/bind_naming_wrappers.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/util/unique_function.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>

#include <sstream>
#include <utility>

namespace hpx { namespace detail
{
    HPX_API_EXPORT naming::gid_type get_next_id(std::size_t count = 1);
}}

namespace hpx { namespace components
{
    template <typename Component>
    class simple_component;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    class simple_component_base : public traits::detail::simple_component_tag
    {
    protected:
        typedef typename boost::mpl::if_<
                boost::is_same<Component, detail::this_type>,
                simple_component_base, Component
            >::type this_component_type;

    public:
        typedef this_component_type wrapped_type;
        typedef this_component_type base_type_holder;
        typedef simple_component<this_component_type> wrapping_type;

        /// \brief Construct an empty simple_component
        simple_component_base()
        {}

        /// \brief Destruct a simple_component
        ~simple_component_base()
        {
            if (gid_) applier::unbind_gid_local(gid_);
        }

        // Copy construction and copy assignment should not copy the gid_.
        simple_component_base(simple_component_base const&) {}

        simple_component_base& operator=(simple_component_base const&)
        {
            return *this;
        }

        // just move our gid_
        simple_component_base(simple_component_base && rhs)
          : gid_(std::move(rhs.gid_))
        {}

        simple_component_base& operator=(simple_component_base && rhs)
        {
            if (this != &rhs)
                gid_ = std::move(rhs.gid_);
            return *this;
        }

        /// \brief finalize() will be called just before the instance gets
        ///        destructed
        void finalize() {}

        // This exposes the component type.
        static component_type get_component_type()
        {
            return components::get_component_type<this_component_type>();
        }
        static void set_component_type(component_type type)
        {
            components::set_component_type<this_component_type>(type);
        }

        naming::address get_current_address() const
        {
            return naming::address(get_locality(),
                components::get_component_type<wrapped_type>(),
                boost::uint64_t(static_cast<this_component_type const*>(this)));
        }

    protected:
        // declare friends which are allowed to access get_base_gid()
        template <typename Component_>
        friend naming::gid_type server::create(std::size_t count);

        template <typename Component_>
        friend naming::gid_type server::create(
            util::unique_function_nonser<void(void*)> const& ctor);

        template <typename Component_>
        friend naming::gid_type server::create(naming::gid_type const& gid,
            util::unique_function_nonser<void(void*)> const& ctor);

        // Create a new GID (if called for the first time), assign this
        // GID to this instance of a component and register this gid
        // with the AGAS service
        //
        // Returns he global id (GID) assigned to this instance of a component
        naming::gid_type get_base_gid(
            naming::gid_type const& assign_gid = naming::invalid_gid) const
        {
            if (!gid_)
            {
                naming::address addr(get_current_address());
                if (!assign_gid)
                {
                    gid_ = hpx::detail::get_next_id();
                    if (!applier::bind_gid_local(gid_, addr))
                    {
                        std::ostringstream strm;
                        strm << "failed to bind id " << gid_
                             << "to locality: " << hpx::get_locality();

                        gid_ = naming::invalid_gid;   // invalidate GID

                        HPX_THROW_EXCEPTION(duplicate_component_address,
                            "simple_component_base<Component>::get_base_gid",
                            strm.str());
                    }
                }
                else
                {
                    applier::applier& appl = hpx::applier::get_applier();
                    gid_ = assign_gid;
                    naming::detail::strip_credits_from_gid(gid_);

                    if (!agas::bind_sync(gid_, addr, appl.get_locality_id()))
                    {
                        std::ostringstream strm;
                        strm << "failed to rebind id " << gid_
                             << "to locality: " << hpx::get_locality();

                        gid_ = naming::invalid_gid;   // invalidate GID

                        HPX_THROW_EXCEPTION(duplicate_component_address,
                            "simple_component_base<Component>::get_base_gid",
                            strm.str());
                    }
                }
            }

            naming::gid_type::mutex_type::scoped_lock l(gid_.get_mutex());

            if (!naming::detail::has_credits(gid_))
            {
                naming::gid_type gid = gid_;
                return gid;
            }

            // on first invocation take all credits to avoid a self reference
            naming::gid_type gid = gid_;

            naming::detail::strip_credits_from_gid(
                const_cast<naming::gid_type&>(gid_));

            HPX_ASSERT(naming::detail::has_credits(gid));

            // We have to assume this credit was split as otherwise the gid
            // returned at this point will control the lifetime of the
            // component.
            naming::detail::set_credit_split_mask_for_gid(gid);
            return gid;
        }

    public:
        naming::id_type get_id() const
        {
            // all credits should have been taken already
            naming::gid_type gid = get_base_gid();
            HPX_ASSERT(!naming::detail::has_credits(gid));

            // any (subsequent) invocation causes the credits to be replenished
            naming::detail::replenish_credits(gid);
            return naming::id_type(gid, naming::id_type::managed);
        }

        naming::id_type get_unmanaged_id() const
        {
            return naming::id_type(get_base_gid(), naming::id_type::managed);
        }

#if defined(HPX_HAVE_COMPONENT_GET_GID_COMPATIBILITY)
        naming::id_type get_gid() const
        {
            return get_id();
        }
#endif

        /// This is the default hook implementation for decorate_action which
        /// does no hooking at all.
        template <typename F>
        static threads::thread_function_type
        decorate_action(naming::address::address_type, F && f)
        {
            return std::forward<F>(f);
        }

        /// This is the default hook implementation for schedule_thread which
        /// forwards to the default scheduler.
        static void schedule_thread(naming::address::address_type,
            threads::thread_init_data& data,
            threads::thread_state_enum initial_state)
        {
            hpx::threads::register_work_plain(data, initial_state); //-V106
        }

#if defined(HPX_HAVE_SECURITY)
        static components::security::capability get_required_capabilities(
            components::security::traits::capability<>::capabilities caps)
        {
            return components::default_component_creation_capabilities(caps);
        }
#endif

        // This component type requires valid id for its actions to be invoked
        static bool is_target_valid(naming::id_type const& id)
        {
            return !naming::is_locality(id);
        }

        // This component type does not support migration.
        static HPX_CONSTEXPR bool supports_migration() { return false; }

        // Pinning functionality
        void pin() {}
        void unpin() {}
        boost::uint32_t pin_count() const { return 0; }

        void mark_as_migrated()
        {
            // If this assertion is triggered then this component instance is
            // being migrated even if the component type has not been enabled
            // to support migration.
            HPX_ASSERT(false);
        }

    protected:
        mutable naming::gid_type gid_;
    };

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Component>
        struct simple_heap_factory
        {
            static Component* alloc(std::size_t count)
            {
                HPX_ASSERT(1 == count);
                return static_cast<Component*>
                    (::operator new(sizeof(Component)));
            }
            static void free(void* p, std::size_t count)
            {
                HPX_ASSERT(1 == count);
                ::operator delete(p);
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    class simple_component : public Component
    {
    public:
        typedef Component type_holder;
        typedef simple_component<Component> component_type;
        typedef component_type derived_type;
        typedef detail::simple_heap_factory<component_type> heap_type;

        /// \brief Construct a simple_component instance holding a new wrapped
        ///        instance
        template <typename ...Ts>
        simple_component(Ts&&... vs)
          : Component(std::forward<Ts>(vs)...)
        {}

        /// \brief  The function \a create is used for allocation and
        ///         initialization of instances of the derived components.
        static component_type* create(std::size_t count)
        {
            // simple components can be created individually only
            HPX_ASSERT(1 == count);
            return static_cast<component_type*>(new Component()); //-V572
        }

        /// \brief  The function \a destroy is used for destruction and
        ///         de-allocation of instances of the derived components.
        static void destroy(Component* p, std::size_t count = 1)
        {
            // simple components can be deleted individually only
            HPX_ASSERT(1 == count);
            p->finalize();
            delete p;
        }
    };
}}

#endif
