//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SIMPLE_COMPONENT_BASE_JUL_18_2008_0948PM)
#define HPX_COMPONENTS_SIMPLE_COMPONENT_BASE_JUL_18_2008_0948PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/applier/bind_naming_wrappers.hpp>
#include <hpx/util/stringstream.hpp>

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    class simple_component_base : public detail::simple_component_tag
    {
    private:
        typedef typename boost::mpl::if_<
                boost::is_same<Component, detail::this_type>,
                simple_component_base, Component
            >::type this_component_type;

    public:
        typedef this_component_type wrapped_type;
        typedef this_component_type base_type_holder;

        /// \brief Construct an empty simple_component
        simple_component_base()
        {}

        /// \brief Destruct a simple_component
        ~simple_component_base()
        {
            applier::unbind_gid(gid_);
        }

        /// \brief finalize() will be called just before the instance gets
        ///        destructed
        ///
        /// \param self [in] The PX \a thread used to execute this function.
        /// \param appl [in] The applier to be used for finalization of the
        ///             component instance.
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

        /// \brief Create a new GID (if called for the first time), assign this
        ///        GID to this instance of a component and register this gid
        ///        with the AGAS service
        ///
        /// \returns      The global id (GID) assigned to this instance of a
        ///               component
        naming::gid_type get_base_gid() const
        {
            if (!gid_)
            {
                applier::applier& appl = hpx::applier::get_applier();
                naming::address addr(appl.here(),
                    components::get_component_type<wrapped_type>(),
                    boost::uint64_t(static_cast<this_component_type const*>(this)));
                gid_ = hpx::detail::get_next_id();
                if (!applier::bind_gid(gid_, addr))
                {
                    hpx::util::osstream strm;
                    strm << gid_;

                    gid_ = naming::invalid_gid;   // invalidate GID

                    HPX_THROW_EXCEPTION(duplicate_component_address,
                        "simple_component_base<Component>::get_base_gid",
                        hpx::util::osstream_get_string(strm));
                }
            }
            return gid_;
        }

        naming::id_type get_gid() const
        {
            return naming::id_type(get_base_gid(), naming::id_type::unmanaged);
        }

        /// \brief  The function \a get_factory_properties is used to
        ///         determine, whether instances of the derived component can
        ///         be created in blocks (i.e. more than one instance at once).
        ///         This function is used by the \a distributing_factory to
        ///         determine a correct allocation strategy
        static factory_property get_factory_properties()
        {
            // components derived from this template have to be allocated one
            // at a time
            return factory_none;
        }

    private:
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
                return Component::create(count);
            }
            static void free(void* p, std::size_t count)
            {
                Component::destroy(reinterpret_cast<Component *>(p), count);
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
        typedef detail::simple_heap_factory<component_type> heap_type;

        /// \brief  The function \a create is used for allocation and
        ///         initialization of instances of the derived components.
#ifdef NDEBUG
        static component_type* create(std::size_t)
#else
        static component_type* create(std::size_t count)
#endif
        {
            // simple components can be created individually only
            BOOST_ASSERT(1 == count);
            return static_cast<component_type* >(new Component());
        }

        /// \brief  The function \a create is used for allocation and
        //          initialization of a single instance.
#define HPX_SIMPLE_COMPONENT_CREATE_ONE(Z, N, _)                              \
        template <BOOST_PP_ENUM_PARAMS(N, typename T)>                        \
        static component_type*                                                \
        create_one(BOOST_PP_ENUM_BINARY_PARAMS(N, T, const& t))               \
        {                                                                     \
            return static_cast<component_type* >(new Component(BOOST_PP_ENUM_PARAMS(N, t)));                 \
        }                                                                     \
    /**/

        BOOST_PP_REPEAT_FROM_TO(1, HPX_COMPONENT_CREATE_ARGUMENT_LIMIT,
            HPX_SIMPLE_COMPONENT_CREATE_ONE, _)

#undef HPX_SIMPLE_COMPONENT_CREATE_ONE

        /// \brief  The function \a destroy is used for destruction and
        ///         de-allocation of instances of the derived components.
#ifdef NDEBUG
        static void destroy(Component* p, std::size_t /*count*/ = 1)
#else
        static void destroy(Component* p, std::size_t count = 1)
#endif
        {
            // simple components can be deleted individually only
            BOOST_ASSERT(1 == count);
            p->finalize();
            delete p;
        }
    };

}}

#endif
