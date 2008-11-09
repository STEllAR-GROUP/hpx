//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SIMPLE_COMPONENT_BASE_JUL_18_2008_0948PM)
#define HPX_COMPONENTS_SIMPLE_COMPONENT_BASE_JUL_18_2008_0948PM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>

namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class simple_component_base simple_component_base.hpp hpx/runtime/components/server/simple_component_base.hpp
    ///
    template <typename Component>
    class simple_component_base : public detail::simple_component_tag
    {
    private:
        static component_type value;

    public:
        typedef Component wrapped_type;

        /// \brief Construct an empty simple_component
        simple_component_base(threads::thread_self& self, applier::applier& appl) 
          : appl_(appl)
        {}

        /// \brief Destruct a simple_component
        ~simple_component_base()
        {
            if (gid_)
                appl_.get_agas_client().unbind(gid_);
        }

        /// \brief finalize() will be called just before the instance gets 
        ///        destructed
        ///
        /// \param self [in] The PX \a thread used to execute this function.
        /// \param appl [in] The applier to be used for finalization of the 
        ///             component instance. 
        void finalize(threads::thread_self& self, applier::applier& appl) {}

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        static component_type get_component_type()
        {
            return value;
        }
        static void set_component_type(component_type type)
        {
            value = type;
        }

        /// \brief Create a new GID (if called for the first time), assign this 
        ///        GID to this instance of a component and register this gid 
        ///        with the AGAS service
        ///
        /// \param appl   The applier instance to be used for accessing the 
        ///               AGAS service.
        ///
        /// \returns      The global id (GID)  assigned to this instance of a 
        ///               component
        naming::id_type const&
        get_gid(applier::applier& appl) const
        {
            if (!gid_) 
            {
                naming::address addr(appl.here(), Component::get_component_type(), 
                    boost::uint64_t(static_cast<Component const*>(this)));
                gid_ = appl_.get_parcel_handler().get_next_id();
                if (!appl_.get_agas_client().bind(gid_, addr))
                {
                    HPX_OSSTREAM strm;
                    strm << gid_;

                    gid_ = naming::id_type();   // invalidate GID

                    HPX_THROW_EXCEPTION(duplicate_component_address,
                        HPX_OSSTREAM_GETSTRING(strm));
                }
            }
            return gid_;
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
        mutable naming::id_type gid_;
        applier::applier& appl_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    component_type simple_component_base<Component>::value = component_invalid;


    ///////////////////////////////////////////////////////////////////////////
    /// \class simple_component simple_component.hpp hpx/runtime/components/server/simple_component_base.hpp
    ///
    template <typename Component>
    class simple_component : public Component
    {
    public:
        simple_component(applier::applier& appl)
          : Component(appl)
        {}

        /// \brief  The function \a create is used for allocation and 
        ///         initialization of instances of the derived components.
        static Component* 
        create(threads::thread_self& self, applier::applier& appl, 
            std::size_t count)
        {
            // simple components can be created individually only
            BOOST_ASSERT(1 == count);
            return new Component(self, appl);
        }

        /// \brief  The function \a destroy is used for destruction and 
        ///         de-allocation of instances of the derived components.
        static void destroy(threads::thread_self& self, applier::applier& appl, 
            Component* p, std::size_t count = 1)
        {
            // simple components can be deleted individually only
            BOOST_ASSERT(1 == count);
            p->finalize(self, appl);
            delete p;
        }

    };

}}

#endif
