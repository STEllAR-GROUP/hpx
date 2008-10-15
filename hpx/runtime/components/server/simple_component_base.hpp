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
        /// \brief Construct an empty managed_component_base
        simple_component_base() 
          : appl_(NULL)
        {}

        /// \brief Destruct an empty managed_component_base
        ~simple_component_base()
        {
            if (gid_ && appl_)
                appl_->get_dgas_client().unbind(gid_);
        }

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        HPX_COMPONENT_EXPORT static component_type get_component_type();
        HPX_COMPONENT_EXPORT static void set_component_type(component_type);

        /// \brief Create a new GID (if called for the first time), assign this 
        ///        GID to this instance of a component and register this gid 
        ///        with the DGAS service
        ///
        /// \param appl   The applier instance to be used for accessing the 
        ///               DGAS service.
        ///
        /// \returns      The global id (GID)  assigned to this instance of a 
        ///               component
        naming::id_type const&
        get_gid(applier::applier& appl) const
        {
            if (!gid_ || !appl_) 
            {
                naming::address addr(appl.here(), Component::get_component_type(), 
                    boost::uint64_t(static_cast<Component const*>(this)));
                appl_ = &appl;
                gid_ = appl.get_parcel_handler().get_next_id();
                if (!appl.get_dgas_client().bind(gid_, addr))
                {
                    HPX_OSSTREAM strm;
                    strm << gid_;

                    appl_ = NULL;
                    gid_ = naming::id_type();   // invalidate GID

                    HPX_THROW_EXCEPTION(duplicate_component_address,
                        HPX_OSSTREAM_GETSTRING(strm));
                }
            }
            return gid_;
        }

        /// \brief  The function \a create is used for allocation and 
        //          initialization of components.
        static simple_component_base* 
        create(applier::applier& appl, std::size_t count)
        {
            // simple components can be created individually only
            BOOST_ASSERT(1 == count);
            return new Component(appl);
        }

        static void destroy(Component* p, std::size_t count)
        {
            // simple components can be deleted individually only
            BOOST_ASSERT(1 == count);
            delete p;
        }

    private:
        mutable naming::id_type gid_;
        mutable applier::applier const* appl_;
    };

}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_SIMPLE_COMPONENT(component)                              \
    namespace hpx { namespace components {                                    \
        component_type                                                        \
        simple_component_base<component>::value = component_invalid;          \
                                                                              \
        component_type                                                        \
        simple_component_base<component>::get_component_type()                \
        { return value; }                                                     \
                                                                              \
        void simple_component_base<component>::                               \
            set_component_type(component_type type)                           \
        { value = type; }                                                     \
                                                                              \
        template<> HPX_ALWAYS_EXPORT                                          \
        component_type get_component_type<component>()                        \
        { return component::get_component_type(); }                           \
    }}                                                                        \
    /**/

#endif
