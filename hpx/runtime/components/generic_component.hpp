//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_GENERIC_COMPONENT_OCT_12_2008_0947PM)
#define HPX_COMPONENTS_GENERIC_COMPONENT_OCT_12_2008_0947PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/stubs/generic_component.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ServerComponent>
    class generic_component 
      : public stubs::generic_component<ServerComponent>
    {
    private:
        typedef stubs::generic_component<ServerComponent> base_type;
        typedef typename base_type::result_type result_type;

    public:
        /// Create a client side representation for the existing 
        /// \a server#generic_component instance with the given global id 
        /// \a gid.
        generic_component(applier::applier& app, naming::id_type const& gid,
                bool freeonexit = false) 
          : base_type(app), gid_(gid), freeonexit_(freeonexit)
        {
            BOOST_ASSERT(gid_);
        }
        ~generic_component()
        {
            if (freeonexit_)
                this->base_type::free(gid_);
        }

        /// Invoke the action exposed by this generic component
        result_type eval(threads::thread_self& self)
        {
            return this->base_type::eval(self, gid_);
        }

        // bring in higher order eval functions
        #include <hpx/runtime/components/generic_component_eval.hpp>

        /// Create a new instance of an generic_component on the locality as 
        /// given by the parameter \a targetgid
        static generic_component 
        create(threads::thread_self& self, applier::applier& appl, 
            naming::id_type const& targetgid, bool freeonexit = false)
        {
            return generic_component(appl, 
                base_type::create(self, appl, targetgid), freeonexit);
        }

        void free()
        {
            base_type::free(gid_);
            gid_ = naming::invalid_id;
        }

        ///////////////////////////////////////////////////////////////////////
        naming::id_type const& get_gid() const
        {
            return gid_;
        }

    private:
        naming::id_type gid_;
        bool freeonexit_;
    };

}}

#endif

