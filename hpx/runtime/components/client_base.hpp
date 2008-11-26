//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CLIENT_BASE_OCT_31_2008_0424PM)
#define HPX_COMPONENTS_CLIENT_BASE_OCT_31_2008_0424PM

#include <hpx/hpx_fwd.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename Stub>
    class client_base : public Stub
    {
    protected:
        typedef Stub stub_type;

    public:
        client_base(applier::applier& appl, naming::id_type gid,
                bool freeonexit = false)
          : stub_type(appl), gid_(gid), freeonexit_(freeonexit)
        {
            BOOST_ASSERT(gid_);
        }

        ~client_base() 
        {
            if (freeonexit_ && gid_)
                this->stub_type::free(gid_);
        }

        ///////////////////////////////////////////////////////////////////////
        /// Create a new instance of an distributing_factory on the locality as 
        /// given by the parameter \a targetgid
        static Derived 
        create(applier::applier& appl, 
            naming::id_type const& targetgid, component_type type,
            bool freeonexit = false)
        {
            return Derived(appl, stub_type::create(appl, targetgid, type), freeonexit);
        }

        static Derived 
        create(applier::applier& appl, 
            naming::id_type const& targetgid, bool freeonexit = false)
        {
            return Derived(appl, stub_type::create(appl, targetgid), freeonexit);
        }

        void free(component_type type)
        {
            if (gid_) {
                this->stub_type::free(type, gid_);
                gid_ = naming::invalid_id;
            }
        }

        void free()
        {
            if (gid_) {
                this->stub_type::free(gid_);
                gid_ = naming::invalid_id;
            }
        }

        ///////////////////////////////////////////////////////////////////////
        naming::id_type const& get_gid() const
        {
            return gid_;
        }

    protected:
        naming::id_type gid_;
        bool freeonexit_;
    };

}}

#endif

