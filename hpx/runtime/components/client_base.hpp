//  Copyright (c) 2007-2010 Hartmut Kaiser
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
        client_base()
          : gid_(naming::invalid_id)
        {}

        client_base(naming::id_type gid)
          : gid_(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        /// Create a new instance of an distributing_factory on the locality as 
        /// given by the parameter \a targetgid
        Derived& create(naming::id_type const& targetgid, component_type type,
            std::size_t count = 1)
        {
            free();
            gid_ = stub_type::create(targetgid, type, count);
            return static_cast<Derived&>(*this);
        }

        Derived& create(naming::id_type const& targetgid, std::size_t count = 1)
        {
            free();
            gid_ = stub_type::create(targetgid, count);
            return static_cast<Derived&>(*this);
        }

        void free(component_type type)
        {
            gid_ = naming::invalid_id;
        }

        void free()
        {
            gid_ = naming::invalid_id;
        }

        ///////////////////////////////////////////////////////////////////////
        naming::id_type const& get_gid() const
        {
            return gid_;
        }

        naming::id_type detach() 
        {
            naming::id_type g;
            std::swap(gid_, g);
            return g;
        }

    protected:
        naming::id_type gid_;
    };

}}

#endif

