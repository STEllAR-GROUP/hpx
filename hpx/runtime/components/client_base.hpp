//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CLIENT_BASE_OCT_31_2008_0424PM)
#define HPX_COMPONENTS_CLIENT_BASE_OCT_31_2008_0424PM

#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/component_type.hpp>

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

        client_base(naming::id_type const& gid)
          : gid_(gid)
        {}

        client_base& operator=(naming::id_type const& gid)
        {
            if (gid_ != gid)
                gid_ = gid;
            return *this;
        }

        ///////////////////////////////////////////////////////////////////////
        /// Create a new instance of an object on the locality as
        /// given by the parameter \a targetgid
        Derived& create(naming::id_type const& targetgid, component_type type,
            std::size_t count = 1)
        {
            free();
            gid_ = stub_type::create_sync(targetgid, type, count);
            return static_cast<Derived&>(*this);
        }

        Derived& create(naming::id_type const& targetgid, std::size_t count = 1)
        {
            free();
            gid_ = stub_type::create_sync(targetgid, count);
            return static_cast<Derived&>(*this);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Arg0>
        Derived& create_one(naming::id_type const& gid, component_type type,
            BOOST_FWD_REF(Arg0) arg0)
        {
            free();
            gid_ = stub_type::create_one_sync(gid, type, boost::forward<Arg0>(arg0));
            return static_cast<Derived&>(*this);
        }

        template <typename Arg0>
        Derived& create_one(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0)
        {
            free();
            gid_ = stub_type::create_one_sync(gid, boost::forward<Arg0>(arg0));
            return static_cast<Derived&>(*this);
        }

        ///////////////////////////////////////////////////////////////////////
        void free(component_type)
        {
            stub_type::free(gid_);
        }

        void free()
        {
            stub_type::free(gid_);
        }

        ///////////////////////////////////////////////////////////////////////
        naming::id_type const& get_gid() const
        {
            return gid_;
        }

        naming::gid_type const& get_raw_gid() const
        {
            return gid_.get_gid();
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

