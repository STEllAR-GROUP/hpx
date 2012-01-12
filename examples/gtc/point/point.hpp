//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_AA0F78C3_B11C_4173_8FA8_C1A6073FB9BA)
#define HPX_AA0F78C3_B11C_4173_8FA8_C1A6073FB9BA

#include <hpx/runtime/components/client_base.hpp>

#include "stubs/point.hpp"

namespace gtc
{
    ///////////////////////////////////////////////////////////////////////////
    /// The client side representation of a \a gtc::server::point
    /// components.
    class point
      : public hpx::components::client_base<point, stubs::point>
    {
        typedef hpx::components::client_base<point, stubs::point>
            base_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        point()
        {}

        /// Create a client side representation for the existing
        /// \a gtc::server::point instance with the given GID.
        point(hpx::naming::id_type const& gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Initialize the \a gtc::server::point instance with the
        /// given point file. 
        hpx::lcos::promise<void> init_async(std::size_t objectid,parameter const& par)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init_async(gid_,objectid,par);
        }

        /// Initialize the \a gtc::server::point instance with the
        /// given point file.  
        void init(std::size_t objectid,parameter const& par)
        {
            BOOST_ASSERT(gid_);
            this->base_type::init_async(gid_,objectid,par);
        }

        hpx::lcos::promise<void> load_async(std::size_t objectid,parameter const& par)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::load_async(gid_,objectid,par);
        }

        void load(std::size_t objectid,parameter const& par)
        {
            BOOST_ASSERT(gid_);
            this->base_type::load_async(gid_,objectid,par);
        }

        hpx::lcos::promise<void> chargei_async(std::size_t istep, 
            std::vector<hpx::naming::id_type> const& point_components,
            parameter const& par)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::chargei_async(gid_,istep,point_components,par);
        }

        void chargei(std::size_t istep,
                    std::vector<hpx::naming::id_type> const& point_components,
                    parameter const& par)
        {
            BOOST_ASSERT(gid_);
            this->base_type::chargei(gid_,istep,point_components,par);
        }


    };
}

#endif

