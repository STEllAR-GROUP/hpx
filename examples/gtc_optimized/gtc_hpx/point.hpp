//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_8E6BF9EF_1C6E_4EA4_B9C0_764E3CD0335A)
#define HPX_8E6BF9EF_1C6E_4EA4_B9C0_764E3CD0335A

#include <hpx/runtime/components/client_base.hpp>

#include "stubs/point.hpp"

namespace gtc
{
    ///////////////////////////////////////////////////////////////////////////
    /// The client side representation of a \a gtc::server::point components.
    class point : public hpx::components::client_base<point, stubs::point>
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

        hpx::lcos::future<void> setup_async(std::size_t numberpe,std::size_t mype,
                              std::vector<hpx::naming::id_type> const& point_components)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::setup_async(gid_,numberpe,mype,point_components);
        }

        void setup(std::size_t numberpe,std::size_t mype,
                   std::vector<hpx::naming::id_type> const& point_components)
        {
            BOOST_ASSERT(gid_);
            this->base_type::setup(gid_,numberpe,mype,point_components);
        }

        hpx::lcos::future<void> chargei_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::chargei_async(gid_);
        }

        void chargei()
        {
            BOOST_ASSERT(gid_);
            this->base_type::chargei(gid_);
        }
    };
}

#endif

