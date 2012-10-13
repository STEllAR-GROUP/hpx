//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_68F602E3_C235_4660_AEAC_D5BD7AEC4806)
#define HPX_68F602E3_C235_4660_AEAC_D5BD7AEC4806

#include <hpx/include/client.hpp>

#include "stubs/point.hpp"

namespace ad
{
    ///////////////////////////////////////////////////////////////////////////
    /// The client side representation of a \a ad::server::point components.
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
        /// \a ad::server::point instance with the given GID.
        point(hpx::naming::id_type const& gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        hpx::lcos::future<void> init_async(std::size_t scale,std::size_t np)
        {
            return this->base_type::init_async(get_gid(),scale,np);
        }

        void init(std::size_t scale,std::size_t np)
        {
            this->base_type::init(get_gid(),scale,np);
        }

        hpx::lcos::future<void> remove_item_async(std::size_t scale,std::size_t np)
        {
            return this->base_type::remove_item_async(get_gid(),scale,np);
        }

        void remove_item(std::size_t scale,std::size_t np)
        {
            this->base_type::remove_item(get_gid(),scale,np);
        }

        hpx::lcos::future<void> compute_async(
                  std::vector<hpx::naming::id_type> const& point_components)
        {
            return this->base_type::compute_async(get_gid(),point_components);
        }

        void compute(
                   std::vector<hpx::naming::id_type> const& point_components)
        {
            this->base_type::compute(get_gid(),point_components);
        }

        hpx::lcos::future<std::size_t> get_item_async()
        {
            return this->base_type::get_item_async(get_gid());
        }

        std::size_t get_item()
        {
            return this->base_type::get_item(get_gid());
        }
    };
}

#endif

