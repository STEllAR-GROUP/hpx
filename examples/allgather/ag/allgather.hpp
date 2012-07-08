//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_68F602E3_C235_4660_AEAC_D5BD7AEC4806)
#define HPX_68F602E3_C235_4660_AEAC_D5BD7AEC4806

#include <hpx/runtime/components/client_base.hpp>

#include "stubs/allgather.hpp"

namespace ag
{
    ///////////////////////////////////////////////////////////////////////////
    /// The client side representation of a \a ag::server::point components.
    class allgather : public hpx::components::client_base<allgather, stubs::point>
    {
        typedef hpx::components::client_base<allgather, stubs::point>
            base_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        allgather()
        {}

        /// Create a client side representation for the existing
        /// \a ag::server::point instance with the given GID.
        allgather(hpx::naming::id_type const& gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        hpx::lcos::future<void> init_async(std::size_t scale,std::size_t np)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init_async(gid_,scale,np);
        }

        void init(std::size_t scale,std::size_t np)
        {
            BOOST_ASSERT(gid_);
            this->base_type::init(gid_,scale,np);
        }

        hpx::lcos::future<void> compute_async(
                  std::vector<hpx::naming::id_type> const& point_components)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::compute_async(gid_,point_components);
        }

        void compute(
                   std::vector<hpx::naming::id_type> const& point_components)
        {
            BOOST_ASSERT(gid_);
            this->base_type::compute(gid_,point_components);
        }

        hpx::lcos::future<void> print_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::print_async(gid_);
        }

        void print()
        {
            BOOST_ASSERT(gid_);
            this->base_type::print(gid_);
        }

        hpx::lcos::future<double> get_item_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_item_async(gid_);
        }

        double get_item()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_item(gid_);
        }
    };
}

#endif

