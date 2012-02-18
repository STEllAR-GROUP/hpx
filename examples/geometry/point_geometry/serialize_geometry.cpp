//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include "serialize_geometry.hpp"

namespace boost { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////////
    // implement the serialization functions
    template <typename Archive>
    void save(Archive& ar, hpx::geometry::plain_point_type const& pt, unsigned int const)
    {
        double x = pt.x(), y = pt.y();
        ar & x & y;
    }

    template <typename Archive>
    void load(Archive& ar, hpx::geometry::plain_point_type& pt, unsigned int const)
    {
        double x = 0, y = 0;
        ar & x & y;

        pt.x(x);
        pt.y(y);
    }

    template <typename Archive>
    void save(Archive& ar, hpx::geometry::plain_polygon_type const& p, unsigned int const)
    {
        std::vector<hpx::geometry::plain_point_type> const& ring = p.outer();
        ar & ring;
        // no need to serialize the inner rings for now
    }

    template <typename Archive>
    void load(Archive& ar, hpx::geometry::plain_polygon_type& p, unsigned int const)
    {
    //         ar & p.outer();
        // no need to serialize the inner rings for now
    }

    ///////////////////////////////////////////////////////////////////////////
    // explicit instantiation for the correct archive types
//     template HPX_COMPONENT_EXPORT void
//     save(hpx::util::portable_binary_oarchive&,
//         hpx::geometry::plain_point_type const&, unsigned int const);
    template HPX_COMPONENT_EXPORT void
    load(hpx::util::portable_binary_iarchive&,
        hpx::geometry::plain_point_type&, unsigned int const);

    template HPX_COMPONENT_EXPORT void
    save(hpx::util::portable_binary_oarchive&,
        hpx::geometry::plain_polygon_type const&, unsigned int const);
    template HPX_COMPONENT_EXPORT void
    load(hpx::util::portable_binary_iarchive&,
        hpx::geometry::plain_polygon_type&, unsigned int const);
}}


