//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERIALIZE_GEOMETRY_JUL_10_2011_0432PM)
#define HPX_SERIALIZE_GEOMETRY_JUL_10_2011_0432PM

#include <hpx/hpx_fwd.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

namespace hpx { namespace geometry
{
    ///////////////////////////////////////////////////////////////////////////
    typedef boost::geometry::model::d2::point_xy<double> plain_point_type;
    typedef boost::geometry::model::polygon<plain_point_type> plain_polygon_type;
}}

namespace boost { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    // declarations only
    template <typename Archive>
    HPX_COMPONENT_EXPORT void
    save(Archive& ar, hpx::geometry::plain_point_type const& p, unsigned int const);

    template <typename Archive>
    HPX_COMPONENT_EXPORT void
    load(Archive& ar, hpx::geometry::plain_point_type& p, unsigned int const);

    ///////////////////////////////////////////////////////////////////////////
    // declarations only
    template <typename Archive>
    HPX_COMPONENT_EXPORT void
    save(Archive& ar, hpx::geometry::plain_polygon_type const& p, unsigned int const);

    template <typename Archive>
    HPX_COMPONENT_EXPORT void
    load(Archive& ar, hpx::geometry::plain_polygon_type& p, unsigned int const);
}}

// load and save are separate functions
BOOST_SERIALIZATION_SPLIT_FREE(hpx::geometry::plain_point_type);
BOOST_SERIALIZATION_SPLIT_FREE(hpx::geometry::plain_polygon_type);

#endif

