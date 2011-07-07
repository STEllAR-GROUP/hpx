//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "point_geometry/point.hpp"
#include <boost/geometry/geometries/polygon.hpp>

namespace hpx { namespace geometry
{
    typedef boost::geometry::model::polygon<hpx::geometry::point> polygon_2d; 
}}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map &vm)
{
    {
        namespace bg = boost::geometry;

        hpx::geometry::point pt1(hpx::find_here(), 1, 1);
        hpx::geometry::point pt2(hpx::find_here(), 2, 2);
        hpx::geometry::point pt3(hpx::find_here(), 1, 2);

//         bg::assign_values(pt1, 1, 1);
//         bg::assign_values(pt2, 2, 2);

        double d = bg::distance(pt1, pt2);

        std::cout << "Distance: " << d << std::endl;

        hpx::geometry::polygon_2d p;
        p.outer().push_back(pt1);
        p.outer().push_back(pt2);
        p.outer().push_back(pt3);
        p.outer().push_back(pt1);
        bg::correct(p);

        hpx::geometry::point pt4(hpx::find_here(), 1.5, 1.25);
        bool inside = bg::within(pt4, p);

        std::cout << "Point is " << (inside ? "inside" : "outside") << std::endl;
    }

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init("point_geometry_client", argc, argv); // Initialize and run HPX
}
