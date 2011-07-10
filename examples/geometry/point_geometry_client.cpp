//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2007-2011 Matthew Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "point_geometry/point.hpp"
#include <boost/geometry/geometries/polygon.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

namespace hpx { namespace geometry
{
    typedef boost::geometry::model::polygon<hpx::geometry::point> polygon_2d; 
}}

inline void
init(hpx::components::server::distributing_factory::iterator_range_type r,
    std::vector<hpx::geometry::point>& accu)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        accu.push_back(hpx::geometry::point(id));
    }
}


///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map &vm)
{
    {

        std::size_t array_size = 8;

        namespace bg = boost::geometry;

        // create a distributing factory locally
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        hpx::components::component_type block_type =
            hpx::components::get_component_type<
                hpx::geometry::point::server_component_type>();

        hpx::components::distributing_factory::result_type blocks =
            factory.create_components(block_type, array_size);

        std::vector<hpx::geometry::point> accu;

        init(locality_results(blocks), accu);

        std::vector<hpx::lcos::future_value<void> > initial_phase;
        initial_phase.push_back(accu[0].init_async(0,0));
        initial_phase.push_back(accu[1].init_async(0,1));
        initial_phase.push_back(accu[2].init_async(1,1));
        initial_phase.push_back(accu[3].init_async(1,0));

        initial_phase.push_back(accu[4].init_async(0.5,0.5));
        initial_phase.push_back(accu[5].init_async(0,1.5));
        initial_phase.push_back(accu[6].init_async(0.5,2));
        initial_phase.push_back(accu[7].init_async(1,1.5));

        hpx::components::wait(initial_phase);

        hpx::geometry::polygon_2d p;
        p.outer().push_back(accu[0]);
        p.outer().push_back(accu[1]);
        p.outer().push_back(accu[2]);
        p.outer().push_back(accu[3]);
        p.outer().push_back(accu[0]);
        bg::correct(p);

        hpx::geometry::polygon_2d q;
        q.outer().push_back(accu[4]);
        q.outer().push_back(accu[5]);
        q.outer().push_back(accu[6]);
        q.outer().push_back(accu[7]);
        q.outer().push_back(accu[4]);
        bg::correct(q);

        hpx::geometry::plain_polygon_type plain_p;

        // should be: boost::geometry::assign(plain_p, p));, but boost::geometry
        // has a bug preventing this from compiling
        boost::geometry::assign(plain_p.outer(), p.outer());

        std::vector<hpx::lcos::future_value<bool> > search_phase;
        search_phase.push_back(accu[4].search_async(plain_p));

        hpx::components::wait(search_phase);

        hpx::geometry::point pt5(hpx::find_here(), 0.5, 0.5);
        bool inside = bg::within(pt5, p);
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
