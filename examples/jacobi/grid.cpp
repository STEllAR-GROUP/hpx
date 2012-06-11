
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "grid.hpp"
#include "row.hpp"

#include <hpx/components/remote_object/object.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>
#include <hpx/components/remote_object/distributed_new.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/include/iostreams.hpp>

namespace jacobi
{
    grid::grid(std::size_t nx, std::size_t ny, double value)
        //: rows(ny)
    {
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        // make get the type of the solver component
        hpx::components::component_type
            type = hpx::components::get_component_type<
                server::row
            >();
 
        hpx::components::distributing_factory::result_type rows_allocated =
            factory.create_components(type, ny);

        rows.reserve(ny);
        std::vector<hpx::lcos::future<void> > init_futures;
        init_futures.reserve(ny);
        BOOST_FOREACH(hpx::naming::id_type id, hpx::util::locality_results(rows_allocated))
        {
            row r; r.id = id;
            init_futures.push_back(r.init(nx, value));
            rows.push_back(r);
        }

        hpx::lcos::wait(init_futures);
        
        /*
        // get list of locality prefixes
        std::vector<hpx::naming::id_type> localities = hpx::find_all_localities(type);

        typedef
            hpx::components::server::create_one_component_action2<
                hpx::components::managed_component<server::row>
              , std::size_t
              , double
            >::type
            create_component_action;

        std::size_t objs_per_loc = ny / localities.size();
        std::size_t lid = 0;
        std::size_t count = 0;

        for(std::size_t y = 0; y < ny; ++y)
        {
            rows[y].id
              = hpx::naming::id_type(
                    hpx::async<create_component_action>(
                        localities[lid]
                      , type
                      , nx
                      , value
                    ).get();
                    hpx::naming::id_type::managed
                )

            if(count == objs_per_loc)
            {
                count = 0;
                lid = (lid + 1) % localities.size();
            }
            else
            {
                ++count;
            }
        }
        */
    }
}
