//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/async_future_wait.hpp>

#include <boost/bind.hpp>
#include <boost/ref.hpp>

#include "../../particle/stubs/particle.hpp"
#include "../stubs/point.hpp"
#include "./point.hpp"

#include <iostream>
#include <fstream>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace geometry { namespace server
{
        int point::search(std::vector<hpx::naming::id_type> const& particle_components)
        {
          // For demonstration, a simple search strategy:
          // is the particle within a certain distance of the gridpoint.  If so, then get it's charge
          typedef std::vector<lcos::promise<double> > lazy_results_type;
          lazy_results_type lazy_results;
          BOOST_FOREACH(naming::id_type gid, particle_components)
          {
            lazy_results.push_back( stubs::particle::distance_async( gid,posx_,posy_,posz_ ) );
          }
          lcos::wait(lazy_results, boost::bind(&point::search_callback, this, _1, _2));

          return 0;
        }

        bool point::search_callback(std::size_t i, double const& distance)
        {
          double neighbor_distance = 0.1;
          if ( distance < neighbor_distance ) {
            // deposit the charge of this particle on the gridpoint
          }
          return true;
        }

}}}

