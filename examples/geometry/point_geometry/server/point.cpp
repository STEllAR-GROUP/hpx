//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2007-2011 Matthew Anderson
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

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/boost_tuple.hpp>
#include <boost/bind.hpp>

#include "../serialize_geometry.hpp"
#include "../stubs/point.hpp"
#include "./point.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace geometry { namespace server 
{
        /// search for contact
        bool point::search(std::vector<hpx::naming::id_type> const& search_objects) const
        {
            typedef std::vector<lcos::future_value<polygon_type> > lazy_results_type;

            lazy_results_type lazy_results;
            BOOST_FOREACH(naming::id_type gid, search_objects)
            {
              lazy_results.push_back( stubs::point::get_poly_async( gid ) );
            }

            // will return the number of invoked futures
            wait(lazy_results, boost::bind(&point::search_callback, this, _1));

            return false;
        }

        bool point::search_callback(polygon_type const& poly) const {
              
          // return type says continue or not
          // usually return true
          return true;
        }
}}}

