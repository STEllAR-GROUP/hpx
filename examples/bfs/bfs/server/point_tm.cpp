//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>

#include "../stubs/point_tm.hpp"
#include "../stubs/point.hpp"
#include "point_tm.hpp"

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>

///////////////////////////////////////////////////////////////////////////////
namespace bfs_tm { namespace server
{
    void point::manager(std::size_t level,std::size_t edge,std::vector<std::size_t> const& neighbors)
    {
      std::vector<hpx::lcos::promise<void> > manager_futures;
      for (std::size_t i=0;i<neighbors.size();i++) {
        std::size_t nghb = index_[ neighbors[i] ];
        hpx::naming::id_type gid = points_components_[ nghb ];
        manager_futures.push_back( bfs::stubs::point::traverse_async(gid,level,edge,nghb) );
      }
      hpx::lcos::wait(manager_futures);
    }

    void point::init(std::size_t objectid,
                     boost::numeric::ublas::mapped_vector<std::size_t> const& index,
                     std::vector<hpx::naming::id_type> const& points_components)
    {
      idx_ = objectid;
      index_ = index;
      points_components_ = points_components;
    }
}}

