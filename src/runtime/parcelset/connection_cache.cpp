//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Regex library
//  Copyright (c) 2004 John Maddock
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/connection_cache.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>

namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////////
    connection_cache::connection_type 
    connection_cache::get (key_type const& l)
    {
        boost::mutex::scoped_lock lock(mtx_);

//         std::cerr << "get: " << l.address().to_string() << ":" << l.port();

        // see if the object is already in the cache:
        map_iterator mpos = index_.find(l);
        if (mpos != index_.end()) {
        // We have a cached item, return it
            connection_type result(mpos->second->first);

//             std::cerr << " (found)" << std::flush << std::endl;

            cont_.erase(mpos->second);
            index_.erase(mpos);
            return result;
        }
        
//         std::cerr << " (not found)" << std::flush << std::endl;
        
        // if we get here then the item is not in the cache
        return connection_type();
    }
    
    ///////////////////////////////////////////////////////////////////////////
    void
    connection_cache::add (key_type const& l, connection_type conn)
    {
        boost::mutex::scoped_lock lock(mtx_);

//         std::cerr << "add: "
//                   << l.address().to_string() << ":" << l.port()
//                   << std::flush << std::endl;

        // Add it to the list, and index it
        cont_.push_back(value_type(conn, NULL));
        index_.insert(std::make_pair(l, --(cont_.end())));
        cont_.back().second = &(index_.find(l)->first);
        
        map_size_type s = index_.size();
        if (s > max_cache_size_) {
        // We have too many items in the list, so we need to start popping them 
        // off the back of the list
            list_iterator pos = cont_.begin();
            list_iterator last = cont_.end();
            while (pos != last && s > max_cache_size_) {
                // now remove the items from our containers
                list_iterator condemmed(pos);
                ++pos;

                index_.erase(*(condemmed->second));
                cont_.erase(condemmed); 
                --s;
            }
        }
    }

}}
