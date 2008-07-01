//  Copyright (c) 2007 Alexandre (aka Alex) TABBAL
//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/asio.hpp>

#include <hpx/runtime/parcelset/server/parcelport_queue.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace server
{
    void parcelport_queue::add_parcel(
        boost::shared_ptr<std::vector<char> > const& parcel_data)
    {
// we don't queue the incoming parcels in the parcelport for now
//         {
//             mutex_type::scoped_lock l(mtx_);
//             parcel_queue_.push_back(p);
//         }
        notify_(parcel_port_, parcel_data);      // do some work (notify event handlers)
    }

//     bool parcelport_queue::get_parcel(parcel& p)
//     {
//         mutex_type::scoped_lock l(mtx_);
// 
//         if (parcel_queue_.empty()) 
//             return false;
// 
//         p = parcel_queue_.front();
//         parcel_queue_.pop_front();
//         return true;
//     }

///////////////////////////////////////////////////////////////////////////////
}}}
