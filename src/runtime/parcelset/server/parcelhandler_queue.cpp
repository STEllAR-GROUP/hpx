//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/server/parcelhandler_queue.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace server
{
    void parcelhandler_queue::add_parcel(parcel const& p)
    {
        {
            mutex_type::scoped_lock l(mtx_);
            parcel_queue_.push_back(p);
            notify_(parcelhandler_, p.get_destination_addr());      // do some work (notify event handlers)
        }
    }

    bool parcelhandler_queue::get_parcel(parcel& p)
    {
        mutex_type::scoped_lock l(mtx_);

        if (parcel_queue_.empty()) 
            return false;

        p = parcel_queue_.front();
        parcel_queue_.pop_front();
        return true;
    }

    bool parcelhandler_queue::get_parcel(components::component_type c, parcel& p)
    {
        mutex_type::scoped_lock l(mtx_);

        if (parcel_queue_.empty()) 
            return false;   // empty parcel queue

        typedef parcel_list_type::iterator iterator;
        iterator end = parcel_queue_.end();
        for (iterator it = parcel_queue_.begin(); it != end; ++it)
        {
            if ((*it).get_action()->get_component_type() == c) {
                p = *it;
                parcel_queue_.erase(it);
                return true;
            }
        }
        return false;
    }

    bool parcelhandler_queue::get_parcel(parcel_id tag, parcel& p)
    {
        mutex_type::scoped_lock l(mtx_);

        if (parcel_queue_.empty()) 
            return false;   // empty parcel queue

        typedef std::list<parcel>::iterator iterator;
        iterator end = parcel_queue_.end();
        for (iterator it = parcel_queue_.begin(); it != end; ++it)
        {
            if ((*it).get_parcel_id() == tag) {
                p = *it;
                parcel_queue_.erase(it);
                return true;
            }
        }
        return false;
    }

    bool parcelhandler_queue::get_parcel_from(naming::id_type const& src, parcel& p)
    {
        mutex_type::scoped_lock l(mtx_);

        if (parcel_queue_.empty()) 
            return false;   // empty parcel queue

        typedef std::list<parcel>::iterator iterator;
        iterator end = parcel_queue_.end();
        for (iterator it = parcel_queue_.begin(); it != end; ++it)
        {
            if ((*it).get_source() == src) {
                p = *it;
                parcel_queue_.erase(it);
                return true;
            }
        }
        return false;
    }

    bool parcelhandler_queue::get_parcel_for(naming::gid_type const& dest, parcel& p)
    {
        mutex_type::scoped_lock l(mtx_);

        if (parcel_queue_.empty()) 
            return false;   // empty parcel queue

        typedef std::list<parcel>::iterator iterator;
        iterator end = parcel_queue_.end();
        for (iterator it = parcel_queue_.begin(); it != end; ++it)
        {
            if ((*it).get_destination() == dest) {
                p = *it;
                parcel_queue_.erase(it);
                return true;
            }
        }
        return false;
    }

///////////////////////////////////////////////////////////////////////////////
}}}
