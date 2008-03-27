//  Copyright (c) 2007 Alexandre (aka Alex) TABBAL
//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/asio.hpp>

#include <hpx/parcelset/server/parcel_queue.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace server
{
    void parcel_queue::add_parcel(parcel const& p)
    {
        {
            mutex_type::scoped_lock l(mtx_);
            parcel_queue_.push_back(p);
        }
        notify_(parcel_port_);      // do some work (notify event handlers)
    }

    bool parcel_queue::get_parcel(parcel& p)
    {
        mutex_type::scoped_lock l(mtx_);

        if (parcel_queue_.empty()) 
            return false;

        p = parcel_queue_.front();
        parcel_queue_.pop_front();
        return true;
    }

    bool parcel_queue::get_parcel(components::component_type c, parcel& p)
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

    bool parcel_queue::get_parcel(parcel_id tag, parcel& p)
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

    bool parcel_queue::get_parcel_from(naming::id_type src, parcel& p)
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

    bool parcel_queue::get_parcel_for(naming::id_type dest, parcel& p)
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
