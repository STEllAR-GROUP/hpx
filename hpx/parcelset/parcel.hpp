//  Copyright (c) 2007-2008 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2007 Alexandre (aka Alex) TABBAL
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_PARCEL_MAR_26_2008_1051AM)
#define HPX_PARCELSET_PARCEL_MAR_26_2008_1051AM

#include <boost/serialization/version.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/serialization.hpp>

#include <hpx/components/action.hpp>
#include <hpx/naming/name.hpp>
#include <hpx/naming/address.hpp>
#include <hpx/exception.hpp>

///////////////////////////////////////////////////////////////////////////////
//  parcel serialization format version
#define HPX_PARCEL_VERSION 0x20

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    // parcel continuation
    enum continuation
    {
        none = 0
    };

    ///////////////////////////////////////////////////////////////////////////
    typedef boost::uint64_t parcel_id;

    ///////////////////////////////////////////////////////////////////////////
    class parcel
    {
    public:
        parcel() 
          : destination_id_(0), action_(), cont_(none), source_id_(), 
            tag_(0)
        {
        }
        
        parcel(naming::id_type apply_to, components::action_base* act)
          : destination_id_(apply_to), action_(act), 
            cont_(none), source_id_(0), tag_(0)
        {
        }
    
        parcel(naming::id_type apply_to, components::action_base* act, 
                continuation do_after) 
          : destination_id_(apply_to), action_(act), 
            cont_(do_after), source_id_(0), tag_(0)
        {
        }
        
        ~parcel()
        {
        }
    
        // default copy constructor is ok    
        // default assignment operator is ok
    
        parcel_id get_parcel_id() const 
        {
            return tag_;
        }
        naming::id_type get_destination() const
        {
            return destination_id_;
        }
        naming::address const& get_destination_addr() const
        {
            return destination_addr_;
        }
        naming::id_type get_source() const
        {
            return source_id_;
        }
        components::action_type get_action() const 
        {
            return action_;
        }
    
        /// set the parcel id
        void set_parcel_id(parcel_id id) const
        {
            tag_ = id;
        }

        /// set the source locality/component id
        void set_source(naming::id_type source_id) const
        {
            source_id_ = source_id;
        }
        
        /// set the destination address
        void set_destination_addr(naming::address const& addr) const
        {
            destination_addr_ = addr;
        }
        
    private:
        // serialization support    
        friend class boost::serialization::access;
    
        template<class Archive>
        void save(Archive & ar, const unsigned int version) const;

        template<class Archive>
        void load(Archive & ar, const unsigned int version);

        BOOST_SERIALIZATION_SPLIT_MEMBER()

    private:
        naming::id_type destination_id_;
        components::action_type action_;
        continuation cont_;
        
        // may be modified even if seems to be const
        mutable naming::id_type source_id_;
        mutable parcel_id tag_;   
        mutable naming::address destination_addr_;
    };

///////////////////////////////////////////////////////////////////////////////
}}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the parcel serialization format
// this definition needs to be in the global namespace
BOOST_CLASS_VERSION(hpx::parcelset::parcel, HPX_PARCEL_VERSION)

#endif
