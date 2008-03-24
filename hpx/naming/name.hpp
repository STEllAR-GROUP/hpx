//  Copyright (c) 2007 Richard D. Guidry Jr.
//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_NAME_MAR_24_2008_0942AM)
#define HPX_NAMING_NAME_MAR_24_2008_0942AM

#include <boost/cstdint.hpp>
#include <boost/serialization/serialization.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    /// Global identifier for components across the PX system
    struct id_type
    {
        id_type (boost::uint64_t id = 0) 
          : id_(id) 
        {}
        
        operator boost::uint64_t() const 
        {
            return id_;
        }
        id_type operator= (boost::uint64_t id)
        {
            id_ = id;
            return *this;
        }
        
        boost::uint64_t id_;    // Type that we use for global IDs

    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version) 
        {
            ar & id_;
        }
    };

///////////////////////////////////////////////////////////////////////////////
}}

#endif 
