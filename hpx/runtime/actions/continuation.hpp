//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_CONTINUATION_JUN_13_2008_1031AM)
#define HPX_RUNTIME_ACTIONS_CONTINUATION_JUN_13_2008_1031AM

#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/serialization.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/full_address.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
// Version of continuation
#define HPX_CONTINUATION_VERSION 0x10

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    // parcel continuations are simply lists of global ids of LCO's to call 
    // set_event on
    class HPX_EXPORT continuation
    {
    public:
        continuation()
        {}

        explicit continuation(naming::id_type const& gid)
          : gid_(gid)
        {}

        explicit continuation(naming::full_address const& gid)
          : gid_(gid)
        {}

        ///
        void trigger();

        ///
        template <typename Arg0>
        Arg0 const& trigger(Arg0 const& arg0);

        ///
        void trigger_error(hpx::exception const& e);

    private:
        // serialization support    
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & gid_;
        }

        naming::full_address gid_;
    };

}}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the id_type serialization format
BOOST_CLASS_VERSION(hpx::actions::continuation, HPX_CONTINUATION_VERSION)
BOOST_CLASS_TRACKING(hpx::actions::continuation, boost::serialization::track_never)

#include <hpx/config/warnings_suffix.hpp>

#endif
