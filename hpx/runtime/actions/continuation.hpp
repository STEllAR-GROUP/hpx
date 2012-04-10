//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_CONTINUATION_JUN_13_2008_1031AM)
#define HPX_RUNTIME_ACTIONS_CONTINUATION_JUN_13_2008_1031AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/runtime/naming/name.hpp>

#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/move/move.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
// Version of continuation
#define HPX_CONTINUATION_VERSION 0x10

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace applier
    {
        template <typename Action, typename Arg0>
        inline bool
        apply(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0);

        template <typename Action>
        inline bool apply(naming::id_type const& gid);
    }

    namespace actions
    {
        // parcel continuations are simply lists of global ids of LCO's to call
        // set_event on
        class HPX_API_EXPORT continuation
        {
        public:
            continuation()
            {}

            explicit continuation(naming::id_type const& gid)
              : gid_(gid)
            {}

            ///
            void trigger();

            template <typename Arg0>
            void trigger(BOOST_FWD_REF(Arg0) arg0)
            {
                typedef typename
                    lcos::template base_lco_with_value<Arg0>::set_result_action
                action_type;

                LLCO_(info) << "continuation::trigger(" << gid_ << ")";

                applier::apply<action_type>(gid_, boost::forward<Arg0>(arg0));
            }

            ///
            void trigger_error(boost::exception_ptr const& e);
            void trigger_error(BOOST_RV_REF(boost::exception_ptr) e);

            naming::gid_type const& get_raw_gid()
            {
                return gid_.get_gid();
            }

        private:
            // serialization support
            friend class boost::serialization::access;

            template<class Archive>
            void serialize(Archive& ar, const unsigned int /*version*/)
            {
                ar & gid_;
            }

            naming::id_type gid_;
        };
    }
}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the id_type serialization format
BOOST_CLASS_VERSION(hpx::actions::continuation, HPX_CONTINUATION_VERSION)
BOOST_CLASS_TRACKING(hpx::actions::continuation, boost::serialization::track_never)

#include <hpx/config/warnings_suffix.hpp>

#endif
