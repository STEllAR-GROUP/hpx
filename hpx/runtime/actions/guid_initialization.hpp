//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_CONTINUATION_HPP)
#define HPX_RUNTIME_ACTIONS_CONTINUATION_HPP

#include <hpx/traits/needs_guid_initialization.hpp>
//#include <hpx/util/detail/serialization_registration.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/archive/detail/check.hpp>

namespace hpx { namespace actions
{
    namespace detail
    {
        template <typename Target>
        void guid_initialization(boost::mpl::false_) {}

        template <typename Target>
        void guid_initialization(boost::mpl::true_)
        {
            // force serialization self registration to happen
            using namespace boost::archive::detail::extra_detail;
            init_guid<Target>::g.initialize();
        }

        template <typename Target>
        void guid_initialization()
        {
            guid_initialization<Target>(
                typename traits::needs_guid_initialization<Target>::type());
        }

        ///////////////////////////////////////////////////////////////////////
        // Helper to invoke the registration code for serialization at startup
        template <typename Target>
        struct register_base_helper
        {
            register_base_helper()
            {
                Target::register_base();
            }
        };
    }
}}

#endif

