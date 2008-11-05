//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/lcos/base_lco.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>

///////////////////////////////////////////////////////////////////////////////
// enable serialization of continuations through shared_ptr's
BOOST_CLASS_EXPORT(hpx::actions::continuation);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    void continuation::trigger_all(applier::applier& app)
    {
        std::vector<naming::id_type>::iterator end = gids_.end();
        for (std::vector<naming::id_type>::iterator it = gids_.begin();
             it != end; ++it)
        {
            if (!app.apply<lcos::base_lco::set_event_action>(*it))
                break;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void continuation::trigger_error(applier::applier& app, 
        hpx::exception const& e)
    {
        std::vector<naming::id_type>::iterator end = gids_.end();
        for (std::vector<naming::id_type>::iterator it = gids_.begin();
             it != end; ++it)
        {
            if (!app.apply<lcos::base_lco::set_error_action>(
                    *it, e.get_error(), std::string(e.what())))
            {
                break;
            }
        }
    }

}}

