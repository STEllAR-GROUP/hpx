//  Copyright (c) 2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <list>

namespace hpx { namespace serialization
{
    void output_archive::add_gid(
        naming::gid_type const & gid,
        naming::gid_type const & split_gid)
    {
        HPX_ASSERT(is_future_awaiting());
        buffer_->add_gid(gid, split_gid);
    }

    naming::gid_type output_archive::get_new_gid(naming::gid_type const & gid)
    {
        if(!new_gids_) return naming::gid_type();

        new_gids_map::iterator it = new_gids_->find(gid);

        std::list<naming::gid_type>& gids = it->second;

        HPX_ASSERT(it != new_gids_->end() && !gids.empty());

        naming::gid_type new_gid = gids.front();
        gids.pop_front();

        return new_gid;
    }

}}
