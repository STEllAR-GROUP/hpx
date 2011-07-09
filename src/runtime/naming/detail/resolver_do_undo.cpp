//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if HPX_AGAS_VERSION <= 0x10

#include <hpx/runtime/naming/detail/resolver_do_undo.hpp>

#include <boost/bind.hpp>
#include <boost/ref.hpp>

namespace hpx { namespace naming { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    void do_undo_incref::do_it(bulk_resolver_client& resolver, int index, 
        error_code& ec)
    {
        resolver.incref(index, ec);
        if (ec)
            do_it_();
    }

    void do_undo_incref::undo_it()
    {
        undo_it_();
    }

    ///////////////////////////////////////////////////////////////////////////
    void do_undo_resolve::do_it(bulk_resolver_client& resolver, int index, 
        error_code& ec)
    {
        naming::address addr;
        resolver.resolve(index, addr, ec);
        if (ec)
            do_it_(addr);
    }

    ///////////////////////////////////////////////////////////////////////////
    int bulk_resolver_helper::resolve(gid_type const& id, parcelset::parcel& p)
    {
        tasks_.push_back(new do_undo_resolve(
            boost::bind(&parcelset::parcel::set_destination_addr, boost::ref(p), _1)));
          return resolver_.resolve(id);
    }

    void bulk_resolver_helper::add_credit_to_gid(naming::id_type const& id, 
        boost::uint16_t credit)
    {
        id.add_credit(credit);
    }

    void bulk_resolver_helper::set_credit_for_gid(naming::id_type const& id, 
        boost::uint16_t credit)
    {
        id.set_credit(credit);
    }

    int bulk_resolver_helper::incref(
        id_type const& id, boost::uint32_t credits, 
        id_type const& oldid, boost::uint32_t oldcredits)
    {
        tasks_.push_back(new do_undo_incref(
            boost::bind(&bulk_resolver_helper::add_credit_to_gid, id, credits),
            boost::bind(&bulk_resolver_helper::set_credit_for_gid, oldid, oldcredits)));
          return resolver_.incref(id.get_gid(), credits);
    }

    void bulk_resolver_helper::execute(error_code& ec)
    {
        // execute all resolver related tasks
        resolver_.execute(ec);

        // execute local tasks if successful
        if (!ec) {
            for (std::size_t i = 0; !ec && i < tasks_.size(); ++i)
            {
                tasks_[i].do_it(resolver_, i, ec);
            }
        }

        // invoke the undo action and undo all tasks already finished executing
        if (ec) {
            for (std::size_t i = 0; i < tasks_.size(); ++i)
            {
                tasks_[i].undo_it();
            }
        }
    }

    // trigger a full undo for all queued items
    void bulk_resolver_helper::undo()
    {
        for (std::size_t i = 0; i < tasks_.size(); ++i)
        {
            tasks_[i].undo_it();
        }
    }
}}}

#endif
