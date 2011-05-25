////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/runtime/components/server/wrapper_heap.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>

namespace hpx { namespace components { namespace detail
{

void store_agas_client(naming::resolver_client*& p, hpx::applier::applier& appl)
{ p = &appl.get_agas_client(); }

bool bind_range(naming::resolver_client* client, naming::gid_type const& gid,
                std::size_t count, naming::address const& addr,
                std::size_t offset)
{ return client->bind_range(gid, count, addr, offset); }

void unbind_range(naming::resolver_client* client, naming::gid_type const& gid,
                  std::size_t count)
{ client->unbind_range(gid, count); }

}}}

