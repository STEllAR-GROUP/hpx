//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_MAKE_CLIENT_JAN_02_2017_0220PM)
#define HPX_COMPONENTS_MAKE_CLIENT_JAN_02_2017_0220PM

#include <hpx/config.hpp>
#include <hpx/lcos_fwd.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/traits/is_client.hpp>

#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Create client objects from id_type, future<id_type>, etc.
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Client>
    inline typename std::enable_if<
        traits::is_client<Client>::value, Client
    >::type
    make_client(hpx::id_type const& id)
    {
        return Client(id);
    }

    template <typename Client>
    inline typename std::enable_if<
        traits::is_client<Client>::value, Client
    >::type
    make_client(hpx::future<hpx::id_type> const& id)
    {
        return Client(id);
    }

    template <typename Client>
    inline typename std::enable_if<
        traits::is_client<Client>::value, Client
    >::type
    make_client(hpx::future<hpx::id_type> && id)
    {
        return Client(std::move(id));
    }

    template <typename Client>
    inline typename std::enable_if<
        traits::is_client<Client>::value, Client
    >::type
    make_client(hpx::shared_future<hpx::id_type> const& id)
    {
        return Client(id);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Client>
    inline typename std::enable_if<
        traits::is_client<Client>::value, std::vector<Client>
    >::type
    make_clients(std::vector<hpx::id_type> const& ids)
    {
        std::vector<Client> result;
        result.reserve(ids.size());
        for (hpx::id_type const& id: ids)
        {
            result.push_back(Client(id));
        }
        return result;
    }

    template <typename Client>
    inline typename std::enable_if<
        traits::is_client<Client>::value, std::vector<Client>
    >::type
    make_clients(std::vector<hpx::future<hpx::id_type> > const& ids)
    {
        std::vector<Client> result;
        result.reserve(ids.size());
        for (hpx::future<hpx::id_type> const& id: ids)
        {
            result.push_back(Client(id));
        }
        return result;
    }

    template <typename Client>
    inline typename std::enable_if<
        traits::is_client<Client>::value, std::vector<Client>
    >::type
    make_clients(std::vector<hpx::future<hpx::id_type> > && ids)
    {
        std::vector<Client> result;
        result.reserve(ids.size());
        for (hpx::future<hpx::id_type>& id: ids)
        {
            result.push_back(Client(std::move(id)));
        }
        return result;
    }

    template <typename Client>
    inline typename std::enable_if<
        traits::is_client<Client>::value, std::vector<Client>
    >::type
    make_clients(std::vector<hpx::shared_future<hpx::id_type> > const& ids)
    {
        std::vector<Client> result;
        result.reserve(ids.size());
        for (hpx::shared_future<hpx::id_type> const& id: ids)
        {
            result.push_back(Client(id));
        }
        return result;
    }
}}

#endif

