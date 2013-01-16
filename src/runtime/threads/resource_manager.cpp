//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/resource_manager.hpp>
#include <hpx/lcos/local/once.hpp>
#include <hpx/util/reinitializable_static.hpp>

#include <boost/foreach.hpp>

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    resource_manager& resource_manager::get()
    {
        typedef util::reinitializable_static<
           resource_manager, tag, 1, lcos::local::once_flag
        > static_type;

        static_type instance;
        return instance.get();
    }

    ///////////////////////////////////////////////////////////////////////////
    resource_manager::resource_manager()
      : next_cookie_(0)
    {}

    std::size_t resource_manager::initial_allocation(
        detail::manage_executor* proxy, error_code& ec)
    {
        if (0 == proxy) {
            HPX_THROWS_IF(ec, bad_parameter, 
                "resource_manager::init_allocation", 
                "manage_executor pointer is a nullptr");
        }

        // ask executor for its policies
        error_code ec1(lightweight);
        std::size_t min_punits = proxy->get_policy_element(detail::min_concurrency, ec1);
        if (ec1) min_punits = 1;
        std::size_t max_punits = proxy->get_policy_element(detail::max_concurrency, ec1);
        if (ec1) max_punits = 1;

        // allocate initial resources for the given executor
        std::vector<std::size_t> virt_cores;
        virt_cores.reserve(max_punits);

        std::size_t i = 0;
        for (/**/; i < max_punits; ++i)
        {
            proxy->add_processing_unit(i, i, ec);     // poor man's allocation
            if (ec) break;

            virt_cores.push_back(i);
        }

        if (ec) {
            // remove the already assigned virtual cores
            for (std::size_t j = 0; j < i; ++j)
                proxy->remove_processing_unit(i, ec1);
            return 0;
        }

        // attach the given proxy to this resource manager
        std::size_t cookie = ++next_cookie_;

        {
            mutex_type::scoped_lock l(mtx_);
            proxies_.insert(proxies_map_type::value_type(
                cookie, proxy_data(proxy, boost::move(virt_cores))));
        }

        if (&ec != &throws)
            ec = make_success_code();
        return cookie;
    }

    // Detach the executor identified with the given cookie
    void resource_manager::detach(std::size_t cookie, error_code& ec)
    {
        mutex_type::scoped_lock l(mtx_);
        proxies_map_type::iterator it = proxies_.find(cookie);
        if (it == proxies_.end()) {
            HPX_THROWS_IF(ec, bad_parameter, "resource_manager::detach", 
                "the given cookie is not known to the resource manager");
            return;
        }

        // inform executor to give up virtual core
        proxy_data& p = (*it).second;
        BOOST_FOREACH(std::size_t virt_core, p.virt_cores_)
            p.proxy_->remove_processing_unit(virt_core, ec);

        proxies_.erase(cookie);
    }
}}

