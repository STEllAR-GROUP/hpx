// Copyright Vladimir Prus 2004.
// Copyright (c) 2005-2014 Hartmut Kaiser
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_VIRTUAL_CONSTRUCTORS_VP_2004_08_05
#define HPX_VIRTUAL_CONSTRUCTORS_VP_2004_08_05

#include <boost/shared_ptr.hpp>
#include <boost/any.hpp>

#include <string>
#include <map>

#include <hpx/util/plugin/config.hpp>
#include <hpx/util/detail/pack.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace plugin {

    ///////////////////////////////////////////////////////////////////////////
    typedef std::map<std::string, boost::any> exported_plugins_type;
    typedef exported_plugins_type* (HPX_PLUGIN_API *get_plugins_list_type)();
    typedef exported_plugins_type* (HPX_PLUGIN_API get_plugins_list_np)();
    typedef boost::shared_ptr<get_plugins_list_np> dll_handle;

    ///////////////////////////////////////////////////////////////////////////
    template<typename BasePlugin>
    struct virtual_constructor
    {
        typedef hpx::util::detail::pack<> type;
    };

///////////////////////////////////////////////////////////////////////////////
}}}  // namespace hpx::util::plugin

#endif
