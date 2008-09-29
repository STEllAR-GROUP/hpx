// Copyright Vladimir Prus 2004.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_VIRTUAL_CONSTRUCTORS_VP_2004_08_05
#define BOOST_VIRTUAL_CONSTRUCTORS_VP_2004_08_05

#include <boost/config.hpp>
#include <boost/mpl/list.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/any.hpp>

#include <string>
#include <map>

#include <boost/plugin/config.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace plugin {

    ///////////////////////////////////////////////////////////////////////////
    typedef std::map<std::string, boost::any> exported_plugins_type;
    typedef exported_plugins_type& (BOOST_PLUGIN_API *get_plugins_list_type)();
    typedef exported_plugins_type& (BOOST_PLUGIN_API get_plugins_list_np)();
    typedef shared_ptr<get_plugins_list_np> dll_handle;

    ///////////////////////////////////////////////////////////////////////////
    template<typename BasePlugin>
    struct virtual_constructors 
    {
        typedef boost::mpl::list<boost::mpl::list<> > type;
    };

///////////////////////////////////////////////////////////////////////////////
}}  // namespace boost::plugin

#endif
