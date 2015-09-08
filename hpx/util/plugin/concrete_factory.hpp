// Copyright Vladimir Prus 2004.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONCRETE_FACTORY_VP_2004_08_25
#define HPX_CONCRETE_FACTORY_VP_2004_08_25

#include <iostream>

#include <hpx/util/plugin/abstract_factory.hpp>
#include <hpx/util/plugin/detail/plugin_wrapper.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace plugin {

    namespace detail
    {
        template<
            typename BasePlugin, typename Concrete, typename Base,
            typename Parameters
        >
        struct concrete_factory_item;

        template<typename BasePlugin, typename Concrete, typename Base>
        struct concrete_factory_item<
            BasePlugin, Concrete, Base, boost::mpl::list<>
        >
        :   public Base
        {
            BasePlugin* create(dll_handle dll)
            {
                return new plugin_wrapper<Concrete, boost::mpl::list<> >(dll);
            }
        };

        template<typename BasePlugin, typename Concrete, typename Base, typename A1>
        struct concrete_factory_item<
            BasePlugin, Concrete, Base, boost::mpl::list<A1>
        >
        :   public Base
        {
            BasePlugin* create(dll_handle dll, A1 a1)
            {
                return new plugin_wrapper<Concrete, boost::mpl::list<A1> >(dll, a1);
            }
        };

        template<typename BasePlugin, typename Concrete, typename Base,
            typename A1, typename A2>
        struct concrete_factory_item<BasePlugin, Concrete, Base,
            boost::mpl::list<A1, A2> >
        :   public Base
        {
            BasePlugin* create(dll_handle dll, A1 a1, A2 a2)
            {
                return new plugin_wrapper<Concrete, boost::mpl::list<A1
                    , A2> >(dll, a1, a2);
            }
        };
    }

///////////////////////////////////////////////////////////////////////////////
//
//  Bring in the remaining concrete_factory_item definitions for parameter
//  counts greater 2
//
///////////////////////////////////////////////////////////////////////////////
#include <hpx/util/plugin/detail/concrete_factory_impl.hpp>

    ///////////////////////////////////////////////////////////////////////////
    template<typename BasePlugin, typename Concrete>
    struct concrete_factory
    :   public boost::mpl::inherit_linearly<
            typename virtual_constructors<BasePlugin>::type,
            detail::concrete_factory_item<BasePlugin, Concrete,
                boost::mpl::placeholders::_, boost::mpl::placeholders::_>,
            abstract_factory<BasePlugin>
        >::type
    {};

///////////////////////////////////////////////////////////////////////////////
}}}  // namespace hpx::util::plugin

#endif
