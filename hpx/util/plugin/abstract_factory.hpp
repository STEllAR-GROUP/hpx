// Copyright Vladimir Prus 2004.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_ABSTRACT_FACTORY_VP_2004_08_25
#define HPX_ABSTRACT_FACTORY_VP_2004_08_25

#include <boost/mpl/inherit_linearly.hpp>
#include <boost/mpl/list.hpp>
#include <boost/shared_ptr.hpp>

#include <hpx/util/plugin/virtual_constructors.hpp>

namespace hpx { namespace util { namespace plugin {

    namespace detail
    {
        struct abstract_factory_item_base
        {
            virtual ~abstract_factory_item_base() {}
            void create(int*******);
        };

        /** A template class, which is given the base type of plugin and a set
            of constructor parameter types and defines the appropriate virtual
            'create' function.
        */
        template<typename BasePlugin, typename Base, typename Parameters>
        struct abstract_factory_item;

        template<typename BasePlugin, typename Base>
        struct abstract_factory_item<BasePlugin, Base, boost::mpl::list<> >
        :   public Base
        {
            using Base::create;
            virtual BasePlugin* create(dll_handle dll) = 0;
        };

        template<typename BasePlugin, typename Base, typename A1>
        struct abstract_factory_item<BasePlugin, Base, boost::mpl::list<A1> >
        :   public Base
        {
            using Base::create;
            virtual BasePlugin* create(dll_handle dll, A1 a1) = 0;
        };

        template<typename BasePlugin, typename Base, typename A1, typename A2>
        struct abstract_factory_item<BasePlugin, Base, boost::mpl::list<A1, A2> >
        :   public Base
        {
            using Base::create;
            virtual BasePlugin* create(dll_handle dll, A1 a1, A2 a2) = 0;
        };

    ///////////////////////////////////////////////////////////////////////////
    //
    //  Bring in the remaining abstract_factory_item definitions for parameter
    //  counts greater 2
    //
    ///////////////////////////////////////////////////////////////////////////
    #include <hpx/util/plugin/detail/abstract_factory_impl.hpp>

    ///////////////////////////////////////////////////////////////////////////
    }   // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template<class BasePlugin>
    struct abstract_factory :
        public boost::mpl::inherit_linearly<
            typename virtual_constructors<BasePlugin>::type,
            detail::abstract_factory_item<BasePlugin,
                boost::mpl::placeholders::_, boost::mpl::placeholders::_>,
            detail::abstract_factory_item_base
        >::type
    {
    };

}}}

#endif
