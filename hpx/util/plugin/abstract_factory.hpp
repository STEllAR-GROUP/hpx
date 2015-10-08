// Copyright Vladimir Prus 2004.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_ABSTRACT_FACTORY_VP_2004_08_25
#define HPX_ABSTRACT_FACTORY_VP_2004_08_25

#include <hpx/util/plugin/virtual_constructor.hpp>

namespace hpx { namespace util { namespace plugin {

    namespace detail
    {
        struct abstract_factory_item_base
        {
            virtual ~abstract_factory_item_base() {}
            void create(int*******);
        };

        template<typename BasePlugin, typename Base, typename Parameter>
        struct abstract_factory_item;
        /** A template class, which is given the base type of plugin and a set
            of constructor parameter types and defines the appropriate virtual
            'create' function.
        */
        template<typename BasePlugin, typename Base, typename...Parameters>
        struct abstract_factory_item<BasePlugin, Base,
            hpx::util::detail::pack<Parameters...> >
          : public Base
        {
            using Base::create;
            virtual BasePlugin* create(dll_handle dll, Parameters...parameters) = 0;
        };

    ///////////////////////////////////////////////////////////////////////////
    }   // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template<class BasePlugin>
    struct abstract_factory
      : detail::abstract_factory_item<
            BasePlugin,
            detail::abstract_factory_item_base,
            typename virtual_constructor<BasePlugin>::type
        >
    {};

}}}

#endif
