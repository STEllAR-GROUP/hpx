// Copyright Vladimir Prus 2004.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONCRETE_FACTORY_VP_2004_08_25
#define HPX_CONCRETE_FACTORY_VP_2004_08_25

#include <hpx/config.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/plugin/abstract_factory.hpp>
#include <hpx/util/plugin/plugin_wrapper.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace plugin {

    namespace detail
    {
        template<
            typename BasePlugin, typename Concrete, typename Base,
            typename Parameter
        >
        struct concrete_factory_item;


        template<
            typename BasePlugin, typename Concrete, typename Base,
            typename...Parameters
        >
        struct concrete_factory_item<BasePlugin, Concrete, Base,
            hpx::util::detail::pack<Parameters...> >
          : public Base
        {
            BasePlugin* create(dll_handle dll, Parameters...parameters)
            {
                return new plugin_wrapper<Concrete, Parameters...>(dll, parameters...);
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template<typename BasePlugin, typename Concrete>
    struct concrete_factory
      : detail::concrete_factory_item<
            BasePlugin,
            Concrete, abstract_factory<BasePlugin>,
            typename virtual_constructor<BasePlugin>::type
    > {};

///////////////////////////////////////////////////////////////////////////////
}}}  // namespace hpx::util::plugin

#endif
