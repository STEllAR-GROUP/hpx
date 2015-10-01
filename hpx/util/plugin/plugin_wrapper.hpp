// Copyright Vladimir Prus 2004.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PLUGIN_WRAPPER_VP_2004_08_25
#define HPX_PLUGIN_WRAPPER_VP_2004_08_25

#include <hpx/util/plugin/virtual_constructor.hpp>

namespace hpx { namespace util { namespace plugin {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct dll_handle_holder
        {
            dll_handle_holder(dll_handle dll)
            :   m_dll(dll) {}

            ~dll_handle_holder()
            {}

        private:
            dll_handle m_dll;
        };
    }

    ///////////////////////////////////////////////////////////////////////////

    template<typename Wrapped, typename...Parameters>
    struct plugin_wrapper
    :   public detail::dll_handle_holder,
        public Wrapped
    {
        plugin_wrapper(dll_handle dll, Parameters...parameters)
          : detail::dll_handle_holder(dll)
          , Wrapped(parameters...)
        {}
    };
}}}

#endif
