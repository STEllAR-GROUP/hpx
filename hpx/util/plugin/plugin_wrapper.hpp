// Copyright Vladimir Prus 2004.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PLUGIN_WRAPPER_VP_2004_08_25
#define HPX_PLUGIN_WRAPPER_VP_2004_08_25

#include <boost/mpl/list.hpp>
#include <boost/shared_ptr.hpp>

#include <iostream>

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
    template<typename Wrapped, typename Parameters>
    struct plugin_wrapper;

    template<typename Wrapped>
    struct plugin_wrapper<Wrapped, boost::mpl::list<> >
    :   public detail::dll_handle_holder,
        public Wrapped
    {
        plugin_wrapper(dll_handle dll)
        :   detail::dll_handle_holder(dll)
        {}
    };

    template<typename Wrapped, typename A1>
    struct plugin_wrapper<Wrapped, boost::mpl::list<A1> >
    :   public detail::dll_handle_holder,
        public Wrapped
    {
        plugin_wrapper(dll_handle dll, A1 a1)
        :   detail::dll_handle_holder(dll),
            Wrapped(a1)
        {}
    };

    template<typename Wrapped, typename A1, typename A2>
    struct plugin_wrapper<Wrapped, boost::mpl::list<A1, A2> >
    :   public detail::dll_handle_holder,
        public Wrapped
    {
        plugin_wrapper(dll_handle dll, A1 a1, A2 a2)
        :   detail::dll_handle_holder(dll),
            Wrapped(a1, a2)
        {}
    };

}}}

#endif
