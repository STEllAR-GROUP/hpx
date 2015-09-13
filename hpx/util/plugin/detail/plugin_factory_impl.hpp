// Copyright Vladimir Prus 2004.
// Copyright (c) 2005-2014 Hartmut Kaiser
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#ifndef HPX_PLUGIN_FACTORY_IMPL_HK_2005_11_07
#define HPX_PLUGIN_FACTORY_IMPL_HK_2005_11_07

#include <boost/mpl/list.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#include <hpx/util/plugin/config.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (3, HPX_PLUGIN_ARGUMENT_LIMIT,                                        \
    "hpx/util/plugin/detail/plugin_factory_impl.hpp"))
#include BOOST_PP_ITERATE()

#endif  // HPX_PLUGIN_FACTORY_IMPL_HK_2005_11_07

///////////////////////////////////////////////////////////////////////////////
//
//  Preprocessor vertical repetition code
//
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

///////////////////////////////////////////////////////////////////////////////
namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template<
        typename BasePlugin, typename Base,
        BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    struct plugin_factory_item<
        BasePlugin, Base,
        boost::mpl::list<BOOST_PP_ENUM_PARAMS(N, A)>
    >
    :   public Base
    {
        plugin_factory_item(dll& d, std::string const& basename)
          : Base(d, basename)
        {}

        using Base::create;
        BasePlugin* create(std::string const& name, BOOST_PP_ENUM_BINARY_PARAMS(N, A, a))
        {
            std::pair<abstract_factory<BasePlugin> *, dll_handle> r =
                get_abstract_factory<BasePlugin>(this->m_dll, name, this->m_basename);
            return r.first->create(r.second, BOOST_PP_ENUM_PARAMS(N, a));
        }

        BasePlugin* create(std::string const& name, error_code& ec,
            BOOST_PP_ENUM_BINARY_PARAMS(N, A, a))
        {
            std::pair<abstract_factory<BasePlugin> *, dll_handle> r =
                get_abstract_factory<BasePlugin>
                (this->m_dll, name, this->m_basename, ec);
            if (ec) return 0;

            return r.first->create(r.second, BOOST_PP_ENUM_PARAMS(N, a));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template<
        typename BasePlugin, typename Base,
        BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    struct static_plugin_factory_item<
        BasePlugin, Base,
        boost::mpl::list<BOOST_PP_ENUM_PARAMS(N, A)>
    >
    :   public Base
    {
        static_plugin_factory_item(get_plugins_list_type const& f)
          : Base(f)
        {}

        using Base::create;
        BasePlugin* create(std::string const& name, BOOST_PP_ENUM_BINARY_PARAMS(N, A, a))
        {
            std::pair<abstract_factory<BasePlugin> *, dll_handle> r =
                get_abstract_factory_static<BasePlugin>(
                    this->f, &empty_deleter, name);
            return r.first->create(r.second, BOOST_PP_ENUM_PARAMS(N, a));
        }

        BasePlugin* create(std::string const& name, error_code& ec,
            BOOST_PP_ENUM_BINARY_PARAMS(N, A, a))
        {
            std::pair<abstract_factory<BasePlugin> *, dll_handle> r =
                get_abstract_factory_static<BasePlugin>(
                    this->f, &empty_deleter, name, "", ec);
            if (ec) return 0;

            return r.first->create(r.second, BOOST_PP_ENUM_PARAMS(N, a));
        }
    };

///////////////////////////////////////////////////////////////////////////////
}   // namespace hpx::util::plugin::detail

#undef N
#endif // defined(BOOST_PP_IS_ITERATING)

