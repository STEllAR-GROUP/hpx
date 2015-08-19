// Copyright Vladimir Prus 2004.
// Copyright (c) 2005-2014 Hartmut Kaiser
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PLUGIN_FACTORY_VP_2004_08_25
#define HPX_PLUGIN_FACTORY_VP_2004_08_25

#include <utility>
#include <stdexcept>
#include <string>
#include <utility>

#include <boost/config.hpp>
#include <boost/any.hpp>
#include <boost/function.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/type_traits/remove_pointer.hpp>

#include <hpx/util/plugin/virtual_constructors.hpp>
#include <hpx/util/plugin/abstract_factory.hpp>
#include <hpx/util/plugin/dll.hpp>
#include <hpx/util/plugin/export_plugin.hpp>

#include <hpx/exception.hpp>

namespace hpx { namespace util { namespace plugin {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template<typename BasePlugin, typename DeleterType>
        std::pair<abstract_factory<BasePlugin> *, dll_handle>
        get_abstract_factory_static(get_plugins_list_type f, DeleterType d,
            std::string const &class_name, std::string const& libname = "",
            error_code& ec = throws)
        {
            typedef typename boost::remove_pointer<get_plugins_list_type>
                ::type PointedType;

            exported_plugins_type& e = *f();
            std::string clsname(class_name);
            boost::algorithm::to_lower(clsname);

            typename exported_plugins_type::iterator it = e.find(clsname);
            if (it != e.end()) {
                abstract_factory<BasePlugin>** xw =
                    boost::unsafe_any_cast<abstract_factory<BasePlugin> *>
                    (&(*it).second);

                if (!xw) {
                    HPX_THROWS_IF(ec, filesystem_error,
                        "get_abstract_factory_static",
                        "Hpx.Plugin: Can't cast to the right factory type\n");
                    return std::pair<abstract_factory<BasePlugin> *, dll_handle>();
                }

                abstract_factory<BasePlugin> *w = *xw;
                return make_pair(w, boost::shared_ptr<PointedType>(f, d));
            }
            else {
                std::ostringstream str;
                str << "Hpx.Plugin: Class '" << class_name
                    << "' was not found";

                if (!libname.empty())
                    str << " in the shared library '" << libname << "'.";

                if (!e.empty()) {
                    str << " Existing classes: ";

                    bool first = true;
                    typename exported_plugins_type::iterator end = e.end();
                    for (typename exported_plugins_type::iterator jt = e.begin();
                         jt != end; ++jt)
                    {
                        if (first) {
                            str << "'" << (*jt).first << "'";
                            first = false;
                        }
                        else {
                            str << ", '" << (*jt).first << "'";
                        }
                    }
                    str << ".";
                }
                else {
                    str << " No classes exist.";
                }

                HPX_THROWS_IF(ec, filesystem_error,
                    "get_abstract_factory_static",
                    str.str());
                return std::pair<abstract_factory<BasePlugin> *, dll_handle>();
            }
        }

        template<typename BasePlugin>
        std::pair<abstract_factory<BasePlugin> *, dll_handle>
        get_abstract_factory(dll const& d, std::string const &class_name,
            std::string const &base_name, error_code& ec = throws)
        {
            typedef boost::function<void (get_plugins_list_type)> DeleterType;

            std::string plugin_entry(HPX_PLUGIN_SYMBOLS_PREFIX_DYNAMIC_STR
                "_exported_plugins_list_");
            plugin_entry += d.get_mapname();
            plugin_entry += "_" + base_name;

            std::pair<get_plugins_list_type, DeleterType> f =
                d.get<get_plugins_list_type, DeleterType>(plugin_entry, ec);
            if (ec) return std::pair<abstract_factory<BasePlugin> *, dll_handle>();

            return get_abstract_factory_static<BasePlugin>(f.first, f.second,
                class_name, d.get_name(), ec);
        }

        ///////////////////////////////////////////////////////////////////////
        inline void
        get_abstract_factory_names_static(get_plugins_list_type f,
            std::vector<std::string>& names, error_code& ec = throws)
        {
            exported_plugins_type& e = *f();

            exported_plugins_type::iterator end = e.end();
            for (exported_plugins_type::iterator it = e.begin(); it != end; ++it)
            {
                names.push_back((*it).first);
            }
        }

        inline void
        get_abstract_factory_names(dll const& d, std::string const &base_name,
            std::vector<std::string>& names, error_code& ec = throws)
        {
            typedef boost::function<void (get_plugins_list_type)> DeleterType;

            std::string plugin_entry(HPX_PLUGIN_SYMBOLS_PREFIX_DYNAMIC_STR
                "_exported_plugins_list_");
            plugin_entry += d.get_mapname();
            plugin_entry += "_" + base_name;

            std::pair<get_plugins_list_type, DeleterType> f =
                d.get<get_plugins_list_type, DeleterType>(plugin_entry, ec);
            if (ec) return;

            get_abstract_factory_names_static(f.first, names, ec);
        }

        ///////////////////////////////////////////////////////////////////////
        struct plugin_factory_item_base
        {
            plugin_factory_item_base(dll& d, std::string const& basename)
              : m_dll(d), m_basename(basename)
            {}

            void create(int******) const;

            void get_names(std::vector<std::string>& names,
                error_code& ec = throws) const
            {
                get_abstract_factory_names(this->m_dll, this->m_basename, names, ec);
            }

        protected:
            dll& m_dll;
            std::string m_basename;
        };

        ///////////////////////////////////////////////////////////////////////
        template<typename BasePlugin, typename Base, typename Parameters>
        struct plugin_factory_item;

        template<typename BasePlugin, typename Base>
        struct plugin_factory_item<BasePlugin, Base, boost::mpl::list<> >
        :   public Base
        {
            plugin_factory_item(dll& d, std::string const& basename)
              : Base(d, basename)
            {}

            BasePlugin* create(std::string const& name, error_code& ec = throws) const
            {
                std::pair<abstract_factory<BasePlugin> *, dll_handle> r =
                    get_abstract_factory<BasePlugin>(this->m_dll, name,
                        this->m_basename, ec);
                if (ec) return 0;

                return r.first->create(r.second);
            }
        };

        template<typename BasePlugin, typename Base, typename A1>
        struct plugin_factory_item<BasePlugin, Base, boost::mpl::list<A1> >
        :   public Base
        {
            plugin_factory_item(dll& d, std::string const& basename)
              : Base(d, basename)
            {}

            using Base::create;
            BasePlugin* create(std::string const& name, A1 a1) const
            {
                std::pair<abstract_factory<BasePlugin> *, dll_handle> r =
                    get_abstract_factory<BasePlugin>(this->m_dll, name,
                        this->m_basename);
                return r.first->create(r.second, a1);
            }

            BasePlugin* create(std::string const& name, error_code& ec, A1 a1) const
            {
                std::pair<abstract_factory<BasePlugin> *, dll_handle> r =
                    get_abstract_factory<BasePlugin>(this->m_dll, name,
                        this->m_basename, ec);
                if (ec) return 0;

                return r.first->create(r.second, a1);
            }
        };

        template<typename BasePlugin, typename Base, typename A1, typename A2>
        struct plugin_factory_item<BasePlugin, Base, boost::mpl::list<A1, A2> >
        :   public Base
        {
            plugin_factory_item(dll& d, std::string const& basename)
              : Base(d, basename)
            {}

            using Base::create;
            BasePlugin* create(std::string const& name, A1 a1, A2 a2) const
            {
                std::pair<abstract_factory<BasePlugin> *, dll_handle> r =
                    get_abstract_factory<BasePlugin>(this->m_dll, name,
                        this->m_basename);
                return r.first->create(r.second, a1, a2);
            }

            BasePlugin* create(std::string const& name, error_code& ec,
                A1 a1, A2 a2) const
            {
                std::pair<abstract_factory<BasePlugin> *, dll_handle> r =
                    get_abstract_factory<BasePlugin>(this->m_dll, name,
                        this->m_basename, ec);
                if (ec) return 0;

                return r.first->create(r.second, a1, a2);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // empty deleter for the smart pointer to be used for static
        // plugin_factories
        inline void empty_deleter(get_plugins_list_type) {}

        ///////////////////////////////////////////////////////////////////////
        struct static_plugin_factory_item_base
        {
            static_plugin_factory_item_base(get_plugins_list_type const& f_)
              : f(f_)
            {}

            void create(int******) const;

            void get_names(std::vector<std::string>& names,
                error_code& ec = throws) const
            {
                get_abstract_factory_names_static(f, names, ec);
            }

        protected:
            get_plugins_list_type f;
        };

        ///////////////////////////////////////////////////////////////////////
        template<typename BasePlugin, typename Base, typename Parameters>
        struct static_plugin_factory_item;

        template<typename BasePlugin, typename Base>
        struct static_plugin_factory_item<BasePlugin, Base, boost::mpl::list<> >
        :   public Base
        {
            static_plugin_factory_item(get_plugins_list_type const& f)
              : Base(f)
            {}

            BasePlugin* create(std::string const& name, error_code& ec = throws) const
            {
                std::pair<abstract_factory<BasePlugin> *, dll_handle> r =
                    get_abstract_factory_static<BasePlugin>(
                        this->f, &empty_deleter, name, "", ec);
                if (ec) return 0;

                return r.first->create(r.second);
            }
        };

        template<typename BasePlugin, typename Base, typename A1>
        struct static_plugin_factory_item<BasePlugin, Base, boost::mpl::list<A1> >
        :   public Base
        {
            static_plugin_factory_item(get_plugins_list_type const& f)
              : Base(f)
            {}

            using Base::create;
            BasePlugin* create(std::string const& name, A1 a1) const
            {
                std::pair<abstract_factory<BasePlugin> *, dll_handle> r =
                    get_abstract_factory_static<BasePlugin>(
                        this->f, &empty_deleter, name);
                return r.first->create(r.second, a1);
            }

            BasePlugin* create(std::string const& name, error_code& ec, A1 a1) const
            {
                std::pair<abstract_factory<BasePlugin> *, dll_handle> r =
                    get_abstract_factory_static<BasePlugin>(
                        this->f, &empty_deleter, name, "", ec);
                if (ec) return 0;

                return r.first->create(r.second, a1);
            }
        };

        template<typename BasePlugin, typename Base, typename A1, typename A2>
        struct static_plugin_factory_item<BasePlugin, Base, boost::mpl::list<A1, A2> >
        :   public Base
        {
            static_plugin_factory_item(get_plugins_list_type const& f)
              : Base(f)
            {}

            using Base::create;
            BasePlugin* create(std::string const& name, A1 a1, A2 a2) const
            {
                std::pair<abstract_factory<BasePlugin> *, dll_handle> r =
                    get_abstract_factory_static<BasePlugin>(
                        this->f, &empty_deleter, name);
                return r.first->create(r.second, a1, a2);
            }

            BasePlugin* create(std::string const& name,
                error_code& ec, A1 a1, A2 a2) const
            {
                std::pair<abstract_factory<BasePlugin> *, dll_handle> r =
                    get_abstract_factory_static<BasePlugin>(
                        this->f, &empty_deleter, name, ec);
                if (ec) return 0;

                return r.first->create(r.second, a1, a2);
            }
        };
    }

///////////////////////////////////////////////////////////////////////////////
//
//  Bring in the remaining plugin_factory_item definitions for parameter
//  counts greater 2
//
///////////////////////////////////////////////////////////////////////////////
#include <hpx/util/plugin/detail/plugin_factory_impl.hpp>

    ///////////////////////////////////////////////////////////////////////////
    template<class BasePlugin>
    struct plugin_factory
    :   public boost::mpl::inherit_linearly <
            typename virtual_constructors<BasePlugin>::type,
            detail::plugin_factory_item<BasePlugin,
                boost::mpl::placeholders::_, boost::mpl::placeholders::_>,
            detail::plugin_factory_item_base
        >::type
    {
    private:
        typedef typename boost::mpl::inherit_linearly <
            typename virtual_constructors<BasePlugin>::type,
            detail::plugin_factory_item<BasePlugin,
                boost::mpl::placeholders::_, boost::mpl::placeholders::_>,
            detail::plugin_factory_item_base
        >::type base_type;

    public:
        plugin_factory(dll& d, std::string const& basename)
          : base_type(d, basename)
        {}
    };

    ///////////////////////////////////////////////////////////////////////////
    template<class BasePlugin>
    struct static_plugin_factory
    :   public boost::mpl::inherit_linearly <
            typename virtual_constructors<BasePlugin>::type,
            detail::static_plugin_factory_item<BasePlugin,
                boost::mpl::placeholders::_, boost::mpl::placeholders::_>,
            detail::static_plugin_factory_item_base
        >::type
    {
    private:
        typedef typename boost::mpl::inherit_linearly <
            typename virtual_constructors<BasePlugin>::type,
            detail::static_plugin_factory_item<BasePlugin,
                boost::mpl::placeholders::_, boost::mpl::placeholders::_>,
            detail::static_plugin_factory_item_base
        >::type base_type;

    public:
        static_plugin_factory(get_plugins_list_type const& f)
          : base_type(f)
        {}
    };

///////////////////////////////////////////////////////////////////////////////
}}}  // namespace hpx::util::plugin

#endif
