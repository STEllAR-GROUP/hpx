// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace components { namespace server
{
    
    
    template <typename Component
       >
    struct create_component_action0
      : ::hpx::actions::result_action0<
            naming::gid_type (runtime_support::*)()
          , &runtime_support::create_component0<
                Component >
          , create_component_action0<
                Component > >
    {};
    template <typename Component
       >
    struct create_component_direct_action0
      : ::hpx::actions::direct_result_action0<
            naming::gid_type (runtime_support::*)()
          , &runtime_support::create_component0<
                Component >
          , create_component_direct_action0<
                Component > >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component >) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action0< Component >))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component >) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action0< Component >))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action0< Component >))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component >) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action0< Component >))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action0< Component >))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component >) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action0< Component >))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action0< Component >))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action0< Component >))>))> >::get_mutable_instance().export_guid(); }}} }
namespace hpx { namespace components { namespace server
{
    template <typename Component, typename A0>
    naming::gid_type runtime_support::create_component1(
        A0 a0)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_the_lock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor1<
                    typename Component::wrapping_type,
                    A0>(
                        hpx::util::detail::move_if_no_ref< A0> ::call(a0)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);
        return id;
    }
    
    
    template <typename Component
      , typename A0>
    struct create_component_action1
      : ::hpx::actions::result_action1<
            naming::gid_type (runtime_support::*)(A0)
          , &runtime_support::create_component1<
                Component , A0>
          , create_component_action1<
                Component , A0> >
    {};
    template <typename Component
      , typename A0>
    struct create_component_direct_action1
      : ::hpx::actions::direct_result_action1<
            naming::gid_type (runtime_support::*)(A0)
          , &runtime_support::create_component1<
                Component , A0>
          , create_component_direct_action1<
                Component , A0> >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component , typename A0>) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action1< Component , A0>))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component , typename A0>) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action1< Component , A0>))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action1< Component , A0>))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component , typename A0>) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action1< Component , A0>))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action1< Component , A0>))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component , typename A0>) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action1< Component , A0>))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action1< Component , A0>))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action1< Component , A0>))>))> >::get_mutable_instance().export_guid(); }}} }
namespace hpx { namespace components { namespace server
{
    template <typename Component, typename A0 , typename A1>
    naming::gid_type runtime_support::create_component2(
        A0 a0 , A1 a1)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_the_lock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor2<
                    typename Component::wrapping_type,
                    A0 , A1>(
                        hpx::util::detail::move_if_no_ref< A0> ::call(a0) , hpx::util::detail::move_if_no_ref< A1> ::call(a1)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);
        return id;
    }
    
    
    template <typename Component
      , typename A0 , typename A1>
    struct create_component_action2
      : ::hpx::actions::result_action2<
            naming::gid_type (runtime_support::*)(A0 , A1)
          , &runtime_support::create_component2<
                Component , A0 , A1>
          , create_component_action2<
                Component , A0 , A1> >
    {};
    template <typename Component
      , typename A0 , typename A1>
    struct create_component_direct_action2
      : ::hpx::actions::direct_result_action2<
            naming::gid_type (runtime_support::*)(A0 , A1)
          , &runtime_support::create_component2<
                Component , A0 , A1>
          , create_component_direct_action2<
                Component , A0 , A1> >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1>) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action2< Component , A0 , A1>))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1>) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action2< Component , A0 , A1>))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action2< Component , A0 , A1>))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1>) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action2< Component , A0 , A1>))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action2< Component , A0 , A1>))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1>) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action2< Component , A0 , A1>))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action2< Component , A0 , A1>))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action2< Component , A0 , A1>))>))> >::get_mutable_instance().export_guid(); }}} }
namespace hpx { namespace components { namespace server
{
    template <typename Component, typename A0 , typename A1 , typename A2>
    naming::gid_type runtime_support::create_component3(
        A0 a0 , A1 a1 , A2 a2)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_the_lock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor3<
                    typename Component::wrapping_type,
                    A0 , A1 , A2>(
                        hpx::util::detail::move_if_no_ref< A0> ::call(a0) , hpx::util::detail::move_if_no_ref< A1> ::call(a1) , hpx::util::detail::move_if_no_ref< A2> ::call(a2)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);
        return id;
    }
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2>
    struct create_component_action3
      : ::hpx::actions::result_action3<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2)
          , &runtime_support::create_component3<
                Component , A0 , A1 , A2>
          , create_component_action3<
                Component , A0 , A1 , A2> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2>
    struct create_component_direct_action3
      : ::hpx::actions::direct_result_action3<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2)
          , &runtime_support::create_component3<
                Component , A0 , A1 , A2>
          , create_component_direct_action3<
                Component , A0 , A1 , A2> >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2>) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action3< Component , A0 , A1 , A2>))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2>) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action3< Component , A0 , A1 , A2>))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action3< Component , A0 , A1 , A2>))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2>) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action3< Component , A0 , A1 , A2>))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action3< Component , A0 , A1 , A2>))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2>) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action3< Component , A0 , A1 , A2>))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action3< Component , A0 , A1 , A2>))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action3< Component , A0 , A1 , A2>))>))> >::get_mutable_instance().export_guid(); }}} }
namespace hpx { namespace components { namespace server
{
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3>
    naming::gid_type runtime_support::create_component4(
        A0 a0 , A1 a1 , A2 a2 , A3 a3)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_the_lock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor4<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3>(
                        hpx::util::detail::move_if_no_ref< A0> ::call(a0) , hpx::util::detail::move_if_no_ref< A1> ::call(a1) , hpx::util::detail::move_if_no_ref< A2> ::call(a2) , hpx::util::detail::move_if_no_ref< A3> ::call(a3)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);
        return id;
    }
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3>
    struct create_component_action4
      : ::hpx::actions::result_action4<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3)
          , &runtime_support::create_component4<
                Component , A0 , A1 , A2 , A3>
          , create_component_action4<
                Component , A0 , A1 , A2 , A3> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3>
    struct create_component_direct_action4
      : ::hpx::actions::direct_result_action4<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3)
          , &runtime_support::create_component4<
                Component , A0 , A1 , A2 , A3>
          , create_component_direct_action4<
                Component , A0 , A1 , A2 , A3> >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3>) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action4< Component , A0 , A1 , A2 , A3>))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3>) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action4< Component , A0 , A1 , A2 , A3>))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action4< Component , A0 , A1 , A2 , A3>))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3>) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action4< Component , A0 , A1 , A2 , A3>))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action4< Component , A0 , A1 , A2 , A3>))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3>) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action4< Component , A0 , A1 , A2 , A3>))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action4< Component , A0 , A1 , A2 , A3>))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action4< Component , A0 , A1 , A2 , A3>))>))> >::get_mutable_instance().export_guid(); }}} }
namespace hpx { namespace components { namespace server
{
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    naming::gid_type runtime_support::create_component5(
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_the_lock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor5<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4>(
                        hpx::util::detail::move_if_no_ref< A0> ::call(a0) , hpx::util::detail::move_if_no_ref< A1> ::call(a1) , hpx::util::detail::move_if_no_ref< A2> ::call(a2) , hpx::util::detail::move_if_no_ref< A3> ::call(a3) , hpx::util::detail::move_if_no_ref< A4> ::call(a4)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);
        return id;
    }
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    struct create_component_action5
      : ::hpx::actions::result_action5<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4)
          , &runtime_support::create_component5<
                Component , A0 , A1 , A2 , A3 , A4>
          , create_component_action5<
                Component , A0 , A1 , A2 , A3 , A4> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    struct create_component_direct_action5
      : ::hpx::actions::direct_result_action5<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4)
          , &runtime_support::create_component5<
                Component , A0 , A1 , A2 , A3 , A4>
          , create_component_direct_action5<
                Component , A0 , A1 , A2 , A3 , A4> >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4>) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action5< Component , A0 , A1 , A2 , A3 , A4>))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4>) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action5< Component , A0 , A1 , A2 , A3 , A4>))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action5< Component , A0 , A1 , A2 , A3 , A4>))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4>) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action5< Component , A0 , A1 , A2 , A3 , A4>))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action5< Component , A0 , A1 , A2 , A3 , A4>))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4>) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action5< Component , A0 , A1 , A2 , A3 , A4>))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action5< Component , A0 , A1 , A2 , A3 , A4>))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action5< Component , A0 , A1 , A2 , A3 , A4>))>))> >::get_mutable_instance().export_guid(); }}} }
namespace hpx { namespace components { namespace server
{
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    naming::gid_type runtime_support::create_component6(
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_the_lock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor6<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4 , A5>(
                        hpx::util::detail::move_if_no_ref< A0> ::call(a0) , hpx::util::detail::move_if_no_ref< A1> ::call(a1) , hpx::util::detail::move_if_no_ref< A2> ::call(a2) , hpx::util::detail::move_if_no_ref< A3> ::call(a3) , hpx::util::detail::move_if_no_ref< A4> ::call(a4) , hpx::util::detail::move_if_no_ref< A5> ::call(a5)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);
        return id;
    }
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    struct create_component_action6
      : ::hpx::actions::result_action6<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5)
          , &runtime_support::create_component6<
                Component , A0 , A1 , A2 , A3 , A4 , A5>
          , create_component_action6<
                Component , A0 , A1 , A2 , A3 , A4 , A5> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    struct create_component_direct_action6
      : ::hpx::actions::direct_result_action6<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5)
          , &runtime_support::create_component6<
                Component , A0 , A1 , A2 , A3 , A4 , A5>
          , create_component_direct_action6<
                Component , A0 , A1 , A2 , A3 , A4 , A5> >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action6< Component , A0 , A1 , A2 , A3 , A4 , A5>))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action6< Component , A0 , A1 , A2 , A3 , A4 , A5>))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action6< Component , A0 , A1 , A2 , A3 , A4 , A5>))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action6< Component , A0 , A1 , A2 , A3 , A4 , A5>))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action6< Component , A0 , A1 , A2 , A3 , A4 , A5>))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action6< Component , A0 , A1 , A2 , A3 , A4 , A5>))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action6< Component , A0 , A1 , A2 , A3 , A4 , A5>))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action6< Component , A0 , A1 , A2 , A3 , A4 , A5>))>))> >::get_mutable_instance().export_guid(); }}} }
namespace hpx { namespace components { namespace server
{
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    naming::gid_type runtime_support::create_component7(
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_the_lock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor7<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4 , A5 , A6>(
                        hpx::util::detail::move_if_no_ref< A0> ::call(a0) , hpx::util::detail::move_if_no_ref< A1> ::call(a1) , hpx::util::detail::move_if_no_ref< A2> ::call(a2) , hpx::util::detail::move_if_no_ref< A3> ::call(a3) , hpx::util::detail::move_if_no_ref< A4> ::call(a4) , hpx::util::detail::move_if_no_ref< A5> ::call(a5) , hpx::util::detail::move_if_no_ref< A6> ::call(a6)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);
        return id;
    }
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    struct create_component_action7
      : ::hpx::actions::result_action7<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6)
          , &runtime_support::create_component7<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6>
          , create_component_action7<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    struct create_component_direct_action7
      : ::hpx::actions::direct_result_action7<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6)
          , &runtime_support::create_component7<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6>
          , create_component_direct_action7<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6> >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action7< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6>))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action7< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6>))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action7< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6>))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action7< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6>))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action7< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6>))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action7< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6>))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action7< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6>))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action7< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6>))>))> >::get_mutable_instance().export_guid(); }}} }
namespace hpx { namespace components { namespace server
{
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    naming::gid_type runtime_support::create_component8(
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_the_lock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor8<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>(
                        hpx::util::detail::move_if_no_ref< A0> ::call(a0) , hpx::util::detail::move_if_no_ref< A1> ::call(a1) , hpx::util::detail::move_if_no_ref< A2> ::call(a2) , hpx::util::detail::move_if_no_ref< A3> ::call(a3) , hpx::util::detail::move_if_no_ref< A4> ::call(a4) , hpx::util::detail::move_if_no_ref< A5> ::call(a5) , hpx::util::detail::move_if_no_ref< A6> ::call(a6) , hpx::util::detail::move_if_no_ref< A7> ::call(a7)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);
        return id;
    }
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    struct create_component_action8
      : ::hpx::actions::result_action8<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
          , &runtime_support::create_component8<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>
          , create_component_action8<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    struct create_component_direct_action8
      : ::hpx::actions::direct_result_action8<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
          , &runtime_support::create_component8<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>
          , create_component_direct_action8<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7> >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action8< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action8< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action8< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action8< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action8< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action8< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action8< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action8< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>))>))> >::get_mutable_instance().export_guid(); }}} }
namespace hpx { namespace components { namespace server
{
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    naming::gid_type runtime_support::create_component9(
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_the_lock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor9<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>(
                        hpx::util::detail::move_if_no_ref< A0> ::call(a0) , hpx::util::detail::move_if_no_ref< A1> ::call(a1) , hpx::util::detail::move_if_no_ref< A2> ::call(a2) , hpx::util::detail::move_if_no_ref< A3> ::call(a3) , hpx::util::detail::move_if_no_ref< A4> ::call(a4) , hpx::util::detail::move_if_no_ref< A5> ::call(a5) , hpx::util::detail::move_if_no_ref< A6> ::call(a6) , hpx::util::detail::move_if_no_ref< A7> ::call(a7) , hpx::util::detail::move_if_no_ref< A8> ::call(a8)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);
        return id;
    }
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    struct create_component_action9
      : ::hpx::actions::result_action9<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
          , &runtime_support::create_component9<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>
          , create_component_action9<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    struct create_component_direct_action9
      : ::hpx::actions::direct_result_action9<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
          , &runtime_support::create_component9<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>
          , create_component_direct_action9<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8> >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action9< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action9< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action9< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action9< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action9< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action9< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action9< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action9< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>))>))> >::get_mutable_instance().export_guid(); }}} }
namespace hpx { namespace components { namespace server
{
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
    naming::gid_type runtime_support::create_component10(
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_the_lock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor10<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>(
                        hpx::util::detail::move_if_no_ref< A0> ::call(a0) , hpx::util::detail::move_if_no_ref< A1> ::call(a1) , hpx::util::detail::move_if_no_ref< A2> ::call(a2) , hpx::util::detail::move_if_no_ref< A3> ::call(a3) , hpx::util::detail::move_if_no_ref< A4> ::call(a4) , hpx::util::detail::move_if_no_ref< A5> ::call(a5) , hpx::util::detail::move_if_no_ref< A6> ::call(a6) , hpx::util::detail::move_if_no_ref< A7> ::call(a7) , hpx::util::detail::move_if_no_ref< A8> ::call(a8) , hpx::util::detail::move_if_no_ref< A9> ::call(a9)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);
        return id;
    }
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
    struct create_component_action10
      : ::hpx::actions::result_action10<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
          , &runtime_support::create_component10<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>
          , create_component_action10<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
    struct create_component_direct_action10
      : ::hpx::actions::direct_result_action10<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
          , &runtime_support::create_component10<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>
          , create_component_direct_action10<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9> >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action10< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action10< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action10< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action10< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action10< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action10< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action10< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action10< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>))>))> >::get_mutable_instance().export_guid(); }}} }
namespace hpx { namespace components { namespace server
{
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
    naming::gid_type runtime_support::create_component11(
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_the_lock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor11<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10>(
                        hpx::util::detail::move_if_no_ref< A0> ::call(a0) , hpx::util::detail::move_if_no_ref< A1> ::call(a1) , hpx::util::detail::move_if_no_ref< A2> ::call(a2) , hpx::util::detail::move_if_no_ref< A3> ::call(a3) , hpx::util::detail::move_if_no_ref< A4> ::call(a4) , hpx::util::detail::move_if_no_ref< A5> ::call(a5) , hpx::util::detail::move_if_no_ref< A6> ::call(a6) , hpx::util::detail::move_if_no_ref< A7> ::call(a7) , hpx::util::detail::move_if_no_ref< A8> ::call(a8) , hpx::util::detail::move_if_no_ref< A9> ::call(a9) , hpx::util::detail::move_if_no_ref< A10> ::call(a10)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);
        return id;
    }
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
    struct create_component_action11
      : ::hpx::actions::result_action11<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
          , &runtime_support::create_component11<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10>
          , create_component_action11<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
    struct create_component_direct_action11
      : ::hpx::actions::direct_result_action11<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
          , &runtime_support::create_component11<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10>
          , create_component_direct_action11<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10> >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action11< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10>))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action11< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10>))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action11< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10>))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action11< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10>))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action11< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10>))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action11< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10>))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action11< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10>))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action11< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10>))>))> >::get_mutable_instance().export_guid(); }}} }
namespace hpx { namespace components { namespace server
{
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
    naming::gid_type runtime_support::create_component12(
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_the_lock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor12<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11>(
                        hpx::util::detail::move_if_no_ref< A0> ::call(a0) , hpx::util::detail::move_if_no_ref< A1> ::call(a1) , hpx::util::detail::move_if_no_ref< A2> ::call(a2) , hpx::util::detail::move_if_no_ref< A3> ::call(a3) , hpx::util::detail::move_if_no_ref< A4> ::call(a4) , hpx::util::detail::move_if_no_ref< A5> ::call(a5) , hpx::util::detail::move_if_no_ref< A6> ::call(a6) , hpx::util::detail::move_if_no_ref< A7> ::call(a7) , hpx::util::detail::move_if_no_ref< A8> ::call(a8) , hpx::util::detail::move_if_no_ref< A9> ::call(a9) , hpx::util::detail::move_if_no_ref< A10> ::call(a10) , hpx::util::detail::move_if_no_ref< A11> ::call(a11)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);
        return id;
    }
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
    struct create_component_action12
      : ::hpx::actions::result_action12<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
          , &runtime_support::create_component12<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11>
          , create_component_action12<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
    struct create_component_direct_action12
      : ::hpx::actions::direct_result_action12<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
          , &runtime_support::create_component12<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11>
          , create_component_direct_action12<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11> >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action12< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11>))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action12< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11>))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action12< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11>))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action12< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11>))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action12< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11>))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action12< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11>))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action12< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11>))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action12< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11>))>))> >::get_mutable_instance().export_guid(); }}} }
namespace hpx { namespace components { namespace server
{
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
    naming::gid_type runtime_support::create_component13(
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_the_lock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor13<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12>(
                        hpx::util::detail::move_if_no_ref< A0> ::call(a0) , hpx::util::detail::move_if_no_ref< A1> ::call(a1) , hpx::util::detail::move_if_no_ref< A2> ::call(a2) , hpx::util::detail::move_if_no_ref< A3> ::call(a3) , hpx::util::detail::move_if_no_ref< A4> ::call(a4) , hpx::util::detail::move_if_no_ref< A5> ::call(a5) , hpx::util::detail::move_if_no_ref< A6> ::call(a6) , hpx::util::detail::move_if_no_ref< A7> ::call(a7) , hpx::util::detail::move_if_no_ref< A8> ::call(a8) , hpx::util::detail::move_if_no_ref< A9> ::call(a9) , hpx::util::detail::move_if_no_ref< A10> ::call(a10) , hpx::util::detail::move_if_no_ref< A11> ::call(a11) , hpx::util::detail::move_if_no_ref< A12> ::call(a12)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);
        return id;
    }
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
    struct create_component_action13
      : ::hpx::actions::result_action13<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
          , &runtime_support::create_component13<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12>
          , create_component_action13<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
    struct create_component_direct_action13
      : ::hpx::actions::direct_result_action13<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
          , &runtime_support::create_component13<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12>
          , create_component_direct_action13<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12> >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action13< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12>))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action13< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12>))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action13< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12>))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action13< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12>))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action13< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12>))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action13< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12>))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action13< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12>))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action13< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12>))>))> >::get_mutable_instance().export_guid(); }}} }
namespace hpx { namespace components { namespace server
{
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
    naming::gid_type runtime_support::create_component14(
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_the_lock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor14<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13>(
                        hpx::util::detail::move_if_no_ref< A0> ::call(a0) , hpx::util::detail::move_if_no_ref< A1> ::call(a1) , hpx::util::detail::move_if_no_ref< A2> ::call(a2) , hpx::util::detail::move_if_no_ref< A3> ::call(a3) , hpx::util::detail::move_if_no_ref< A4> ::call(a4) , hpx::util::detail::move_if_no_ref< A5> ::call(a5) , hpx::util::detail::move_if_no_ref< A6> ::call(a6) , hpx::util::detail::move_if_no_ref< A7> ::call(a7) , hpx::util::detail::move_if_no_ref< A8> ::call(a8) , hpx::util::detail::move_if_no_ref< A9> ::call(a9) , hpx::util::detail::move_if_no_ref< A10> ::call(a10) , hpx::util::detail::move_if_no_ref< A11> ::call(a11) , hpx::util::detail::move_if_no_ref< A12> ::call(a12) , hpx::util::detail::move_if_no_ref< A13> ::call(a13)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);
        return id;
    }
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
    struct create_component_action14
      : ::hpx::actions::result_action14<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
          , &runtime_support::create_component14<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13>
          , create_component_action14<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
    struct create_component_direct_action14
      : ::hpx::actions::direct_result_action14<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
          , &runtime_support::create_component14<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13>
          , create_component_direct_action14<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13> >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action14< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13>))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action14< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13>))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action14< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13>))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action14< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13>))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action14< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13>))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action14< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13>))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action14< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13>))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action14< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13>))>))> >::get_mutable_instance().export_guid(); }}} }
namespace hpx { namespace components { namespace server
{
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>
    naming::gid_type runtime_support::create_component15(
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return naming::invalid_gid;
        }
        naming::gid_type id;
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::unlock_the_lock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor15<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14>(
                        hpx::util::detail::move_if_no_ref< A0> ::call(a0) , hpx::util::detail::move_if_no_ref< A1> ::call(a1) , hpx::util::detail::move_if_no_ref< A2> ::call(a2) , hpx::util::detail::move_if_no_ref< A3> ::call(a3) , hpx::util::detail::move_if_no_ref< A4> ::call(a4) , hpx::util::detail::move_if_no_ref< A5> ::call(a5) , hpx::util::detail::move_if_no_ref< A6> ::call(a6) , hpx::util::detail::move_if_no_ref< A7> ::call(a7) , hpx::util::detail::move_if_no_ref< A8> ::call(a8) , hpx::util::detail::move_if_no_ref< A9> ::call(a9) , hpx::util::detail::move_if_no_ref< A10> ::call(a10) , hpx::util::detail::move_if_no_ref< A11> ::call(a11) , hpx::util::detail::move_if_no_ref< A12> ::call(a12) , hpx::util::detail::move_if_no_ref< A13> ::call(a13) , hpx::util::detail::move_if_no_ref< A14> ::call(a14)));
        }
        LRT_(info) << "successfully created component " << id
            << " of type: " << components::get_component_type_name(type);
        return id;
    }
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>
    struct create_component_action15
      : ::hpx::actions::result_action15<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
          , &runtime_support::create_component15<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14>
          , create_component_action15<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>
    struct create_component_direct_action15
      : ::hpx::actions::direct_result_action15<
            naming::gid_type (runtime_support::*)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
          , &runtime_support::create_component15<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14>
          , create_component_direct_action15<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14> >
    {};
}}}
namespace boost { namespace serialization { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>) ) struct guid_defined<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action15< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14>))>))> : mpl::true_ {}; namespace ext { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>) ) struct guid_impl<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action15< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14>))>))> { static inline const char * call() { return hpx::util::detail::type_hash<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action15< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14>))>))>(); } }; } } namespace archive { namespace detail { namespace extra_detail { HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>) ) struct init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action15< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14>))>))> { static hpx::util::detail::guid_initializer_helper< HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action15< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14>))>)) > const & g; }; HPX_UTIL_STRIP( (template <typename Component , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>) ) hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action15< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14>))>))> const & init_guid<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action15< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14>))>))>::g = ::boost::serialization::singleton< hpx::util::detail::guid_initializer_helper<HPX_UTIL_STRIP( (hpx::actions::transfer_action<HPX_UTIL_STRIP( (hpx::components::server::create_component_action15< Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14>))>))> >::get_mutable_instance().export_guid(); }}} }
