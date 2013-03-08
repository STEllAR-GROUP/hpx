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
