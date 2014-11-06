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
                << " (component type not found in map)";
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
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor1<
                    typename Component::wrapping_type,
                    A0>(
                        std::forward<A0>( a0 )));
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
    
    template <typename Component
       >
    struct bulk_create_component_action1
      : ::hpx::actions::result_action1<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t )
          , &runtime_support::bulk_create_component1<
                Component >
          , bulk_create_component_action1<
                Component > >
    {};
    template <typename Component
       >
    struct bulk_create_component_direct_action1
      : ::hpx::actions::direct_result_action1<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t )
          , &runtime_support::bulk_create_component1<
                Component >
          , bulk_create_component_direct_action1<
                Component > >
    {};
}}}
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
                << " (component type not found in map)";
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
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor2<
                    typename Component::wrapping_type,
                    A0 , A1>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 )));
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
    template <typename Component, typename A0>
    std::vector<naming::gid_type>
    runtime_support::bulk_create_component2(std::size_t count,
        A0 a0)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component(s) instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component type not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance(s) of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        std::vector<naming::gid_type> ids;
        ids.reserve(count);
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            for (std::size_t i = 0; i != count; ++i)
            {
                ids.push_back(factory->create_with_args(
                    component_constructor_functor1<
                        typename Component::wrapping_type,
                        A0
                    >(std::forward<A0>( a0 )))
                );
            }
        }
        LRT_(info) << "successfully created " << count
                   << " component(s) of type: "
                   << components::get_component_type_name(type);
        return std::move(ids);
    }
    
    template <typename Component
      , typename A0>
    struct bulk_create_component_action2
      : ::hpx::actions::result_action2<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0)
          , &runtime_support::bulk_create_component2<
                Component , A0>
          , bulk_create_component_action2<
                Component , A0> >
    {};
    template <typename Component
      , typename A0>
    struct bulk_create_component_direct_action2
      : ::hpx::actions::direct_result_action2<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0)
          , &runtime_support::bulk_create_component2<
                Component , A0>
          , bulk_create_component_direct_action2<
                Component , A0> >
    {};
}}}
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
                << " (component type not found in map)";
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
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor3<
                    typename Component::wrapping_type,
                    A0 , A1 , A2>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 )));
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
    template <typename Component, typename A0 , typename A1>
    std::vector<naming::gid_type>
    runtime_support::bulk_create_component3(std::size_t count,
        A0 a0 , A1 a1)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component(s) instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component type not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance(s) of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        std::vector<naming::gid_type> ids;
        ids.reserve(count);
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            for (std::size_t i = 0; i != count; ++i)
            {
                ids.push_back(factory->create_with_args(
                    component_constructor_functor2<
                        typename Component::wrapping_type,
                        A0 , A1
                    >(std::forward<A0>( a0 ) , std::forward<A1>( a1 )))
                );
            }
        }
        LRT_(info) << "successfully created " << count
                   << " component(s) of type: "
                   << components::get_component_type_name(type);
        return std::move(ids);
    }
    
    template <typename Component
      , typename A0 , typename A1>
    struct bulk_create_component_action3
      : ::hpx::actions::result_action3<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1)
          , &runtime_support::bulk_create_component3<
                Component , A0 , A1>
          , bulk_create_component_action3<
                Component , A0 , A1> >
    {};
    template <typename Component
      , typename A0 , typename A1>
    struct bulk_create_component_direct_action3
      : ::hpx::actions::direct_result_action3<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1)
          , &runtime_support::bulk_create_component3<
                Component , A0 , A1>
          , bulk_create_component_direct_action3<
                Component , A0 , A1> >
    {};
}}}
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
                << " (component type not found in map)";
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
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor4<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 )));
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
    template <typename Component, typename A0 , typename A1 , typename A2>
    std::vector<naming::gid_type>
    runtime_support::bulk_create_component4(std::size_t count,
        A0 a0 , A1 a1 , A2 a2)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component(s) instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component type not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance(s) of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        std::vector<naming::gid_type> ids;
        ids.reserve(count);
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            for (std::size_t i = 0; i != count; ++i)
            {
                ids.push_back(factory->create_with_args(
                    component_constructor_functor3<
                        typename Component::wrapping_type,
                        A0 , A1 , A2
                    >(std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 )))
                );
            }
        }
        LRT_(info) << "successfully created " << count
                   << " component(s) of type: "
                   << components::get_component_type_name(type);
        return std::move(ids);
    }
    
    template <typename Component
      , typename A0 , typename A1 , typename A2>
    struct bulk_create_component_action4
      : ::hpx::actions::result_action4<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1 , A2)
          , &runtime_support::bulk_create_component4<
                Component , A0 , A1 , A2>
          , bulk_create_component_action4<
                Component , A0 , A1 , A2> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2>
    struct bulk_create_component_direct_action4
      : ::hpx::actions::direct_result_action4<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1 , A2)
          , &runtime_support::bulk_create_component4<
                Component , A0 , A1 , A2>
          , bulk_create_component_direct_action4<
                Component , A0 , A1 , A2> >
    {};
}}}
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
                << " (component type not found in map)";
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
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor5<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 )));
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
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3>
    std::vector<naming::gid_type>
    runtime_support::bulk_create_component5(std::size_t count,
        A0 a0 , A1 a1 , A2 a2 , A3 a3)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component(s) instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component type not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance(s) of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        std::vector<naming::gid_type> ids;
        ids.reserve(count);
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            for (std::size_t i = 0; i != count; ++i)
            {
                ids.push_back(factory->create_with_args(
                    component_constructor_functor4<
                        typename Component::wrapping_type,
                        A0 , A1 , A2 , A3
                    >(std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 )))
                );
            }
        }
        LRT_(info) << "successfully created " << count
                   << " component(s) of type: "
                   << components::get_component_type_name(type);
        return std::move(ids);
    }
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3>
    struct bulk_create_component_action5
      : ::hpx::actions::result_action5<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1 , A2 , A3)
          , &runtime_support::bulk_create_component5<
                Component , A0 , A1 , A2 , A3>
          , bulk_create_component_action5<
                Component , A0 , A1 , A2 , A3> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3>
    struct bulk_create_component_direct_action5
      : ::hpx::actions::direct_result_action5<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1 , A2 , A3)
          , &runtime_support::bulk_create_component5<
                Component , A0 , A1 , A2 , A3>
          , bulk_create_component_direct_action5<
                Component , A0 , A1 , A2 , A3> >
    {};
}}}
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
                << " (component type not found in map)";
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
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor6<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4 , A5>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 )));
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
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    std::vector<naming::gid_type>
    runtime_support::bulk_create_component6(std::size_t count,
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component(s) instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component type not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance(s) of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        std::vector<naming::gid_type> ids;
        ids.reserve(count);
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            for (std::size_t i = 0; i != count; ++i)
            {
                ids.push_back(factory->create_with_args(
                    component_constructor_functor5<
                        typename Component::wrapping_type,
                        A0 , A1 , A2 , A3 , A4
                    >(std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 )))
                );
            }
        }
        LRT_(info) << "successfully created " << count
                   << " component(s) of type: "
                   << components::get_component_type_name(type);
        return std::move(ids);
    }
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    struct bulk_create_component_action6
      : ::hpx::actions::result_action6<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1 , A2 , A3 , A4)
          , &runtime_support::bulk_create_component6<
                Component , A0 , A1 , A2 , A3 , A4>
          , bulk_create_component_action6<
                Component , A0 , A1 , A2 , A3 , A4> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    struct bulk_create_component_direct_action6
      : ::hpx::actions::direct_result_action6<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1 , A2 , A3 , A4)
          , &runtime_support::bulk_create_component6<
                Component , A0 , A1 , A2 , A3 , A4>
          , bulk_create_component_direct_action6<
                Component , A0 , A1 , A2 , A3 , A4> >
    {};
}}}
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
                << " (component type not found in map)";
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
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor7<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4 , A5 , A6>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 )));
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
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    std::vector<naming::gid_type>
    runtime_support::bulk_create_component7(std::size_t count,
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component(s) instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component type not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance(s) of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        std::vector<naming::gid_type> ids;
        ids.reserve(count);
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            for (std::size_t i = 0; i != count; ++i)
            {
                ids.push_back(factory->create_with_args(
                    component_constructor_functor6<
                        typename Component::wrapping_type,
                        A0 , A1 , A2 , A3 , A4 , A5
                    >(std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 )))
                );
            }
        }
        LRT_(info) << "successfully created " << count
                   << " component(s) of type: "
                   << components::get_component_type_name(type);
        return std::move(ids);
    }
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    struct bulk_create_component_action7
      : ::hpx::actions::result_action7<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1 , A2 , A3 , A4 , A5)
          , &runtime_support::bulk_create_component7<
                Component , A0 , A1 , A2 , A3 , A4 , A5>
          , bulk_create_component_action7<
                Component , A0 , A1 , A2 , A3 , A4 , A5> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    struct bulk_create_component_direct_action7
      : ::hpx::actions::direct_result_action7<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1 , A2 , A3 , A4 , A5)
          , &runtime_support::bulk_create_component7<
                Component , A0 , A1 , A2 , A3 , A4 , A5>
          , bulk_create_component_direct_action7<
                Component , A0 , A1 , A2 , A3 , A4 , A5> >
    {};
}}}
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
                << " (component type not found in map)";
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
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor8<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 )));
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
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    std::vector<naming::gid_type>
    runtime_support::bulk_create_component8(std::size_t count,
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component(s) instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component type not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance(s) of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        std::vector<naming::gid_type> ids;
        ids.reserve(count);
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            for (std::size_t i = 0; i != count; ++i)
            {
                ids.push_back(factory->create_with_args(
                    component_constructor_functor7<
                        typename Component::wrapping_type,
                        A0 , A1 , A2 , A3 , A4 , A5 , A6
                    >(std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 )))
                );
            }
        }
        LRT_(info) << "successfully created " << count
                   << " component(s) of type: "
                   << components::get_component_type_name(type);
        return std::move(ids);
    }
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    struct bulk_create_component_action8
      : ::hpx::actions::result_action8<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1 , A2 , A3 , A4 , A5 , A6)
          , &runtime_support::bulk_create_component8<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6>
          , bulk_create_component_action8<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    struct bulk_create_component_direct_action8
      : ::hpx::actions::direct_result_action8<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1 , A2 , A3 , A4 , A5 , A6)
          , &runtime_support::bulk_create_component8<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6>
          , bulk_create_component_direct_action8<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6> >
    {};
}}}
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
                << " (component type not found in map)";
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
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor9<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 )));
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
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    std::vector<naming::gid_type>
    runtime_support::bulk_create_component9(std::size_t count,
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component(s) instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component type not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance(s) of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        std::vector<naming::gid_type> ids;
        ids.reserve(count);
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            for (std::size_t i = 0; i != count; ++i)
            {
                ids.push_back(factory->create_with_args(
                    component_constructor_functor8<
                        typename Component::wrapping_type,
                        A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7
                    >(std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 )))
                );
            }
        }
        LRT_(info) << "successfully created " << count
                   << " component(s) of type: "
                   << components::get_component_type_name(type);
        return std::move(ids);
    }
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    struct bulk_create_component_action9
      : ::hpx::actions::result_action9<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
          , &runtime_support::bulk_create_component9<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>
          , bulk_create_component_action9<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    struct bulk_create_component_direct_action9
      : ::hpx::actions::direct_result_action9<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
          , &runtime_support::bulk_create_component9<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>
          , bulk_create_component_direct_action9<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7> >
    {};
}}}
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
                << " (component type not found in map)";
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
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            id = factory->create_with_args(
                component_constructor_functor10<
                    typename Component::wrapping_type,
                    A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>(
                        std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 )));
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
    template <typename Component, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    std::vector<naming::gid_type>
    runtime_support::bulk_create_component10(std::size_t count,
        A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8)
    {
        components::component_type const type =
            components::get_component_type<
                typename Component::wrapped_type>();
        component_map_mutex_type::scoped_lock l(cm_mtx_);
        component_map_type::const_iterator it = components_.find(type);
        if (it == components_.end()) {
            hpx::util::osstream strm;
            strm << "attempt to create component(s) instance of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (component type not found in map)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        if (!(*it).second.first) {
            hpx::util::osstream strm;
            strm << "attempt to create component instance(s) of "
                << "invalid/unknown type: "
                << components::get_component_type_name(type)
                << " (map entry is NULL)";
            HPX_THROW_EXCEPTION(hpx::bad_component_type,
                "runtime_support::create_component",
                hpx::util::osstream_get_string(strm));
            return std::vector<naming::gid_type>();
        }
        std::vector<naming::gid_type> ids;
        ids.reserve(count);
        boost::shared_ptr<component_factory_base> factory((*it).second.first);
        {
            util::scoped_unlock<component_map_mutex_type::scoped_lock> ul(l);
            for (std::size_t i = 0; i != count; ++i)
            {
                ids.push_back(factory->create_with_args(
                    component_constructor_functor9<
                        typename Component::wrapping_type,
                        A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8
                    >(std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 )))
                );
            }
        }
        LRT_(info) << "successfully created " << count
                   << " component(s) of type: "
                   << components::get_component_type_name(type);
        return std::move(ids);
    }
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    struct bulk_create_component_action10
      : ::hpx::actions::result_action10<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
          , &runtime_support::bulk_create_component10<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>
          , bulk_create_component_action10<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8> >
    {};
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    struct bulk_create_component_direct_action10
      : ::hpx::actions::direct_result_action10<
            std::vector<naming::gid_type> (runtime_support::*)(
                std::size_t , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
          , &runtime_support::bulk_create_component10<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>
          , bulk_create_component_direct_action10<
                Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8> >
    {};
}}}
