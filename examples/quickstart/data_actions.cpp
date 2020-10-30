//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/hpx.hpp>

#include <iostream>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
/// placeholder type allowing to integrate the data action templates below
/// with the existing component based action template infrastructure
template <typename Action>
struct plain_data
{
    static hpx::components::component_type get_component_type()
    {
        return hpx::components::get_component_type<plain_data<Action> >();
    }
    static void set_component_type(hpx::components::component_type type)
    {
        hpx::components::set_component_type<plain_data<Action> >(type);
    }

    static bool is_target_valid(hpx::naming::id_type const&)
    {
        return true;
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename T, typename Derived>
struct data_get_action_base
    : public hpx::actions::basic_action<plain_data<Derived>,
        typename std::remove_pointer<T>::type(), Derived>
{};

template <typename T, T Data, typename Derived = hpx::actions::detail::this_type>
struct data_get_action
    : public data_get_action_base<
        typename std::remove_pointer<T>::type,
        typename hpx::actions::detail::action_type<
            data_get_action<T, Data, Derived>, Derived
        >::type>
{
    typedef std::false_type direct_execution;

    typedef typename std::remove_pointer<T>::type data_type;

    // Return the referenced data
    static data_type invoke(
        hpx::naming::address::address_type /*lva*/,
        hpx::naming::address::component_type /*comptype*/)
    {
        return *Data;
    }
};

///////////////////////////////////////////////////////////////////////////
template <typename T, typename Derived>
struct data_set_action_base
    : public hpx::actions::basic_action<plain_data<Derived>,
        void(T), Derived>
{};

template <typename T, T Data, typename Derived = hpx::actions::detail::this_type>
struct data_set_action
    : public data_set_action_base<
        typename std::remove_pointer<T>::type,
        typename hpx::actions::detail::action_type<
            data_set_action<T, Data, Derived>, Derived
        >::type>
{
    typedef typename std::remove_pointer<T>::type data_type;

    typedef std::false_type direct_execution;

    // Return the referenced data
    static void invoke(
        hpx::naming::address::address_type /*lva*/,
        hpx::naming::address::component_type /*comptype*/,
        data_type const& data)
    {
        *Data = data;
    }
};

///////////////////////////////////////////////////////////////////////////////
int data = 0;       // this variable is exposed using the actions below

typedef data_get_action<decltype(&data), &data> get_action;
typedef data_set_action<decltype(&data), &data> set_action;

HPX_DEFINE_GET_COMPONENT_TYPE(plain_data<get_action>);
HPX_DEFINE_GET_COMPONENT_TYPE(plain_data<set_action>);

///////////////////////////////////////////////////////////////////////////////
int main()
{
    data = 0;

    set_action set;
    hpx::future<void> f1 = hpx::async(set, hpx::find_here(), 42);

    f1.get();

    get_action get;
    hpx::future<int> f2 = hpx::async(get, hpx::find_here());

    std::cout << f2.get() << std::endl;

    return 0;
}

#endif
