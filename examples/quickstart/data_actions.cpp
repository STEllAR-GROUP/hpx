//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/hpx.hpp>

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

    static bool is_target_valid(hpx::naming::id_type const& id) { return true; }

    /// This is the default hook implementation for decorate_action which
    /// does no hooking at all.
    template <typename F>
    static hpx::threads::thread_function_type
    decorate_action(hpx::naming::address::address_type, F && f)
    {
        return std::forward<F>(f);
    }

    /// This is the default hook implementation for schedule_thread which
    /// forwards to the default scheduler.
    static void schedule_thread(hpx::naming::address::address_type,
        hpx::threads::thread_init_data& data,
        hpx::threads::thread_state_enum initial_state)
    {
        hpx::threads::register_work_plain(data, initial_state);
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename T, typename Derived>
struct data_get_action_base
    : public hpx::actions::basic_action<plain_data<Derived>,
        typename boost::remove_pointer<T>::type(), Derived>
{};

template <typename T, T Data, typename Derived = hpx::actions::detail::this_type>
struct data_get_action
    : public data_get_action_base<
        typename boost::remove_pointer<T>::type,
        typename hpx::actions::detail::action_type<
            data_get_action<T, Data, Derived>, Derived
        >::type>
{
    typedef typename boost::add_reference<
        typename boost::add_const<
            typename boost::remove_pointer<T>::type
        >::type
    >::type result_type;

    typedef typename hpx::actions::detail::remote_action_result<
        typename boost::remove_pointer<T>::type
    >::type remote_result_type;

    typedef typename hpx::actions::detail::action_type<
        data_get_action, Derived
    >::type derived_type;

    typedef data_get_action_base<T, derived_type> base_type;

    typedef boost::mpl::false_ direct_execution;

protected:
    /// The \a thread_function will be registered as the thread
    /// function of a thread.
    BOOST_FORCEINLINE static hpx::threads::thread_state_enum
    thread_function(hpx::threads::thread_state_ex_enum)
    {
        // there isn't anything to do for us
        return hpx::threads::terminated;
    }

    // Return the referenced data
    static result_type get_value_function()
    {
        return *Data;
    }

public:
    /// \brief This static \a construct_thread_function allows to construct
    /// a proper thread function for a \a thread.
    ///
    /// This is used by in case no continuation has been supplied.
    template <typename Arguments>
    static hpx::threads::thread_function_type
    construct_thread_function(hpx::naming::address::address_type lva,
        Arguments && /*args*/)
    {
        return hpx::traits::action_decorate_function<derived_type>::call(lva,
            &data_get_action::thread_function);
    }

    /// \brief This static \a construct_thread_function allows to construct
    /// a proper thread function for a \a thread.
    ///
    /// This is used in case a continuation has been supplied
    template <typename Arguments>
    static hpx::threads::thread_function_type
    construct_thread_function(hpx::actions::continuation_type& cont,
        hpx::naming::address::address_type lva, Arguments && args)
    {
        return hpx::traits::action_decorate_function<derived_type>::call(lva,
            base_type::construct_continuation_thread_function(
                cont, &derived_type::get_value_function,
                std::forward<Arguments>(args)));
    }

    // direct execution
    template <typename Arguments>
    BOOST_FORCEINLINE static result_type
    execute_function(hpx::naming::address::address_type, Arguments && /*args*/)
    {
        return derived_type::get_value_function();
    }
};

///////////////////////////////////////////////////////////////////////////
template <typename T, typename Derived>
struct data_set_action_base
    : public hpx::actions::basic_action<plain_data<Derived>, 
        hpx::util::unused_type(T), Derived>
{};

template <typename T, T Data, typename Derived = hpx::actions::detail::this_type>
struct data_set_action
    : public data_set_action_base<
        typename boost::remove_pointer<T>::type,
        typename hpx::actions::detail::action_type<
            data_set_action<T, Data, Derived>, Derived
        >::type>
{
    typedef hpx::util::unused_type result_type;
    typedef hpx::util::unused_type remote_result_type;

    typedef typename hpx::actions::detail::action_type<
        data_set_action, Derived
    >::type derived_type;

    typedef data_set_action_base<T, derived_type> base_type;
    typedef typename boost::remove_pointer<T>::type data_type;

    typedef boost::mpl::false_ direct_execution;

protected:
    struct thread_function
    {
        typedef hpx::threads::thread_state_enum result_type;

        BOOST_FORCEINLINE result_type operator()(data_type const& data) const
        {
            *Data = data;
            return hpx::threads::terminated;
        }
    };

    // Return the referenced data
    static void set_value_function(data_type const& data)
    {
        *Data = data;
    }

public:
    /// \brief This static \a construct_thread_function allows to construct
    /// a proper thread function for a \a thread.
    ///
    /// This is used by in case no continuation has been supplied.
    template <typename Arguments>
    static hpx::threads::thread_function_type
    construct_thread_function(hpx::naming::address::address_type lva,
        Arguments && args)
    {
        return hpx::traits::action_decorate_function<derived_type>::call(lva,
            hpx::util::bind(
                hpx::util::one_shot(typename derived_type::thread_function()),
                hpx::util::get<0>(std::forward<Arguments>(args))));
    }

    /// \brief This static \a construct_thread_function allows to construct
    /// a proper thread function for a \a thread.
    ///
    /// This is used in case a continuation has been supplied
    template <typename Arguments>
    static hpx::threads::thread_function_type
    construct_thread_function(hpx::actions::continuation_type& cont,
        hpx::naming::address::address_type lva, Arguments && args)
    {
        return hpx::traits::action_decorate_function<derived_type>::call(lva,
            base_type::construct_continuation_thread_function_void(
                cont, &derived_type::set_value_function,
                std::forward<Arguments>(args)));
    }

    // direct execution
    template <typename Arguments>
    BOOST_FORCEINLINE static void
    execute_function(hpx::naming::address::address_type, Arguments && args)
    {
        derived_type::set_value_function(
            hpx::util::get<0>(std::forward<Arguments>(args)));
    }
};

///////////////////////////////////////////////////////////////////////////////
int data = 0;       // this variable is exposed using the actions below

typedef data_get_action<decltype(&data), &data> get_action;
typedef data_set_action<decltype(&data), &data> set_action;

HPX_DEFINE_GET_COMPONENT_TYPE(plain_data<get_action>);
HPX_DEFINE_GET_COMPONENT_TYPE(plain_data<set_action>);

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
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

