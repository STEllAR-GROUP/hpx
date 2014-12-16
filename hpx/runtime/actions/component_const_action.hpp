//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file component_const_action.hpp

#if !defined(HPX_RUNTIME_ACTIONS_COMPONENT_CONST_ACTION_MAR_02_2013_0417PM)
#define HPX_RUNTIME_ACTIONS_COMPONENT_CONST_ACTION_MAR_02_2013_0417PM

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic component action types allowing to hold a different
    //  number of arguments
    ///////////////////////////////////////////////////////////////////////////

    // zero argument version
    template <
        typename Component, typename R,
        R (Component::*F)() const, typename Derived>
    class component_base_action<R (Component::*)() const, F, Derived>
      : public basic_action<Component const, R(), Derived>
    {
    public:
        typedef basic_action<Component const, R(), Derived> base_type;

        // Let the component decide whether the id is valid
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }

    protected:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func), while ignoring the return
        /// value.
        template <typename Address>   // dummy template parameter
        BOOST_FORCEINLINE static threads::thread_state_enum
        thread_function(Address lva)
        {
            try {
                LTM_(debug) << "Executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component const>::call(lva)) << ")";
                (get_lva<Component const>::call(lva)->*F)();      // just call the function
            }
            catch (hpx::thread_interrupted const&) { //-V565
                /* swallow this exception */
            }
            catch (hpx::exception const& e) {
                LTM_(error)
                    << "Unhandled exception while executing component action("
                    << detail::get_action_name<Derived>()
                    << ") lva(" << reinterpret_cast<void const*>
                        (get_lva<Component const>::call(lva)) << "): " << e.what();

                // report this error to the console in any case
                hpx::report_error(boost::current_exception());
            }
            catch (...) {
                LTM_(error)
                    << "Unhandled exception while executing component action("
                    << detail::get_action_name<Derived>()
                    << ") lva(" << reinterpret_cast<void const*>
                        (get_lva<Component const>::call(lva)) << ")";

                // report this error to the console in any case
                hpx::report_error(boost::current_exception());
            }

            // Verify that there are no more registered locks for this
            // OS-thread. This will throw if there are still any locks
            // held.
            util::force_error_on_lock();
            return threads::terminated;
        }

    public:
        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the \a component_base_action type. This is used by the \a
        /// applier in case no continuation has been supplied.
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && /*args*/)
        {
            threads::thread_state_enum (*f)(naming::address::address_type) =
                &Derived::template thread_function<naming::address::address_type>;

            return traits::action_decorate_function<Derived>::call(
                lva, util::bind(f, lva));
        }

        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the \a component_base_action type. This is used by the \a
        /// applier in case a continuation has been supplied
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }

        // direct execution
        template <typename Arguments>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Arguments &&)
        {
            LTM_(debug)
                << "component_base_action::execute_function: name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";

            return (get_lva<Component const>::call(lva)->*F)();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename R,
        R (Component::*F)() const, typename Derived>
    struct component_action<R (Component::*)() const, F, Derived>
      : component_base_action<
            R (Component::*)() const, F,
            typename detail::action_type<
                component_action<R (Component::*)() const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_action, Derived
        >::type derived_type;

        typedef boost::mpl::false_ direct_execution;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename R,
        R (Component::*F)() const, typename Derived>
    struct make_action<R (Component::*)() const, F, Derived, boost::mpl::false_>
      : component_action<R (Component::*)() const, F, Derived>
    {
        typedef component_action<
            R (Component::*)() const, F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename R,
        R (Component::*F)() const, typename Derived>
    struct component_direct_action<R (Component::*)() const, F, Derived>
      : public component_base_action<
            R (Component::*)() const, F,
            typename detail::action_type<
                component_direct_action<R (Component::*)() const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            component_direct_action, Derived
        >::type derived_type;

        typedef boost::mpl::true_ direct_execution;

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        static base_action::action_type get_action_type()
        {
            return base_action::direct_action;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename R,
        R (Component::*F)() const, typename Derived>
    struct make_action<R (Component::*)() const, F, Derived, boost::mpl::true_>
      : component_direct_action<R (Component::*)() const, F, Derived>
    {
        typedef component_direct_action<
            R (Component::*)() const, F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    //  zero parameter version, no result value
    template <typename Component, void (Component::*F)() const, typename Derived>
    class component_base_action<void (Component::*)() const, F, Derived>
      : public basic_action<Component const, util::unused_type(), Derived>
    {
    public:
        typedef basic_action<Component const, util::unused_type(), Derived> base_type;

        // Let the component decide whether the id is valid
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }

    protected:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func), while ignoring the return
        /// value.
        template <typename Address>   // dummy template parameter
        BOOST_FORCEINLINE static threads::thread_state_enum
        thread_function(Address lva)
        {
            try {
                LTM_(debug) << "Executing component action("
                            << detail::get_action_name<Derived>()
                            << ") lva(" << reinterpret_cast<void const*>
                                (get_lva<Component const>::call(lva)) << ")";
                (get_lva<Component const>::call(lva)->*F)();      // just call the function
            }
            catch (hpx::thread_interrupted const&) { //-V565
                /* swallow this exception */
            }
            catch (hpx::exception const& e) {
                LTM_(error)
                    << "Unhandled exception while executing component action("
                    << detail::get_action_name<Derived>()
                    << ") lva(" << reinterpret_cast<void const*>
                        (get_lva<Component const>::call(lva)) << "): " << e.what();

                // report this error to the console in any case
                hpx::report_error(boost::current_exception());
            }
            catch (...) {
                LTM_(error)
                    << "Unhandled exception while executing component action("
                    << detail::get_action_name<Derived>()
                    << ") lva(" << reinterpret_cast<void const*>
                        (get_lva<Component const>::call(lva)) << ")";

                // report this error to the console in any case
                hpx::report_error(boost::current_exception());
            }

            // Verify that there are no more registered locks for this
            // OS-thread. This will throw if there are still any locks
            // held.
            util::force_error_on_lock();
            return threads::terminated;
        }

    public:
        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the component_base_action type. This is used by the \a applier in
        /// case no continuation has been supplied.
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Arguments && /*args*/)
        {
            threads::thread_state_enum (*f)(naming::address::address_type) =
                &Derived::template thread_function<naming::address::address_type>;

            return traits::action_decorate_function<Derived>::call(
                lva, util::bind(f, lva));
        }

        /// \brief This static \a construct_thread_function allows to construct
        /// a proper thread function for a \a thread without having to
        /// instantiate the component_base_action type. This is used by the \a applier in
        /// case a continuation has been supplied
        template <typename Arguments>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Arguments && args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Arguments>(args)));
        }

        // direct execution
        template <typename Arguments>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Arguments &&)
        {
            LTM_(debug)
                << "component_base_action::execute_function: name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)();
            return util::unused;
        }
    };

    /// \endcond
}}

/// \endcond

#include <hpx/config/warnings_suffix.hpp>

#endif

