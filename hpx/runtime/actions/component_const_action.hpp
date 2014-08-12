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
        typename Component, typename Result,
        Result (Component::*F)() const, typename Derived>
    class base_result_action0<Result (Component::*)() const, F, Derived>
      : public action<Component const, Result, hpx::util::tuple<>, Derived>
    {
    public:
        typedef Result result_type;
        typedef typename detail::remote_action_result<Result>::type
            remote_result_type;

        typedef hpx::util::tuple<> arguments_type;
        typedef action<Component const, result_type, arguments_type, Derived>
            base_type;

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
        /// instantiate the \a base_result_action0 type. This is used by the \a
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
        /// instantiate the \a base_result_action0 type. This is used by the \a
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
        BOOST_FORCEINLINE static Result
        execute_function(naming::address::address_type lva,
            Arguments &&)
        {
            LTM_(debug)
                << "base_result_action0::execute_function: name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";

            return (get_lva<Component const>::call(lva)->*F)();
        }
    };

#if BOOST_WORKAROUND(BOOST_MSVC, == 1600)
    namespace detail
    {
        template <typename Obj, typename Result>
        struct synthesize_const_mf<Obj, Result (*)()>
        {
            typedef Result (Obj::*type)() const;
        };

        template <typename Obj, typename Result>
        struct synthesize_const_mf<Obj, Result (Obj::*)() const>
        {
            typedef Result (Obj::*type)() const;
        };

        template <typename Result>
        typename boost::mpl::identity<Result (*)()>::type
        replicate_type(Result (*p)());
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Result,
        Result (Component::*F)() const, typename Derived>
    struct result_action0<Result (Component::*)() const, F, Derived>
      : base_result_action0<
            Result (Component::*)() const, F,
            typename detail::action_type<
                result_action0<Result (Component::*)() const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            result_action0, Derived
        >::type derived_type;

        typedef boost::mpl::false_ direct_execution;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result,
        Result (Component::*F)() const, typename Derived>
    struct make_action<Result (Component::*)() const, F, Derived, boost::mpl::false_>
      : result_action0<Result (Component::*)() const, F, Derived>
    {
        typedef result_action0<
            Result (Component::*)() const, F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Result,
        Result (Component::*F)() const, typename Derived>
    struct direct_result_action0<Result (Component::*)() const, F, Derived>
      : public base_result_action0<
            Result (Component::*)() const, F,
            typename detail::action_type<
                direct_result_action0<Result (Component::*)() const, F, Derived>,
                Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_result_action0, Derived
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
        typename Component, typename Result,
        Result (Component::*F)() const, typename Derived>
    struct make_action<Result (Component::*)() const, F, Derived, boost::mpl::true_>
      : direct_result_action0<Result (Component::*)() const, F, Derived>
    {
        typedef direct_result_action0<
            Result (Component::*)() const, F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    //  zero parameter version, no result value
    template <typename Component, void (Component::*F)() const, typename Derived>
    class base_action0<void (Component::*)() const, F, Derived>
      : public action<Component const, util::unused_type,
            hpx::util::tuple<>, Derived>
    {
    public:
        typedef util::unused_type result_type;
        typedef util::unused_type remote_result_type;

        typedef hpx::util::tuple<> arguments_type;
        typedef action<Component const, result_type, arguments_type, Derived>
            base_type;

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
        /// instantiate the base_action0 type. This is used by the \a applier in
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
        /// instantiate the base_action0 type. This is used by the \a applier in
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
                << "base_action0::execute_function: name("
                << detail::get_action_name<Derived>()
                << ") lva(" << reinterpret_cast<void const*>(
                    get_lva<Component const>::call(lva)) << ")";
            (get_lva<Component const>::call(lva)->*F)();
            return util::unused;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, void (Component::*F)() const, typename Derived>
    struct action0<void (Component::*)() const, F, Derived>
      : base_action0<
            void (Component::*)() const, F,
            typename detail::action_type<
                action0<void (Component::*)() const, F, Derived>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            action0, Derived
        >::type derived_type;

        typedef boost::mpl::false_ direct_execution;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, void (Component::*F)() const, typename Derived>
    struct make_action<void (Component::*)() const, F, Derived, boost::mpl::false_>
      : action0<void (Component::*)() const, F, Derived>
    {
        typedef action0<
            void (Component::*)() const, F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, void (Component::*F)() const, typename Derived>
    struct direct_action0<void (Component::*)() const, F, Derived>
      : base_action0<
            void (Component::*)() const, F,
            typename detail::action_type<
                direct_action0<void (Component::*)() const, F, Derived>, Derived
            >::type>
    {
        typedef typename detail::action_type<
            direct_action0, Derived
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
    template <typename Component, void (Component::*F)() const, typename Derived>
    struct make_action<void (Component::*)() const, F, Derived, boost::mpl::true_>
      : direct_action0<void (Component::*)() const, F, Derived>
    {
        typedef direct_action0<
            void (Component::*)() const, F, Derived
        > type;
    };

    ///////////////////////////////////////////////////////////////////////////
    // the specialization for void return type is just a template alias
    template <typename Component, void (Component::*F)() const, typename Derived>
    struct result_action0<void (Component::*)() const, F, Derived>
      : action0<void (Component::*)() const, F, Derived>
    {};

    /// \endcond
}}

/// \endcond

#include <hpx/config/warnings_suffix.hpp>

#endif

