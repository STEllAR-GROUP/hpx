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

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, with result
    template <
        typename Component, typename R, typename ...Ps,
        typename TF, TF F, typename Derived>
    class basic_action_impl<
            R (Component::*)(Ps...) const, TF, F, Derived>
      : public basic_action<Component const, R(Ps...), Derived>
    {
    public:
        typedef basic_action<Component const, R(Ps...), Derived> base_type;

        // Let the component decide whether the id is valid
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }

    protected:
        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func).
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;

            template <typename ...Ts>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva, Ts&&... vs) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";

                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Ts>(vs)...);
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
        };

    public:
        typedef boost::mpl::false_ direct_execution;

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the basic_action_impl type. This is used by the
        // applier in case no continuation has been supplied.
        template <std::size_t ...Is, typename Args>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            util::detail::pack_c<std::size_t, Is...>, Args&& args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()),
                    lva, util::get<Is>(std::forward<Args>(args))...));
        }

        template <typename Args>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Args&& args)
        {
            return construct_thread_function(lva,
                typename util::detail::make_index_pack<
                    util::tuple_size<typename util::decay<Args>::type>::value
                >::type(), std::forward<Args>(args));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the basic_action_impl type. This is used by the
        // applier in case a continuation has been supplied
        template <typename Args>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Args&& args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Args>(args)));
        }

        // direct execution
        template <std::size_t ...Is, typename Args>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            util::detail::pack_c<std::size_t, Is...>, Args&& args)
        {
            LTM_(debug)
                << "basic_action_impl::execute_function name("
                << detail::get_action_name<Derived>() << ")";

            return (get_lva<Component>::call(lva)->*F)(
                util::get<Is>(std::forward<Args>(args))...);
        }

        template <typename Args>
        BOOST_FORCEINLINE static R
        execute_function(naming::address::address_type lva,
            Args&& args)
        {
            return execute_function(lva,
                typename util::detail::make_index_pack<
                    util::tuple_size<typename util::decay<Args>::type>::value
                >::type(), std::forward<Args>(args));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version, no result type
    template <typename Component, typename ...Ps,
        typename TF, TF F, typename Derived>
    class basic_action_impl<
            void (Component::*)(Ps...) const, TF, F, Derived>
      : public basic_action<Component const, util::unused_type(Ps...), Derived>
    {
    public:
        typedef basic_action<
            Component const, util::unused_type(Ps...), Derived> base_type;

        // Let the component decide whether the id is valid
        static bool is_target_valid(naming::id_type const& id)
        {
            return Component::is_target_valid(id);
        }

    protected:
        /// The \a thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func).
        struct thread_function
        {
            typedef threads::thread_state_enum result_type;

            template <typename ...Ts>
            BOOST_FORCEINLINE result_type operator()(
                naming::address::address_type lva, Ts&&... vs) const
            {
                try {
                    LTM_(debug) << "Executing component action("
                                << detail::get_action_name<Derived>()
                                << ") lva(" << reinterpret_cast<void const*>
                                    (get_lva<Component const>::call(lva)) << ")";

                    (get_lva<Component const>::call(lva)->*F)(
                        std::forward<Ts>(vs)...);
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
        };

    public:
        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the basic_action_impl type. This is used by the applier in
        // case no continuation has been supplied.
        template <std::size_t ...Is, typename Args>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            util::detail::pack_c<std::size_t, Is...>, Args&& args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                util::bind(util::one_shot(typename Derived::thread_function()),
                    lva, util::get<Is>(std::forward<Args>(args))...));
        }

        template <typename Args>
        static threads::thread_function_type
        construct_thread_function(naming::address::address_type lva,
            Args&& args)
        {
            return construct_thread_function(lva,
                typename util::detail::make_index_pack<
                    util::tuple_size<typename util::decay<Args>::type>::value
                >::type(), std::forward<Args>(args));
        }

        // This static construct_thread_function allows to construct
        // a proper thread function for a thread without having to
        // instantiate the basic_action_impl type. This is used by the applier in
        // case a continuation has been supplied
        template <typename Args>
        static threads::thread_function_type
        construct_thread_function(continuation_type& cont,
            naming::address::address_type lva, Args&& args)
        {
            return traits::action_decorate_function<Derived>::call(lva,
                base_type::construct_continuation_thread_object_function_void(
                    cont, F, get_lva<Component const>::call(lva),
                    std::forward<Args>(args)));
        }

        // direct execution
        template <std::size_t ...Is, typename Args>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            util::detail::pack_c<std::size_t, Is...>, Args&& args)
        {
            LTM_(debug)
                << "basic_action_impl::execute_function name("
                << detail::get_action_name<Derived>() << ")";

            (get_lva<Component>::call(lva)->*F)(
                util::get<Is>(std::forward<Args>(args))...);
            return util::unused;
        }

        template <typename Args>
        BOOST_FORCEINLINE static util::unused_type
        execute_function(naming::address::address_type lva,
            Args&& args)
        {
            return execute_function(lva,
                typename util::detail::make_index_pack<
                    util::tuple_size<typename util::decay<Args>::type>::value
                >::type(), std::forward<Args>(args));
        }
    };

    /// \endcond
}}

#include <hpx/config/warnings_suffix.hpp>

#endif

