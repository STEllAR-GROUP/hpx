//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file action_support.hpp

#if !defined(HPX_RUNTIME_ACTIONS_ACTION_SUPPORT_NOV_14_2008_0711PM)
#define HPX_RUNTIME_ACTIONS_ACTION_SUPPORT_NOV_14_2008_0711PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/util/move.hpp>

#include <boost/version.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/ref.hpp>
#include <boost/foreach.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <boost/serialization/extended_type_info_typeid.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/and.hpp>
#if defined(BOOST_NO_DECLTYPE)
#  include <boost/typeof/typeof.hpp>
#endif
#include <boost/utility/enable_if.hpp>

#include <hpx/traits/needs_guid_initialization.hpp>
#include <hpx/runtime/get_lva.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/serialize_sequence.hpp>
#include <hpx/util/serialize_exception.hpp>
#include <hpx/util/demangle_helper.hpp>
#include <hpx/util/base_object.hpp>
#include <hpx/util/void_cast.hpp>
#include <hpx/util/register_locks.hpp>

#include <hpx/config/bind.hpp>
#include <hpx/config/tuple.hpp>
#include <hpx/config/function.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // Helper template meta function removing any 'const' qualifier or
        // reference from the given type (i.e. const& T --> T)
        template <typename T>
        struct remove_qualifiers
        {
            typedef typename hpx::util::detail::remove_reference<T>::type no_ref_type;
            typedef typename boost::remove_const<no_ref_type>::type type;
        };

        template <typename Action>
        char const* get_action_name()
        {
            /// If you encounter this assert while compiling code, that means that
            /// you have a HPX_REGISTER_ACTION macro somewhere in a source file,
            /// but the header in which the action is defined misses a
            /// HPX_REGISTER_ACTION_DECLARATION
            BOOST_MPL_ASSERT_MSG(
                traits::needs_guid_initialization<Action>::value
              , HPX_REGISTER_ACTION_DECLARATION_MISSING
              , (Action)
            );
            return util::type_id<Action>::typeid_.type_id();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The \a base_action class is an abstract class used as the base class
    /// for all action types. It's main purpose is to allow polymorphic
    /// serialization of action instances through a shared_ptr.
    struct base_action
    {
        /// The type of an action defines whether this action will be executed
        /// directly or by a PX-threads
        enum action_type
        {
            plain_action = 0,   ///< The action will be executed by a newly created thread
            direct_action = 1   ///< The action needs to be executed directly
        };

        /// Destructor
        virtual ~base_action() {}

        /// The function \a get_component_type returns the \a component_type
        /// of the component this action belongs to.
        virtual int get_component_type() const = 0;

        /// The function \a get_action_code returns the code of the action
        /// instance it is called for.
        virtual std::size_t get_action_code() const = 0;

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        virtual char const* get_action_name() const = 0;

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        virtual action_type get_action_type() const = 0;

        /// The \a get_thread_function constructs a proper thread function for
        /// a \a thread, encapsulating the functionality and the arguments
        /// of the action it is called for.
        ///
        /// \param lva    [in] This is the local virtual address of the
        ///               component the action has to be invoked on.
        ///
        /// \returns      This function returns a proper thread function usable
        ///               for a \a thread.
        ///
        /// \note This \a get_thread_function will be invoked to retrieve the
        ///       thread function for an action which has to be invoked without
        ///       continuations.
        virtual HPX_STD_FUNCTION<threads::thread_function_type>
            get_thread_function(naming::address::address_type lva) = 0;

        /// The \a get_thread_function constructs a proper thread function for
        /// a \a thread, encapsulating the functionality, the arguments, and
        /// the continuations of the action it is called for.
        ///
        /// \param cont   [in] This is the list of continuations to be
        ///               triggered after the execution of the action
        /// \param lva    [in] This is the local virtual address of the
        ///               component the action has to be invoked on.
        ///
        /// \returns      This function returns a proper thread function usable
        ///               for a \a thread.
        ///
        /// \note This \a get_thread_function will be invoked to retrieve the
        ///       thread function for an action which has to be invoked with
        ///       continuations.
        virtual HPX_STD_FUNCTION<threads::thread_function_type>
            get_thread_function(continuation_type& cont,
                naming::address::address_type lva) = 0;

        /// return the id of the locality of the parent thread
        virtual boost::uint32_t get_parent_locality_id() const = 0;

        /// Return the thread id of the parent thread
        virtual threads::thread_id_type get_parent_thread_id() const = 0;

        /// Return the thread phase of the parent thread
        virtual std::size_t get_parent_thread_phase() const = 0;

        /// Return the thread priority this action has to be executed with
        virtual threads::thread_priority get_thread_priority() const = 0;

        /// Return all data needed for thread initialization
        virtual threads::thread_init_data&
        get_thread_init_data(naming::address::address_type lva,
            threads::thread_init_data& data) = 0;

        virtual threads::thread_init_data&
        get_thread_init_data(continuation_type& cont,
            naming::address::address_type lva,
            threads::thread_init_data& data) = 0;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // Figure out what priority the action has to be be associated with
        // A dynamically specified default priority results in using the static
        // Priority.
        template <threads::thread_priority Priority>
        struct thread_priority
        {
            static threads::thread_priority
            call(threads::thread_priority priority)
            {
                if (priority == threads::thread_priority_default)
                    return Priority;
                return priority;
            }
        };

        // If the static Priority is default, a dynamically specified default
        // priority results in using the normal priority.
        template <>
        struct thread_priority<threads::thread_priority_default>
        {
            static threads::thread_priority
            call(threads::thread_priority priority)
            {
                if (priority == threads::thread_priority_default)
                    return threads::thread_priority_normal;
                return priority;
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct transfer_action : base_action
    {
        typedef typename Action::component_type component_type;
        typedef typename Action::derived_type derived_type;
        typedef typename Action::result_type result_type;
        typedef typename Action::arguments_type arguments_type;

        // This is the action code (id) of this action. It is exposed to allow
        // generic handling of actions.
        enum { value = Action::value };

        // This is the priority value this action has been instantiated with
        // (statically). This value might be different from the priority member
        // holding the runtime value an action has been created with
        enum { priority_value = Action::priority_value };

        typedef typename Action::direct_execution direct_execution;

        // default constructor is needed for serialization
        transfer_action() {}

        // construct an action from its arguments
        explicit transfer_action(threads::thread_priority priority)
          : arguments_(),
            parent_locality_(transfer_action::get_locality_id()),
            parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
            parent_phase_(threads::get_parent_phase()),
            priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(Action::priority_value)
                >::call(priority))
        {}

        template <typename Arg0>
        explicit transfer_action(BOOST_FWD_REF(Arg0) arg0)
          : arguments_(boost::forward<Arg0>(arg0)),
            parent_locality_(transfer_action::get_locality_id()),
            parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
            parent_phase_(threads::get_parent_phase()),
            priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(Action::priority_value)
                >::call(threads::thread_priority_default))
        {}

        template <typename Arg0>
        transfer_action(threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
          : arguments_(boost::forward<Arg0>(arg0)),
            parent_locality_(transfer_action::get_locality_id()),
            parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
            parent_phase_(threads::get_parent_phase()),
            priority_(
                detail::thread_priority<
                    static_cast<threads::thread_priority>(Action::priority_value)
                >::call(priority))
        {}

        // bring in the rest of the constructors
        #include <hpx/runtime/actions/transfer_action_constructors.hpp>

        //
        ~transfer_action()
        {
            detail::guid_initialization<transfer_action>();
        }

    public:
        /// retrieve component type
        static int get_static_component_type()
        {
            return Action::get_component_type();
        }

    private:
        /// The function \a get_component_type returns the \a component_type
        /// of the component this action belongs to.
        int get_component_type() const
        {
            return Action::get_component_type();
        }

        /// The function \a get_action_code returns the code of the action
        /// instance it is called for.
        std::size_t get_action_code() const
        {
            return static_cast<std::size_t>(value);
        }

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* get_action_name() const
        {
            return detail::get_action_name<derived_type>();
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        action_type get_action_type() const
        {
            return Action::get_action_type();
        }

        /// The \a get_thread_function constructs a proper thread function for
        /// a \a thread, encapsulating the functionality and the arguments
        /// of the action it is called for.
        ///
        /// \param lva    [in] This is the local virtual address of the
        ///               component the action has to be invoked on.
        ///
        /// \returns      This function returns a proper thread function usable
        ///               for a \a thread.
        ///
        /// \note This \a get_thread_function will be invoked to retrieve the
        ///       thread function for an action which has to be invoked without
        ///       continuations.
        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(naming::address::address_type lva)
        {
            return boost::move(Action::construct_thread_function(
                lva, arguments_));
        }

        /// The \a get_thread_function constructs a proper thread function for
        /// a \a thread, encapsulating the functionality, the arguments, and
        /// the continuations of the action it is called for.
        ///
        /// \param cont   [in] This is the list of continuations to be
        ///               triggered after the execution of the action
        /// \param lva    [in] This is the local virtual address of the
        ///               component the action has to be invoked on.
        ///
        /// \returns      This function returns a proper thread function usable
        ///               for a \a thread.
        ///
        /// \note This \a get_thread_function will be invoked to retrieve the
        ///       thread function for an action which has to be invoked with
        ///       continuations.
        HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            naming::address::address_type lva)
        {
            return boost::move(Action::construct_thread_function(
                cont, lva, arguments_));
        }

        /// Return the locality of the parent thread
        boost::uint32_t get_parent_locality_id() const
        {
            return parent_locality_;
        }

        /// Return the thread id of the parent thread
        threads::thread_id_type get_parent_thread_id() const
        {
            return reinterpret_cast<threads::thread_id_type>(parent_id_);
        }

        /// Return the phase of the parent thread
        std::size_t get_parent_thread_phase() const
        {
            return parent_phase_;
        }

        /// Return the thread priority this action has to be executed with
        threads::thread_priority get_thread_priority() const
        {
            return priority_;
        }

        /// Return all data needed for thread initialization
        threads::thread_init_data&
        get_thread_init_data(naming::address::address_type lva,
            threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = boost::move(Action::construct_thread_function(lva, arguments_));
            data.description = detail::get_action_name<derived_type>();
            data.parent_id = reinterpret_cast<threads::thread_id_type>(parent_id_);
            data.parent_locality_id = parent_locality_;
            data.priority = priority_;
            return data;
        }

        threads::thread_init_data&
        get_thread_init_data(continuation_type& cont,
            naming::address::address_type lva, threads::thread_init_data& data)
        {
            data.lva = lva;
            data.func = boost::move(Action::construct_thread_function(cont, lva, arguments_));
            data.description = detail::get_action_name<derived_type>();
            data.parent_id = reinterpret_cast<threads::thread_id_type>(parent_id_);
            data.parent_locality_id = parent_locality_;
            data.priority = priority_;
            return data;
        }

    public:
        /// retrieve the N's argument
        template <int N>
        typename boost::fusion::result_of::at_c<arguments_type, N>::type
        get()
        {
            return boost::fusion::at_c<N>(arguments_);
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<transfer_action, base_action>();
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template <class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            util::serialize_sequence(ar, arguments_);

            ar & parent_locality_;
            ar & parent_id_;
            ar & parent_phase_;
            ar & priority_;
        }

    private:
        static boost::uint32_t get_locality_id()
        {
            error_code ec;      // ignore any errors
            return hpx::get_locality_id(ec);
        }

    protected:
        arguments_type arguments_;
        boost::uint32_t parent_locality_;
        std::size_t parent_id_;
        std::size_t parent_phase_;
        threads::thread_priority priority_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <int N, typename Action>
    inline typename boost::fusion::result_of::at_c<
        typename transfer_action<Action>::arguments_type, N
    >::type
    get(transfer_action<Action> & args)
    {
        return args.get<N>();
    }

    #include <hpx/runtime/actions/construct_continuation_function_objects.hpp>

    ///////////////////////////////////////////////////////////////////////////
    /// \tparam Component         component type
    /// \tparam Action            action code
    /// \tparam Result            return type
    /// \tparam Arguments         arguments (fusion vector)
    /// \tparam Derived           derived action class
    /// \tparam threads::thread_priority Priority default priority
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived,
        threads::thread_priority Priority>
    struct action
    {
        typedef Component component_type;
        typedef Derived derived_type;
        typedef Result result_type;
        typedef Arguments arguments_type;

        typedef void action_tag;

        // This is the action code (id) of this action. It is exposed to allow
        // generic handling of actions.
        enum { value = Action };

        // This is the priority value this action has been instantiated with
        // (statically). This value might be different from the priority member
        // holding the runtime value an action has been created with
        enum { priority_value = Priority };

        ///////////////////////////////////////////////////////////////////////
        template <typename Func, typename Arguments_>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_continuation_thread_function_void(
            continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments_) args)
        {
            typedef typename boost::remove_reference<Arguments_>::type arguments_type;
            return detail::construct_continuation_thread_function_voidN<
                    derived_type,
                    boost::fusion::result_of::size<arguments_type>::value>::call(
                cont, boost::forward<Func>(func), boost::forward<Arguments_>(args));
        }

        template <typename Func, typename Arguments_>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_continuation_thread_function(
            continuation_type cont, BOOST_FWD_REF(Func) func,
            BOOST_FWD_REF(Arguments_) args)
        {
            typedef typename boost::remove_reference<Arguments_>::type arguments_type;
            return detail::construct_continuation_thread_functionN<
                    derived_type,
                    boost::fusion::result_of::size<arguments_type>::value>::call(
                cont, boost::forward<Func>(func), boost::forward<Arguments_>(args));
        }

        // bring in all overloads for
        //    construct_continuation_thread_function_void()
        //    construct_continuation_thread_object_function_void()
        //    construct_continuation_thread_function()
        //    construct_continuation_thread_object_function()
        #include <hpx/runtime/actions/construct_continuation_functions.hpp>

        // bring in the definition for all overloads for operator()
        template <typename IdType>
        typename boost::enable_if<
            boost::mpl::and_<
                boost::mpl::bool_<
                    boost::fusion::result_of::size<arguments_type>::value == 0>,
                boost::is_same<IdType, naming::id_type> >,
            typename traits::promise_local_result<Result>::type
        >::type
        operator()(IdType const& id, error_code& ec = throws) const
        {
            return hpx::async(*this, id).get(ec);
        }

        #include <hpx/runtime/actions/define_function_operators.hpp>

        /// retrieve component type
        static int get_component_type()
        {
            return static_cast<int>(components::get_component_type<Component>());
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        static base_action::action_type get_action_type()
        {
            return base_action::plain_action;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // Base template allowing to generate a concrete action type from a function
    // pointer. It is instantiated only if the supplied pointer is not a
    // supported function pointer.
    template <typename F, F funcptr, typename Direct = boost::mpl::false_>
    struct make_action;

    template <typename F, F funcptr>
    struct make_direct_action
      : make_action<F, funcptr, boost::mpl::true_>
    {};

// older compilers require BOOST_TYPEOF, newer compilers have decltype()
#if defined(BOOST_NO_DECLTYPE)
#  define HPX_TYPEOF(x)       BOOST_TYPEOF(x)
#  define HPX_TYPEOF_TPL(x)   BOOST_TYPEOF_TPL(x)
#else
#  define HPX_TYPEOF(x)       decltype(x)
#  define HPX_TYPEOF_TPL(x)   decltype(x)
#endif

    // Macros usable to refer to an action given the function to expose
    #define HPX_MAKE_ACTION(f)                                                \
        hpx::actions::make_action<HPX_TYPEOF(&f), &f>        /**/             \
    /**/
    #define HPX_MAKE_DIRECT_ACTION(f)                                         \
        hpx::actions::make_direct_action<HPX_TYPEOF(&f), &f> /**/             \
    /**/

    #define HPX_MAKE_ACTION_TPL(f)                                            \
        hpx::actions::make_action<HPX_TYPEOF_TPL(&f), &f>        /**/         \
    /**/
    #define HPX_MAKE_DIRECT_ACTION_TPL(f)                                     \
        hpx::actions::make_direct_action<HPX_TYPEOF_TPL(&f), &f> /**/         \
    /**/

#if BOOST_WORKAROUND(BOOST_MSVC, == 1600)
    // workarounds for VC2010
    #define HPX_MAKE_COMPONENT_ACTION(component, f)                           \
        hpx::actions::make_action<                                            \
            HPX_TYPEOF(component::f) component::*, &component::f>  /**/       \
    /**/
    #define HPX_MAKE_DIRECT_COMPONENT_ACTION(component, f)                    \
        hpx::actions::make_direct_action<                                     \
            HPX_TYPEOF(component::f) component::*, &component::f>  /**/       \
    /**/

    #define HPX_MAKE_COMPONENT_ACTION_TPL(component, f)                       \
        hpx::actions::make_action<                                            \
            HPX_TYPEOF_TPL(component::f) component::*, &component::f>  /**/   \
    /**/
    #define HPX_MAKE_DIRECT_COMPONENT_ACTION_TPL(component, f)                \
        hpx::actions::make_direct_action<                                     \
            HPX_TYPEOF_TPL(component::f) component::*, &component::f>  /**/   \
    /**/

    namespace detail
    {
        template <typename Obj, typename F>
        struct synthesize_const_mf;

        template <typename F> F replicate_type(F);
    }

    #define HPX_MAKE_CONST_COMPONENT_ACTION(component, f)                     \
        hpx::actions::make_action<                                            \
            hpx::actions::detail::synthesize_const_mf<                        \
                component, HPX_TYPEOF(                                        \
                    hpx::actions::detail::replicate_type(&component::f)       \
                )                                                             \
            >::type, &component::f>  /**/                                     \
    /**/
    #define HPX_MAKE_CONST_DIRECT_COMPONENT_ACTION(component, f)              \
        hpx::actions::make_direct_action<                                     \
            hpx::actions::detail::synthesize_const_mf<                        \
                component, HPX_TYPEOF(                                        \
                    hpx::actions::detail::replicate_type(&component::f)       \
                )                                                             \
            >::type, &component::f>  /**/                                     \
    /**/

    #define HPX_MAKE_CONST_COMPONENT_ACTION_TPL(component, f)                 \
        hpx::actions::make_action<                                            \
            typename hpx::actions::detail::synthesize_const_mf<               \
                component, HPX_TYPEOF_TPL(                                    \
                    hpx::actions::detail::replicate_type(&component::f)       \
                )                                                             \
            >::type, &component::f>  /**/                                     \
    /**/
    #define HPX_MAKE_CONST_DIRECT_COMPONENT_ACTION_TPL(component, f)          \
        hpx::actions::make_direct_action<                                     \
            typename hpx::actions::detail::synthesize_const_mf<               \
                component, HPX_TYPEOF_TPL(                                    \
                    hpx::actions::detail::replicate_type(&component::f)       \
                )                                                             \
            >::type, &component::f>  /**/                                     \
    /**/
#else
    // the implementation on conforming compilers is almost trivial
    #define HPX_MAKE_COMPONENT_ACTION(component, f)                           \
        HPX_MAKE_ACTION(component::f)::type                                   \
    /**/
    #define HPX_MAKE_DIRECT_COMPONENT_ACTION(component, f)                    \
        HPX_MAKE_DIRECT_ACTION(component::f)::type                            \
    /**/
    #define HPX_MAKE_CONST_COMPONENT_ACTION(component, f)                     \
        HPX_MAKE_ACTION(component::f)::type                                   \
    /**/
    #define HPX_MAKE_CONST_DIRECT_COMPONENT_ACTION(component, f)              \
        HPX_MAKE_DIRECT_ACTION(component::f)::type                            \
    /**/

    #define HPX_MAKE_COMPONENT_ACTION_TPL(component, f)                       \
        typename HPX_MAKE_ACTION_TPL(component::f)::type                      \
    /**/
    #define HPX_MAKE_DIRECT_COMPONENT_ACTION_TPL(component, f)                \
        typename HPX_MAKE_DIRECT_ACTION_TPL(component::f)::type               \
    /**/
    #define HPX_MAKE_CONST_COMPONENT_ACTION_TPL(component, f)                 \
        typename HPX_MAKE_ACTION_TPL(component::f)::type                      \
    /**/
    #define HPX_MAKE_CONST_DIRECT_COMPONENT_ACTION_TPL(component, f)          \
        typename HPX_MAKE_DIRECT_ACTION_TPL(component::f)::type               \
    /**/
#endif

    ///////////////////////////////////////////////////////////////////////////
    // This template meta function can be used to extract the action type, no
    // matter whether it got specified directly or by passing the
    // corresponding make_action<> specialization.
    template <typename Action, typename Enable = void>
    struct extract_action
    {
        typedef typename Action::derived_type type;
        typedef typename type::result_type result_type;
    };

    template <typename Action>
    struct extract_action<Action, typename Action::type>
    {
        typedef typename Action::type type;
        typedef typename type::result_type result_type;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // simple type allowing to distinguish whether an action is the most
        // derived one
        struct this_type {};

        template <typename Action, typename Derived>
        struct action_type
          : boost::mpl::if_<boost::is_same<Derived, this_type>, Action, Derived>
        {};
    }

    /// \endcond
}}

/// \cond NOINTERNAL

#include <hpx/config/warnings_suffix.hpp>

///////////////////////////////////////////////////////////////////////////////
// Helper macro for action serialization, each of the defined actions needs to
// be registered with the serialization library
#define HPX_DEFINE_GET_ACTION_NAME(action)                                    \
    HPX_DEFINE_GET_ACTION_NAME_EX(action, action)                             \
/**/

#define HPX_DEFINE_GET_ACTION_NAME_EX(action, actionname)                     \
    namespace hpx { namespace actions { namespace detail {                    \
        template<> HPX_ALWAYS_EXPORT                                          \
        char const* get_action_name<action>()                                 \
        {                                                                     \
            return BOOST_PP_STRINGIZE(actionname);                            \
        }                                                                     \
    }}}                                                                       \
/**/

#define HPX_REGISTER_ACTION_EX(action, actionname)                            \
    BOOST_CLASS_EXPORT_IMPLEMENT(hpx::actions::transfer_action<action>)       \
    HPX_REGISTER_BASE_HELPER(hpx::actions::transfer_action<action>, actionname) \
    HPX_DEFINE_GET_ACTION_NAME_EX(action, actionname)                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BASE_HELPER(action, actionname)                          \
        hpx::actions::detail::register_base_helper<action>                    \
            BOOST_PP_CAT(                                                     \
                BOOST_PP_CAT(__hpx_action_register_base_helper_, __LINE__),   \
                _##actionname);                                               \
    /**/

#define HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID1(action)              \
    namespace hpx { namespace actions { namespace detail {                    \
        template <> HPX_ALWAYS_EXPORT                                         \
        char const* get_action_name<action>();                                \
    }}}                                                                       \
/**/
#define HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID2(action)              \
    namespace hpx { namespace traits {                                        \
        template <>                                                           \
        struct needs_guid_initialization<action>                              \
          : boost::mpl::false_                                                \
        {};                                                                   \
    }}                                                                        \
/**/
#define HPX_REGISTER_ACTION_DECLARATION_GUID(action)                          \
    namespace boost { namespace archive { namespace detail {                  \
        namespace extra_detail {                                              \
            template <>                                                       \
            struct init_guid<action>;                                         \
        }                                                                     \
    }}}                                                                       \
/**/
#define HPX_REGISTER_ACTION_DECLARATION_EX(action, actionname)                \
    HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID1(action)                  \
    HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID2(                         \
        hpx::actions::transfer_action<action>)                                \
    BOOST_CLASS_EXPORT_KEY2(hpx::actions::transfer_action<action>,            \
        BOOST_PP_STRINGIZE(actionname))                                       \
    HPX_REGISTER_ACTION_DECLARATION_GUID(hpx::actions::transfer_action<action>) \
/**/

/// \endcond

/// \def HPX_REGISTER_ACTION_DECLARATION(action)
///
/// \brief Declare the necessary component action boilerplate code.
///
/// The macro \a HPX_REGISTER_ACTION_DECLARATION can be used to declare all the
/// boilerplate code which is required for proper functioning of component
/// actions in the context of HPX.
///
/// The parameter \a action is the type of the action to declare the
/// boilerplate for.
///
/// \par Example:
///
/// \code
///      namespace app
///      {
///          // Define a simple component exposing one action 'print_greating'
///          class HPX_COMPONENT_EXPORT server
///            : public hpx::components::simple_component_base<server>
///          {
///              void print_greating ()
///              {
///                  hpx::cout << "Hey, how are you?\n" << hpx::flush;
///              }
///
///              // Component actions need to be declared, this also defines the
///              // type 'print_greating_action' representing the action.
///              HPX_DEFINE_COMPONENT_ACTION(server, print_greating, print_greating_action);
///          };
///      }
///
///      // Declare boilerplate code required for each of the component actions.
///      HPX_REGISTER_ACTION_DECLARATION(app::server::print_greating_action);
/// \endcode
///
/// \note This macro has to be used once for each of the component actions
/// defined using one of the \a HPX_DEFINE_COMPONENT_ACTION macros. It has to
/// be visible in all translation units using the action, thus it is
/// recommended to place it into the header file defining the component.
#define HPX_REGISTER_ACTION_DECLARATION(action)                               \
    HPX_REGISTER_ACTION_DECLARATION_EX(action, action)                        \
/**/

/// \def HPX_REGISTER_ACTION(action)
///
/// \brief Define the necessary component action boilerplate code.
///
/// The macro \a HPX_REGISTER_ACTION can be used to define all the
/// boilerplate code which is required for proper functioning of component
/// actions in the context of HPX.
///
/// The parameter \a action is the type of the action to define the
/// boilerplate for.
///
/// \note This macro has to be used once for each of the component actions
/// defined using one of the \a HPX_DEFINE_COMPONENT_ACTION macros. It has to
/// occur exactly once for each of the actions, thus it is recommended to
/// place it into the source file defining the component.
#define HPX_REGISTER_ACTION(action)                                           \
    HPX_REGISTER_ACTION_EX(action, action)                                    \
/**/

#endif

