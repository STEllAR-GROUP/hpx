//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_ACTION_SUPPORT_NOV_14_2008_0711PM)
#define HPX_RUNTIME_ACTIONS_ACTION_SUPPORT_NOV_14_2008_0711PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>

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
#include <boost/move/move.hpp>
#include <boost/typeof/typeof.hpp>

#include <hpx/runtime/get_lva.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/serialize_sequence.hpp>
#include <hpx/util/serialize_exception.hpp>
#include <hpx/util/demangle_helper.hpp>
#include <hpx/util/base_object.hpp>
#include <hpx/util/void_cast.hpp>

#include <hpx/config/bind.hpp>
#include <hpx/config/tuple.hpp>
#include <hpx/config/function.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
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

        template <typename Action, typename Enable = void>
        struct needs_guid_initialization
            : boost::mpl::true_
        {};

        template <typename Action>
        void guid_initialization(boost::mpl::false_) {}

        template <typename Action>
        void guid_initialization(boost::mpl::true_)
        {
            // force serialization self registration to happen
            using namespace boost::archive::detail::extra_detail;
            init_guid<Action>::g.initialize();
        }

        template <typename Action>
        void guid_initialization()
        {
            guid_initialization<Action>(
                typename needs_guid_initialization<Action>::type()
            );
        }

        template <typename Action>
        char const* get_action_name()
        {
            /// If you encounter this assert while compiling code, that means that
            /// you have a HPX_REGISTER_ACTION macro somewhere in a source file,
            /// but the header in which the action is defined misses a
            /// HPX_REGISTER_ACTION_DECLARATION
            BOOST_MPL_ASSERT_MSG(
                needs_guid_initialization<Action>::value
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
    /// \tparam Component         component type
    /// \tparam Action            action code
    /// \tparam Result            return type
    /// \tparam Arguments         arguments (fusion vector)
    /// \tparam Derived           derived action class
    /// \tparam threads::thread_priority Priority default priority
    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived,
        threads::thread_priority Priority>
    class action : public base_action
    {
    public:
        typedef Component component_type;
        typedef Derived derived_type;
        typedef Result result_type;
        typedef Arguments arguments_type;

        // This is the action code (id) of this action. It is exposed to allow
        // generic handling of actions.
        enum { value = Action };

        // This is the priority value this action has been instantiated with
        // (statically). This value might be different from the priority member
        // holding the runtime value an action has been created with
        enum { priority_value = Priority };

        // construct an action from its arguments
        explicit action(threads::thread_priority priority)
          : arguments_(), parent_locality_(0), parent_id_(0), parent_phase_(0),
            priority_(detail::thread_priority<Priority>::call(priority))
        {}

        template <typename Arg0>
        action(BOOST_FWD_REF(Arg0) arg0)
          : arguments_(boost::forward<Arg0>(arg0)),
            parent_locality_(action::get_locality_id()),
            parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
            parent_phase_(threads::get_parent_phase()),
            priority_(detail::thread_priority<Priority>::call(Priority))
        {}

        template <typename Arg0>
        action(threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
          : arguments_(boost::forward<Arg0>(arg0)),
            parent_locality_(action::get_locality_id()),
            parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
            parent_phase_(threads::get_parent_phase()),
            priority_(detail::thread_priority<Priority>::call(priority))
        {}

        // bring in the rest of the constructors
        #include <hpx/runtime/actions/action_constructors.hpp>

        /// destructor
        ~action()
        {
            detail::guid_initialization<derived_type>();
        }

    public:
        /// retrieve the N's argument
        template <int N>
        typename boost::fusion::result_of::at_c<arguments_type, N>::type
        get()
        {
            return boost::fusion::at_c<N>(arguments_);
        }

    protected:
        // bring in all overloads for
        //    construct_continuation_thread_function_void()
        //    construct_continuation_thread_object_function_void()
        //    construct_continuation_thread_function()
        //    construct_continuation_thread_object_function()
        #include <hpx/runtime/actions/construct_continuation_functions.hpp>

    public:
        /// retrieve component type
        static int get_static_component_type()
        {
            return static_cast<int>(components::get_component_type<Component>());
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<action, base_action>();
        }

    private:
        /// retrieve action code
        std::size_t get_action_code() const
        {
            return static_cast<std::size_t>(value);
        }

        /// retrieve component type
        int get_component_type() const
        {
            return get_static_component_type();
        }

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* get_action_name() const
        {
            return detail::get_action_name<Derived>();
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        base_action::action_type get_action_type() const
        {
            return base_action::plain_action;
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

        static boost::uint32_t get_locality_id()
        {
            error_code ec;      // ignore any errors
            return hpx::get_locality_id(ec);
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template <class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            util::serialize_sequence(ar, arguments_);
            //ar & arguments_;
            ar & parent_locality_;
            ar & parent_id_;
            ar & parent_phase_;
            ar & priority_;
        }

    protected:
        arguments_type arguments_;
        boost::uint32_t parent_locality_;
        std::size_t parent_id_;
        std::size_t parent_phase_;
        threads::thread_priority priority_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <int N, typename Component, int Action, typename Result,
      typename Arguments, typename Derived, threads::thread_priority Priority>
    inline typename boost::fusion::result_of::at_c<
        typename action<Component, Action, Result, Arguments, Derived,
            Priority>::arguments_type, N
    >::type
    get(action<Component, Action, Result, Arguments, Derived, Priority> & args)
    {
        return args.get<N>();
    }

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

    // Macros usable to refer to an action given the function to expose
    #define HPX_MAKE_ACTION(f)                                                \
        hpx::actions::make_action<BOOST_TYPEOF(&f), &f>        /**/           \
    /**/
    #define HPX_MAKE_DIRECT_ACTION(f)                                         \
        hpx::actions::make_direct_action<BOOST_TYPEOF(&f), &f> /**/           \
    /**/

#if BOOST_WORKAROUND(BOOST_MSVC, == 1600)
    // workarounds for VC2010
    #define HPX_MAKE_COMPONENT_ACTION(component, f)                           \
        hpx::actions::make_action<                                            \
            BOOST_TYPEOF(component::f) component::*, &component::f>  /**/     \
    /**/
    namespace detail
    {
        template <typename Obj, typename F>
        struct synthesize_const_mf;

        template <typename F> F replicate_type(F);
    }
    #define HPX_MAKE_CONST_COMPONENT_ACTION(component, f)                     \
        hpx::actions::make_action<hpx::actions::detail::synthesize_const_mf<  \
            component, BOOST_TYPEOF(                                          \
                hpx::actions::detail::replicate_type(&component::f)           \
            )                                                                 \
        >::type, &component::f>  /**/                                         \
    /**/
#else
    // the implementation on conforming compilers is almost trivial
    #define HPX_MAKE_COMPONENT_ACTION(component, f)                           \
        HPX_MAKE_ACTION(component::f)::type                                   \
    /**/
    #define HPX_MAKE_CONST_COMPONENT_ACTION(component, f)                     \
        HPX_MAKE_ACTION(component::f)::type                                   \
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
    // Helper to invoke the registration code for serialization at startup
    template <typename Action>
    struct register_base_helper
    {
        register_base_helper()
        {
            Action::register_base();
        }
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
}}

#include <hpx/config/warnings_suffix.hpp>

///////////////////////////////////////////////////////////////////////////////
// Helper macro for action serialization, each of the defined actions needs to
// be registered with the serialization library
#define HPX_DEFINE_GET_ACTION_NAME(action)                                    \
    HPX_DEFINE_GET_ACTION_NAME(action, action)                                \
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
    BOOST_CLASS_EXPORT_IMPLEMENT(action)                                      \
    HPX_REGISTER_BASE_HELPER(action, actionname)                              \
    HPX_DEFINE_GET_ACTION_NAME_EX(action, actionname)                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BASE_HELPER(action, actionname)                          \
        hpx::actions::register_base_helper<action>                            \
            BOOST_PP_CAT(                                                     \
                BOOST_PP_CAT(__hpx_action_register_base_helper_, __LINE__),   \
                _##actionname);                                               \
    /**/

#define HPX_REGISTER_ACTION(action) HPX_REGISTER_ACTION_EX(action, action)

#define HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID(action)               \
    namespace hpx { namespace actions { namespace detail {                    \
        template <> HPX_ALWAYS_EXPORT                                         \
        char const* get_action_name<action>();                                \
        template <typename Enable>                                            \
        struct needs_guid_initialization<action, Enable>                      \
            : boost::mpl::false_                                              \
        {};                                                                   \
    }}}                                                                       \
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
    HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID(action)                   \
    BOOST_CLASS_EXPORT_KEY2(action, BOOST_PP_STRINGIZE(actionname))           \
    HPX_REGISTER_ACTION_DECLARATION_GUID(action)                              \
/**/
#define HPX_REGISTER_ACTION_DECLARATION(action)                               \
    HPX_REGISTER_ACTION_DECLARATION_EX(action, action)                        \
/**/

#endif

