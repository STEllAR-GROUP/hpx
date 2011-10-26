//  Copyright (c) 2007-2011 Hartmut Kaiser
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
#include <boost/fusion/include/any.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/ref.hpp>
#include <boost/foreach.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/if.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/void_cast.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/export.hpp>
#include <boost/move/move.hpp>

#include <hpx/runtime/get_lva.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/util/serialize_sequence.hpp>

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
            typedef typename boost::remove_reference<T>::type no_ref_type;
            typedef typename boost::remove_const<no_ref_type>::type type;
        };

        namespace ext
        {
            template <typename Action>
            struct HPX_ALWAYS_EXPORT get_action_name_impl
            {
                static char const * call();
            };
        }

        template <typename Action>
        char const* get_action_name()
        {
            return ext::get_action_name_impl<Action>::call();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace traits
    {
        // The customization point handle_gid is used to handle reference
        // counting of GIDs while they are transferred to a different locality.
        // It has to be specialized for arbitrary types, which may hold GIDs.
        //
        // It is important to make sure that all GID instances which are
        // contained in any transferred data structure are handled during
        // serialization. For this reason any user defined data type, which
        // is passed as an parameter to a action or which is returned from
        // a result_action needs to provide a corresponding specialization.
        //
        // The purpose of this customization point is to call the provided
        // function for all GIDs held in the data type.
        template <typename T, typename F, typename Enable = void>
        struct handle_gid
        {
            static bool call(T const&, F)
            {
                return true;    // do nothing for arbitrary types
            }
        };

        template <typename F>
        struct handle_gid<naming::id_type, F>
        {
            static bool call(naming::id_type const &id, F const& f)
            {
                f(id);
                return true;
            }
        };

        template <typename F>
        struct handle_gid<std::vector<naming::id_type>, F>
        {
            static bool call(std::vector<naming::id_type> const& ids, F const& f)
            {
                BOOST_FOREACH(naming::id_type const& id, ids)
                    f(boost::ref(id));
                return true;
            }
        };
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
            get_thread_function(naming::address::address_type lva) const = 0;

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
                naming::address::address_type lva) const = 0;

        /// return the id of the locality of the parent thread
        virtual boost::uint32_t get_parent_locality_prefix() const = 0;

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

        /// Enumerate all GIDs which stored as arguments
        typedef HPX_STD_FUNCTION<void(naming::id_type const&)> enum_gid_handler_type;
        virtual void enumerate_argument_gids(enum_gid_handler_type) = 0;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct enum_gid_handler
    {
        /// Enumerate all GIDs which stored as arguments
        typedef base_action::enum_gid_handler_type enum_gid_handler_type;

        enum_gid_handler(enum_gid_handler_type f)
          : f_(f)
        {}

        template <typename T>
        bool operator()(T const& t) const
        {
            return traits::handle_gid<T, enum_gid_handler_type>::call(t, f_);
        }

        enum_gid_handler_type f_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename Arguments>
    struct signature;

    template <typename Result>
    struct signature<Result, boost::fusion::vector<> > : base_action
    {
        typedef boost::fusion::vector<> arguments_type;
        typedef Result result_type;

        virtual result_type execute_function(
            naming::address::address_type lva) const = 0;

        virtual HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(naming::address::address_type lva,
            arguments_type const& args) const = 0;

        virtual HPX_STD_FUNCTION<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            naming::address::address_type lva,
            arguments_type const& args) const = 0;

        virtual threads::thread_init_data&
        get_thread_init_data(naming::address::address_type lva,
            threads::thread_init_data& data,
            arguments_type const& args) = 0;

        virtual threads::thread_init_data&
        get_thread_init_data(continuation_type& cont,
            naming::address::address_type lva,
            threads::thread_init_data& data,
            arguments_type const& args) = 0;

    public:
        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<signature, base_action>();
        }
    };

    #include <hpx/runtime/actions/signature_implementations.hpp>

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

    template <typename Component                // Component type
            , int Action                        // Action code
            , typename Result                   // Return type
            , typename Arguments                // Arguments (fusion vector)
            , typename Derived                  // Derived action class
            , threads::thread_priority Priority /* Default priority */>
    class action : public signature<Result, Arguments>
    {
    public:
        typedef Component component_type;
        typedef Derived derived_type;
        typedef Result result_type;
        typedef Arguments arguments_type;

        /// Enumerate all GIDs which stored as arguments
        typedef HPX_STD_FUNCTION<void(naming::id_type const&)>
            enum_gid_handler_type;

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
        action(Arg0 const& arg0)
          : arguments_(arg0),
            parent_locality_(applier::get_prefix_id()),
            parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
            parent_phase_(threads::get_parent_phase()),
            priority_(detail::thread_priority<Priority>::call(Priority))
        {}

        template <typename Arg0>
        action(threads::thread_priority priority, Arg0 const& arg0)
          : arguments_(arg0),
            parent_locality_(applier::get_prefix_id()),
            parent_id_(reinterpret_cast<std::size_t>(threads::get_parent_id())),
            parent_phase_(threads::get_parent_phase()),
            priority_(detail::thread_priority<Priority>::call(priority))
        {}

        // bring in the rest of the constructors
        #include <hpx/runtime/actions/action_constructors.hpp>

        /// destructor
        ~action()
        {}

    public:
        /// retrieve the N's argument
        template <int N>
        typename boost::fusion::result_of::at_c<arguments_type, N>::type
        get()
        {
            return boost::fusion::at_c<N>(arguments_);
        }
        template <int N>
        typename boost::fusion::result_of::at_c<arguments_type const, N>::type
        get() const
        {
            return boost::fusion::at_c<N>(arguments_);
        }

    protected:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func), and afterwards triggers all
        /// continuations using the result value obtained from the execution
        /// of the original thread function.
        struct continuation_thread_function_void
        {
            typedef threads::thread_state_enum result_type;

            template <typename Func>
            result_type operator()(continuation_type cont,
                Func const & func) const
            {
                try {
                    LTM_(debug) << "Executing action("
                                << detail::get_action_name<derived_type>()
                                << ") with continuation("
                                << cont->get_raw_gid()
                                << ")";
                    func();
                    cont->trigger();
                }
                catch (hpx::exception const&) {
                    // make sure hpx::exceptions are propagated back to the client
                    cont->trigger_error(boost::current_exception());
                }
                return threads::terminated;
             }
        };

        /// The \a construct_continuation_thread_function is a helper function
        /// for constructing the wrapped thread function needed for
        /// continuation support
        template <typename Func>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_continuation_thread_function_void(BOOST_FWD_REF(Func) func,
            continuation_type cont)
        {
            // The following bind constructs the wrapped thread function
            //    f:  is the wrapping thread function
            // cont: continuation
            // func: wrapped function object
            return HPX_STD_BIND(continuation_thread_function_void(), cont,
                HPX_STD_PROTECT(boost::forward<Func>(func)));
        }

        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the
        /// original function (given by \a func), and afterwards triggers all
        /// continuations using the result value obtained from the execution
        /// of the original thread function.
        struct continuation_thread_function
        {
            typedef threads::thread_state_enum result_type;

            template <typename Func>
            result_type operator()(continuation_type cont,
                Func const & func) const
            {
                try {
                    LTM_(debug) << "Executing action("
                                << detail::get_action_name<derived_type>()
                                << ") with continuation("
                                << cont->get_raw_gid()
                                << ")";
                    cont->trigger(func());
                }
                catch (hpx::exception const&) {
                    // make sure hpx::exceptions are propagated back to the client
                    cont->trigger_error(boost::current_exception());
                }
                return threads::terminated;
            }
        };

        /// The \a construct_continuation_thread_function is a helper function
        /// for constructing the wrapped thread function needed for
        /// continuation support
        template <typename Func>
        static HPX_STD_FUNCTION<threads::thread_function_type>
        construct_continuation_thread_function(BOOST_FWD_REF(Func) func,
            continuation_type cont)
        {
            // The following bind constructs the wrapped thread function
            //    f:  is the wrapping thread function
            // cont: continuation
            // func: wrapped function object
            return HPX_STD_BIND(continuation_thread_function(), cont,
                HPX_STD_PROTECT(boost::forward<Func>(func)));
        }

    public:
        /// retrieve component type
        static int get_static_component_type()
        {
            return static_cast<int>(components::get_component_type<Component>());
        }

        /// serialization support
        static void register_base()
        {
            using namespace boost::serialization;
            void_cast_register<action, signature<Result, Arguments> >();
            signature<Result, Arguments>::register_base();
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
        boost::uint32_t get_parent_locality_prefix() const
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

        void enumerate_argument_gids(enum_gid_handler_type f)
        {
            boost::fusion::any(arguments_, enum_gid_handler(f));
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
        typename action<Component, Action, Result, Arguments, Derived, Priority>::arguments_type const, N
    >::type
    get(action<Component, Action, Result, Arguments, Derived, Priority> const& args)
    {
        return args.get<N>();
    }

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
        namespace hpx { namespace actions { namespace detail { namespace ext {\
            template<> HPX_ALWAYS_EXPORT struct get_action_name_impl<action>  \
            {                                                                 \
                static char const* call()                                     \
                { return BOOST_PP_STRINGIZE(action); }                        \
            };                                                                \
        }}}}                                                                  \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_BASE_HELPER(action, actionname)                          \
        hpx::actions::register_base_helper<action>                            \
            BOOST_PP_CAT(                                                     \
                BOOST_PP_CAT(__hpx_action_register_base_helper_, __LINE__),   \
                _##actionname);                                               \
    /**/

#define HPX_REGISTER_ACTION_EX(action, actionname)                            \
        BOOST_CLASS_EXPORT(action)                                            \
        HPX_REGISTER_BASE_HELPER(action, actionname)                          \
        HPX_DEFINE_GET_ACTION_NAME(action)                                    \
    /**/

#define HPX_REGISTER_ACTION(action) HPX_REGISTER_ACTION_EX(action, action)

#endif

