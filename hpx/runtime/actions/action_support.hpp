//  Copyright (c) 2007-2009 Hartmut Kaiser
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
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/serialization/void_cast.hpp>

#include <hpx/runtime/get_lva.hpp>
#include <hpx/util/serialize_sequence.hpp>

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

        template <typename Action>
        HPX_ALWAYS_EXPORT char const* const get_action_name();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The \a base_action class is a abstract class used as the base class for 
    /// all action types. It's main purpose is to allow polymorphic 
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

        /// The function \a get_action_code returns the code of the action 
        /// instance it is called for.
        virtual int get_action_code() const = 0;

        /// The function \a get_component_type returns the \a component_type 
        /// of the component this action belongs to.
        virtual int get_component_type() const = 0;

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        virtual char const* const get_action_name() const = 0;

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
        virtual boost::function<threads::thread_function_type> 
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
        virtual boost::function<threads::thread_function_type>
            get_thread_function(continuation_type& cont,
                naming::address::address_type lva) const = 0;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, typename Arguments>
    class action : public base_action
    {
    public:
        typedef Component component_type;
        typedef Arguments arguments_type;
        typedef void result_type;

        // This is the action code (id) of this action. It is exposed to allow 
        // generic handling of actions.
        enum { value = Action };

        // construct an action from its arguments
        action() 
          : arguments_() 
        {}

        template <typename Arg0>
        action(Arg0 const& arg0) 
          : arguments_(arg0) 
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
        template <typename Func>
        static threads::thread_state 
        continuation_thread_function_void(continuation_type cont, 
            boost::tuple<Func> func)
        {
            try {
                boost::get<0>(func)();
                cont->trigger_all();
            }
            catch (hpx::exception const& e) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(e);
            }
            return threads::terminated;
        }

        /// The \a construct_continuation_thread_function is a helper function
        /// for constructing the wrapped thread function needed for 
        /// continuation support
        template <typename Func>
        static boost::function<threads::thread_function_type>
        construct_continuation_thread_function_void(Func func, continuation_type cont) 
        {
            // we need to assign the address of the thread function to a 
            // variable to  help the compiler to deduce the function type
            threads::thread_state (*f)(continuation_type, boost::tuple<Func>) =
                &action::continuation_thread_function_void;

            // The following bind constructs the wrapped thread function
            //    f:  is the wrapping thread function
            // cont: continuation 
            // func: wrapped function object 
            //       (this is embedded into a tuple because boost::bind can't
            //       pre-bind another bound function as an argument)
            return boost::bind(f, cont, boost::make_tuple(func));
        }

        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the 
        /// original function (given by \a func), and afterwards triggers all
        /// continuations using the result value obtained from the execution
        /// of the original thread function.
        template <typename Func>
        static threads::thread_state 
        continuation_thread_function(continuation_type cont, 
            boost::tuple<Func> func)
        {
            try {
                cont->trigger_all(boost::get<0>(func)());
            }
            catch (hpx::exception const& e) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(e);
            }
            return threads::terminated;
        }

        /// The \a construct_continuation_thread_function is a helper function
        /// for constructing the wrapped thread function needed for 
        /// continuation support
        template <typename Func>
        static boost::function<threads::thread_function_type>
        construct_continuation_thread_function(Func func, continuation_type cont) 
        {
            // we need to assign the address of the thread function to a 
            // variable to  help the compiler to deduce the function type
            threads::thread_state (*f)(continuation_type, boost::tuple<Func>) =
                &action::continuation_thread_function;

            // The following bind constructs the wrapped thread function
            //    f:  is the wrapping thread function
            // cont: continuation 
            // func: wrapped function object 
            //       (this is embedded into a tuple because boost::bind can't
            //       pre-bind another bound function as an argument)
            return boost::bind(f, cont, boost::make_tuple(func));
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
            void_cast_register<action, base_action>();
        }

    private:
        /// retrieve action code
        int get_action_code() const 
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
        char const* const get_action_name() const
        {
            return "<Unknown action type>";
        }

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        base_action::action_type get_action_type() const 
        {
            return base_action::plain_action;
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            util::serialize_sequence(ar, arguments_);
        }

    private:
        arguments_type arguments_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <int N, typename Component, int Action, typename Arguments>
    inline typename boost::fusion::result_of::at_c<
        typename action<Component, Action, Arguments>::arguments_type const, N
    >::type 
    get(action<Component, Action, Arguments> const& args) 
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
}}

#include <hpx/config/warnings_suffix.hpp>

///////////////////////////////////////////////////////////////////////////////
// Helper macro for action serialization, each of the defined actions needs to 
// be registered with the serialization library
#define HPX_DEFINE_GET_ACTION_NAME(action)                                    \
        namespace hpx { namespace actions { namespace detail {                \
            template<> HPX_ALWAYS_EXPORT                                      \
            char const* const get_action_name<action>()                       \
            { return BOOST_PP_STRINGIZE(action); }                            \
        }}}                                                                   \
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
