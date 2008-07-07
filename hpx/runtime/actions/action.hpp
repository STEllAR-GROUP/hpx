//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_ACTION_MAR_26_2008_1054AM)
#define HPX_RUNTIME_ACTIONS_ACTION_MAR_26_2008_1054AM

#include <cstdlib>
#include <stdexcept>

#include <boost/version.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/get_lva.hpp>
#include <hpx/util/serialize_sequence.hpp>

///////////////////////////////////////////////////////////////////////////////
// Helper macro for action serialization, each of the defined actions needs to 
// be registered with the serialization library
#define HPX_SERIALIZE_ACTION(action)                                          \
        BOOST_CLASS_EXPORT(action)                                            \
    /**/

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
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The \a action_base class is a abstract class used as the base class for 
    /// all action types. It's main purpose is to allow polymorphic 
    /// serialization of action instances through a shared_ptr.
    struct action_base
    {
        virtual ~action_base() {}
        
        /// The function \a get_action_code returns the code of the action 
        /// instance it is called for.
        virtual std::size_t get_action_code() const = 0;

        /// The function \a get_component_type returns the \a component_type 
        /// of the component this action belongs to.
        virtual int get_component_type() const = 0;

        /// The \a get_thread_function constructs a proper thread function for 
        /// a \a thread, encapsulating the functionality and the arguments 
        /// of the action it is called for.
        /// 
        /// \param appl   [in] This is a reference to the \a applier instance 
        ///               to be passed as the second parameter to the action 
        ///               function
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
            get_thread_function(applier::applier& appl, 
                naming::address::address_type lva) const = 0;

        /// The \a get_thread_function constructs a proper thread function for 
        /// a \a thread, encapsulating the functionality, the arguments, and 
        /// the continuations of the action it is called for.
        /// 
        /// \param cont   [in] This is the list of continuations to be 
        ///               triggered after the execution of the action
        /// \param appl   [in] This is a reference to the \a applier instance 
        ///               to be passed as the second parameter to the action 
        ///               function
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
                applier::applier& appl, naming::address::address_type lva) const = 0;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, int Action, typename Arguments>
    class action : public action_base
    {
    public:
        typedef Arguments arguments_type;

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
        /// continuations without any additional argument
        template <typename Func>
        static threads::thread_state 
        continuation_thread_function(
            threads::thread_self& self, applier::applier& app, 
            continuation_type cont, boost::tuple<Func> func)
        {
            threads::thread_state newstate = boost::get<0>(func)(self);
            cont->trigger_all(app);
            return newstate;
        }

        /// The \a construct_continuation_thread_function is a helper function
        /// for constructing the wrapped thread function needed for 
        /// continuation support
        template <typename Func>
        static boost::function<threads::thread_function_type>
        construct_continuation_thread_function(Func func, 
            applier::applier& appl, continuation_type cont) 
        {
            // we need to assign the address of the thread function to a 
            // variable to  help the compiler to deduce the function type
            threads::thread_state (*f)(threads::thread_self&, 
                    applier::applier&, continuation_type, 
                    boost::tuple<Func>) =
                &action::continuation_thread_function;

            // The following bind constructs the wrapped thread function
            //   f:  is the wrapping thread function
            //  _1:  is a placeholder which will be replaced by the reference
            //       to thread_self
            //  app: reference to the applier (pre-bound second argument to f)
            // cont: continuation (pre-bound third argument to f)
            // func: wrapped function object (pre-bound forth argument to f)
            //       (this is embedded into a tuple because boost::bind can't
            //       pre-bind another bound function as an argument)
            return boost::bind(f, _1, boost::ref(appl), cont, 
                boost::make_tuple(func));
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
            return static_cast<int>(Component::value);
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            using namespace boost::serialization;
            void_cast_register<action, action_base>();
            
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
    //  Specialized generic action types allowing to hold a different number of
    //  arguments
    ///////////////////////////////////////////////////////////////////////////

    // zero argument version
    template <
        typename Component, typename Result, int Action, 
        threads::thread_state(Component::*F)(
            threads::thread_self&, applier::applier&, Result*)
    >
    class result_action0 
      : public action<Component, Action, boost::fusion::vector<> >
    {
        typedef action<Component, Action, boost::fusion::vector<> > base_type;

    public:
        result_action0()
        {}

    private:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the 
        /// original function (given by \a func), and afterwards triggers all
        /// continuations using the result value obtained from the execution
        /// of the original thread function.
        template <typename Func>
        static threads::thread_state 
        continuation_thread_function(
            threads::thread_self& self, applier::applier& app, 
            continuation_type cont, boost::tuple<Func> func)
        {
            Result result;
            threads::thread_state newstate = 
                boost::get<0>(func)(self, &result);
            cont->trigger_all(app, result);
            return newstate;
        }

        /// The \a construct_continuation_thread_function is a helper function
        /// for constructing the wrapped thread function needed for 
        /// continuation support
        template <typename Func>
        static boost::function<threads::thread_function_type>
        construct_continuation_thread_function(Func func, 
            applier::applier& appl, continuation_type cont) 
        {
            // we need to assign the address of the thread function to a 
            // variable to  help the compiler to deduce the function type
            threads::thread_state (*f)(threads::thread_self&, 
                    applier::applier&, continuation_type, 
                    boost::tuple<Func>) =
                &result_action0::continuation_thread_function;

            // The following bind constructs the wrapped thread function
            //   f:  is the wrapping thread function
            //  _1:  is a placeholder which will be replaced by the reference
            //       to thread_self
            //  app: reference to the applier (pre-bound second argument to f)
            // cont: continuation (pre-bound third argument to f)
            // func: wrapped function object (pre-bound forth argument to f)
            //       (this is embedded into a tuple because boost::bind can't
            //       pre-bind another bound function as an argument)
            return boost::bind(f, _1, boost::ref(appl), cont, 
                boost::make_tuple(func));
        }

    public:
        typedef boost::mpl::false_ direct_execution;
        typedef Result result_type;

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a result_action0 type. This is used by the \a 
        /// applier in case no continuation has been supplied.
        static boost::function<threads::thread_function_type> 
        construct_thread_function(applier::applier& appl, 
            naming::address::address_type lva)
        {
            return boost::bind(F, get_lva<Component>::call(lva), _1, 
                boost::ref(appl), reinterpret_cast<Result*>(NULL));
        }

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a result_action0 type. This is used by the \a 
        /// applier in case a continuation has been supplied
        static boost::function<threads::thread_function_type> 
        construct_thread_function(continuation_type& cont, 
            applier::applier& appl, naming::address::address_type lva)
        {
            return construct_continuation_thread_function(
                boost::bind(F, get_lva<Component>::call(lva), _1, 
                    boost::ref(appl), _2), 
                appl, cont);
        }

    private:
        /// This \a get_thread_function will be invoked to retrieve the thread 
        /// function for an action which has to be invoked without continuations.
        boost::function<threads::thread_function_type>
        get_thread_function(applier::applier& appl, 
            naming::address::address_type lva) const
        {
            return construct_thread_function(appl, lva);
        }

        /// This \a get_thread_function will be invoked to retrieve the thread 
        /// function for an action which has to be invoked with continuations.
        boost::function<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            applier::applier& appl, naming::address::address_type lva) const
        {
            return construct_thread_function(cont, appl, lva);
        }

    private:
        // serialization support    
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result, int Action, 
        threads::thread_state(Component::*F)(
            threads::thread_self&, applier::applier&, Result*),
        Result (Component::*DirectF)(applier::applier&)
    >
    class direct_result_action0 
      : public result_action0<Component, Result, Action, F>
    {
    private:
        typedef result_action0<Component, Result, Action, F> base_type;

    public:
        direct_result_action0()
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        ///
        static Result execute_function(
            applier::applier& appl, naming::address::address_type lva)
        {
            return (get_lva<Component>::call(lva)->*DirectF)(appl);
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  zero parameter version, no result value
    template <
        typename Component, int Action, 
        threads::thread_state(Component::*F)(
            threads::thread_self&, applier::applier&)
    >
    class action0 : public action<Component, Action, boost::fusion::vector<> >
    {
    private:
        typedef action<Component, Action, boost::fusion::vector<> > base_type;

    public:
        action0()
        {}

    public:
        typedef boost::mpl::false_ direct_execution;
        typedef void result_type;

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the action0 type. This is used by the \a applier in 
        /// case no continuation has been supplied.
        static boost::function<threads::thread_function_type> 
        construct_thread_function(applier::applier& appl, 
            naming::address::address_type lva)
        {
            return boost::bind(F, get_lva<Component>::call(lva), _1, 
                boost::ref(appl));
        }

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the action0 type. This is used by the \a applier in 
        /// case a continuation has been supplied
        static boost::function<threads::thread_function_type> 
        construct_thread_function(continuation_type& cont,
            applier::applier& appl, naming::address::address_type lva)
        {
            return base_type::construct_continuation_thread_function(
                boost::bind(F, get_lva<Component>::call(lva), _1, 
                    boost::ref(appl)), 
                appl, cont);
        }

    private:
        boost::function<threads::thread_function_type>
        get_thread_function(applier::applier& appl, 
            naming::address::address_type lva) const
        {
            return construct_thread_function(appl, lva);
        }

        boost::function<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            applier::applier& appl, naming::address::address_type lva) const
        {
            return construct_thread_function(cont, appl, lva);
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    template <
        typename Component, int Action, 
        threads::thread_state(Component::*F)(
            threads::thread_self&, applier::applier&),
        void (Component::*DirectF)(applier::applier&)
    >
    class direct_action0 : public action0<Component, Action, F>
    {
    private:
        typedef action0<Component, Action, F> base_type;

    public:
        direct_action0()
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        ///
        static void execute_function(
            applier::applier& appl, naming::address::address_type lva)
        {
            (get_lva<Component>::call(lva)->*DirectF)(appl);
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    //  one parameter version
    template <
        typename Component, typename Result, int Action, typename T0, 
        threads::thread_state(Component::*F)(
            threads::thread_self&, applier::applier&, Result*, T0)
    >
    class result_action1
      : public action<Component, Action, 
            boost::fusion::vector<typename detail::remove_qualifiers<T0>::type> >
    {
    private:
        typedef 
            action<Component, Action, 
                boost::fusion::vector<typename detail::remove_qualifiers<T0>::type> >
        base_type;

    public:
        result_action1() 
        {}

        // construct an action from its arguments
        template <typename Arg0>
        result_action1(Arg0 const& arg0) 
          : base_type(arg0) 
        {}

    private:
        /// The \a continuation_thread_function will be registered as the thread
        /// function of a thread. It encapsulates the execution of the 
        /// original function (given by \a func), and afterwards triggers all
        /// continuations using the result value obtained from the execution
        /// of the original thread function.
        template <typename Func>
        static threads::thread_state 
        continuation_thread_function(
            threads::thread_self& self, applier::applier& app, 
            continuation_type cont, boost::tuple<Func> func)
        {
            Result result;
            threads::thread_state newstate = 
                boost::get<0>(func)(self, &result);
            cont->trigger_all(app, result);
            return newstate;
        }

        /// The \a construct_continuation_thread_function is a helper function
        /// for constructing the wrapped thread function needed for 
        /// continuation support
        template <typename Func>
        static boost::function<threads::thread_function_type>
        construct_continuation_thread_function(Func func, 
            applier::applier& appl, continuation_type cont) 
        {
            // we need to assign the address of the thread function to a 
            // variable to  help the compiler to deduce the function type
            threads::thread_state (*f)(threads::thread_self&, 
                    applier::applier&, continuation_type, 
                    boost::tuple<Func>) =
                &result_action1::continuation_thread_function;

            // The following bind constructs the wrapped thread function
            //   f:  is the wrapping thread function
            //  _1:  is a placeholder which will be replaced by the reference
            //       to thread_self
            //  app: reference to the applier (pre-bound second argument to f)
            // cont: continuation (pre-bound third argument to f)
            // func: wrapped function object (pre-bound forth argument to f)
            //       (this is embedded into a tuple because boost::bind can't
            //       pre-bind another bound function as an argument)
            return boost::bind(f, _1, boost::ref(appl), cont, 
                boost::make_tuple(func));
        }

    public:
        typedef boost::mpl::false_ direct_execution;
        typedef Result result_type;

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a result_action1 type. This is used by the \a 
        /// applier in case no continuation has been supplied.
        template <typename Arg0>
        static boost::function<threads::thread_function_type> 
        construct_thread_function(applier::applier& appl, 
            naming::address::address_type lva, Arg0 const& arg0) 
        {
            return boost::bind(F, get_lva<Component>::call(lva), _1, 
                boost::ref(appl), reinterpret_cast<Result*>(NULL), arg0);
        }

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a result_action1 type. This is used by the \a 
        /// applier in case a continuation has been supplied
        template <typename Arg0>
        static boost::function<threads::thread_function_type> 
        construct_thread_function(continuation_type& cont,
            applier::applier& appl, naming::address::address_type lva, 
            Arg0 const& arg0) 
        {
            return construct_continuation_thread_function(
                boost::bind(F, get_lva<Component>::call(lva), _1, 
                    boost::ref(appl), _2, arg0), 
                appl, cont);
        }

    private:
        /// This \a get_thread_function will be invoked to retrieve the thread 
        /// function for an action which has to be invoked without continuations.
        boost::function<threads::thread_function_type>
        get_thread_function(applier::applier& appl, 
            naming::address::address_type lva) const
        {
            return construct_thread_function(appl, lva, this->get<0>());
        }

        /// This \a get_thread_function will be invoked to retrieve the thread 
        /// function for an action which has to be invoked with continuations.
        boost::function<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            applier::applier& appl, naming::address::address_type lva) const
        {
            return construct_thread_function(cont, appl, lva, this->get<0>());
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Result, int Action, typename T0, 
        threads::thread_state(Component::*F)(
            threads::thread_self&, applier::applier&, Result*, T0),
        Result (Component::*DirectF)(applier::applier&, T0)
    >
    class direct_result_action1 
      : public result_action1<Component, Result, Action, T0, F>
    {
    private:
        typedef 
            result_action1<Component, Result, Action, T0, F> 
        base_type;

    public:
        direct_result_action1()
        {}

        template <typename Arg0>
        direct_result_action1(Arg0 const& arg0)
          : base_type(arg0)
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        ///
        template <typename Arg0>
        static Result execute_function(
            applier::applier& appl, naming::address::address_type lva,
            Arg0 const& arg0)
        {
            return (get_lva<Component>::call(lva)->*DirectF)(appl, arg0);
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    //  one parameter version, no result value
    template <
        typename Component, int Action, typename T0, 
        threads::thread_state(Component::*F)(
            threads::thread_self&, applier::applier&, T0)
    >
    class action1 
      : public action<Component, Action, 
            boost::fusion::vector<typename detail::remove_qualifiers<T0>::type> >
    {
    private:
        typedef 
            action<Component, Action, 
                boost::fusion::vector<typename detail::remove_qualifiers<T0>::type> >
        base_type;

    public:
        action1() 
        {}

        // construct an action from its arguments
        template <typename Arg0>
        action1(Arg0 const& arg0) 
          : base_type(arg0) 
        {}

    public:
        typedef boost::mpl::false_ direct_execution;
        typedef void result_type;

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a action1 type. This is used by the \a applier in 
        /// case no continuation has been supplied.
        template <typename Arg0>
        static boost::function<threads::thread_function_type> 
        construct_thread_function(applier::applier& appl, 
            naming::address::address_type lva, Arg0 const& arg0) 
        {
            return boost::bind(F, get_lva<Component>::call(lva), _1, 
                boost::ref(appl), arg0);
        }

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a action1 type. This is used by the \a applier in 
        /// case a continuation has been supplied
        template <typename Arg0>
        static boost::function<threads::thread_function_type> 
        construct_thread_function(continuation_type& cont,
            applier::applier& appl, naming::address::address_type lva, 
            Arg0 const& arg0) 
        {
            return base_type::construct_continuation_thread_function(
                boost::bind(F, get_lva<Component>::call(lva), _1, 
                    boost::ref(appl), arg0), 
                appl, cont);
        }

    private:
        ///
        boost::function<threads::thread_function_type>
        get_thread_function(applier::applier& appl, 
            naming::address::address_type lva) const
        {
            return construct_thread_function(appl, lva, this->get<0>());
        }

        ///
        boost::function<threads::thread_function_type>
        get_thread_function(continuation_type& cont,
            applier::applier& appl, naming::address::address_type lva) const
        {
            return construct_thread_function(cont, appl, lva, this->get<0>());
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, int Action, typename T0, 
        threads::thread_state(Component::*F)(
            threads::thread_self&, applier::applier&, T0),
        void (Component::*DirectF)(applier::applier&, T0)
    >
    class direct_action1 : public action1<Component, Action, T0, F>
    {
    private:
        typedef action1<Component, Action, T0, F> base_type;

    public:
        direct_action1()
        {}

        // construct an action from its arguments
        template <typename Arg0>
        direct_action1(Arg0 const& arg0) 
          : base_type(arg0) 
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        ///
        template <typename Arg0>
        static void execute_function(
            applier::applier& appl, naming::address::address_type lva, 
            Arg0 const& arg0)
        {
            (get_lva<Component>::call(lva)->*DirectF)(appl, arg0);
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };

    // bring in the rest of the implementations
    #include <hpx/runtime/actions/action_implementations.hpp>

///////////////////////////////////////////////////////////////////////////////
}}

#endif

