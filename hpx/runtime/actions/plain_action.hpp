//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_PLAIN_ACTION_NOV_14_2008_0706PM)
#define HPX_RUNTIME_ACTIONS_PLAIN_ACTION_NOV_14_2008_0706PM

#include <cstdlib>
#include <stdexcept>

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/components/server/plain_function.hpp>
#include <hpx/runtime/get_lva.hpp>
#include <hpx/util/serialize_sequence.hpp>

#include <boost/version.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
#define HPX_FUNCTION_ARG_ENUM(z, n, data)                                     \
        BOOST_PP_CAT(function_action_arg, BOOST_PP_INC(n)) =                  \
            function_action_base + BOOST_PP_INC(n),                           \
    /**/
#define HPX_FUNCTION_RETARG_ENUM(z, n, data)                                  \
        BOOST_PP_CAT(function_result_action_arg, BOOST_PP_INC(n)) =           \
            function_result_action_base + BOOST_PP_INC(n),                    \
    /**/

    enum function_action
    {
        /// plain (free) remotely callable function identifiers
        function_action_base = 100,
        function_action_arg0 = function_action_base + 0, 
        BOOST_PP_REPEAT(HPX_PLAIN_ACTION_ARGUMENT_LIMIT, HPX_FUNCTION_ARG_ENUM, _)

        /// plain (free) remotely callable function identifiers with result
        function_result_action_base = 200,
        function_result_action_arg0 = function_result_action_base + 0, 
        BOOST_PP_REPEAT(HPX_PLAIN_ACTION_ARGUMENT_LIMIT, HPX_FUNCTION_RETARG_ENUM, _)
    };

#undef HPX_FUNCTION_RETARG_ENUM
#undef HPX_FUNCTION_ARG_ENUM

    ///////////////////////////////////////////////////////////////////////////
    //  Specialized generic plain (free) action types allowing to hold a 
    //  different number of arguments
    ///////////////////////////////////////////////////////////////////////////

    // zero argument version
    template <
        typename Result, 
        threads::thread_state (*F)(
            threads::thread_self&, applier::applier&, Result*)
    >
    class plain_base_result_action0 
      : public action<
            components::server::plain_function, function_result_action_arg0, 
            boost::fusion::vector<> 
        >
    {
        typedef action<
            components::server::plain_function, function_result_action_arg0, 
            boost::fusion::vector<> 
        > base_type;

    public:
        plain_base_result_action0()
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
            threads::thread_self& self, applier::applier& appl, 
            continuation_type cont, boost::tuple<Func> func)
        {
            threads::thread_state newstate = threads::unknown;
            try {
                Result result;
                newstate = boost::get<0>(func)(self, &result);
                cont->trigger_all(appl, result);
            }
            catch (hpx::exception const& e) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(appl, e);
                return threads::terminated;
            }
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
                &plain_base_result_action0::continuation_thread_function;

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
        /// instantiate the \a base_result_action0 type. This is used by the \a 
        /// applier in case no continuation has been supplied.
        static boost::function<threads::thread_function_type> 
        construct_thread_function(applier::applier& appl, 
            naming::address::address_type lva)
        {
            return boost::bind(F, _1, boost::ref(appl), 
                reinterpret_cast<Result*>(NULL));
        }

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a base_result_action0 type. This is used by the \a 
        /// applier in case a continuation has been supplied
        static boost::function<threads::thread_function_type> 
        construct_thread_function(continuation_type& cont, 
            applier::applier& appl, naming::address::address_type lva)
        {
            return construct_continuation_thread_function(
                boost::bind(F, _1, boost::ref(appl), _2), appl, cont);
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
        typename Result, 
        threads::thread_state(*F)(
            threads::thread_self&, applier::applier&, Result*)
    >
    class plain_result_action0 
      : public plain_base_result_action0<Result, F>
    {
    private:
        typedef plain_base_result_action0<Result, F> base_type;

    public:
        plain_result_action0()
        {}

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_result_action0>();
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
        typename Result, 
        threads::thread_state(*F)(
            threads::thread_self&, applier::applier&, Result*),
        Result (*DirectF)(applier::applier&)
    >
    class plain_direct_result_action0 
      : public plain_base_result_action0<Result, F>
    {
    private:
        typedef plain_base_result_action0<Result, F> base_type;

    public:
        plain_direct_result_action0()
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        ///
        static Result execute_function(
            applier::applier& appl, naming::address::address_type lva)
        {
            return (*DirectF)(appl);
        }

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_direct_result_action0>();
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
        threads::thread_state (*F)(threads::thread_self&, applier::applier&)
    >
    class plain_base_action0 
      : public action<
            components::server::plain_function, function_action_arg0, 
            boost::fusion::vector<> 
        >
    {
    private:
        typedef action<
            components::server::plain_function, function_action_arg0, 
            boost::fusion::vector<> 
        > base_type;

    public:
        plain_base_action0()
        {}

    public:
        typedef boost::mpl::false_ direct_execution;
        typedef void result_type;

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the base_action0 type. This is used by the \a applier in 
        /// case no continuation has been supplied.
        static boost::function<threads::thread_function_type> 
        construct_thread_function(applier::applier& appl, 
            naming::address::address_type lva)
        {
            return boost::bind(F, _1, boost::ref(appl));
        }

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the base_action0 type. This is used by the \a applier in 
        /// case a continuation has been supplied
        static boost::function<threads::thread_function_type> 
        construct_thread_function(continuation_type& cont,
            applier::applier& appl, naming::address::address_type lva)
        {
            return base_type::construct_continuation_thread_function(
                boost::bind(F, _1, boost::ref(appl)), appl, cont);
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

    ///////////////////////////////////////////////////////////////////////////
    template <
        threads::thread_state (*F)(threads::thread_self&, applier::applier&)
    >
    class plain_action0 : public plain_base_action0<F>
    {
    private:
        typedef plain_base_action0<F> base_type;

    public:
        plain_action0()
        {}

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_action0>();
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
        threads::thread_state (*F)(threads::thread_self&, applier::applier&),
        void (*DirectF)(applier::applier&)
    >
    class plain_direct_action0 : public plain_base_action0<F>
    {
    private:
        typedef plain_base_action0<F> base_type;

    public:
        plain_direct_action0()
        {}

    public:
        typedef boost::mpl::true_ direct_execution;

        ///
        static void execute_function(
            applier::applier& appl, naming::address::address_type lva)
        {
            (*DirectF)(appl);
        }

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_direct_action0>();
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
        typename Result, typename T0, 
        threads::thread_state (*F)(
            threads::thread_self&, applier::applier&, Result*, T0) 
    >
    class plain_base_result_action1
      : public action<
            components::server::plain_function, function_result_action_arg1, 
            boost::fusion::vector<typename detail::remove_qualifiers<T0>::type> 
        >
    {
    private:
        typedef action<
            components::server::plain_function, function_result_action_arg1, 
            boost::fusion::vector<typename detail::remove_qualifiers<T0>::type> 
        > base_type;

    public:
        plain_base_result_action1() 
        {}

        // construct an action from its arguments
        template <typename Arg0>
        plain_base_result_action1(Arg0 const& arg0) 
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
            threads::thread_state newstate = threads::unknown;
            try {
                Result result;
                newstate = boost::get<0>(func)(self, &result);
                cont->trigger_all(app, result);
            }
            catch (hpx::exception const& e) {
                // make sure hpx::exceptions are propagated back to the client
                cont->trigger_error(app, e);
                return threads::terminated;
            }
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
                &plain_base_result_action1::continuation_thread_function;

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
        /// instantiate the \a base_result_action1 type. This is used by the \a 
        /// applier in case no continuation has been supplied.
        template <typename Arg0>
        static boost::function<threads::thread_function_type> 
        construct_thread_function(applier::applier& appl, 
            naming::address::address_type lva, Arg0 const& arg0) 
        {
            return boost::bind(F, _1, boost::ref(appl), 
                reinterpret_cast<Result*>(NULL), arg0);
        }

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a base_result_action1 type. This is used by the \a 
        /// applier in case a continuation has been supplied
        template <typename Arg0>
        static boost::function<threads::thread_function_type> 
        construct_thread_function(continuation_type& cont,
            applier::applier& appl, naming::address::address_type lva, 
            Arg0 const& arg0) 
        {
            return construct_continuation_thread_function(
                boost::bind(F, _1, boost::ref(appl), _2, arg0), appl, cont);
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
        typename Result, typename T0, 
        threads::thread_state (*F)(
            threads::thread_self&, applier::applier&, Result*, T0)
    >
    class plain_result_action1 
      : public plain_base_result_action1<Result, T0, F>
    {
    private:
        typedef plain_base_result_action1<Result, T0, F> base_type;

    public:
        plain_result_action1()
        {}

        template <typename Arg0>
        plain_result_action1(Arg0 const& arg0)
          : base_type(arg0)
        {}

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_result_action1>();
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
        typename Result, typename T0, 
        threads::thread_state (*F)(
            threads::thread_self&, applier::applier&, Result*, T0),
        Result (*DirectF)(applier::applier&, T0)
    >
    class plain_direct_result_action1 
      : public plain_base_result_action1<Result, T0, F>
    {
    private:
        typedef plain_base_result_action1<Result, T0, F> base_type;

    public:
        plain_direct_result_action1()
        {}

        template <typename Arg0>
        plain_direct_result_action1(Arg0 const& arg0)
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
            return (*DirectF)(appl, arg0);
        }

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_direct_result_action1>();
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
        typename T0, 
        threads::thread_state (*F)(
            threads::thread_self&, applier::applier&, T0)
    >
    class plain_base_action1 
      : public action<
            components::server::plain_function, function_action_arg1, 
            boost::fusion::vector<typename detail::remove_qualifiers<T0>::type> 
        >
    {
    private:
        typedef action<
            components::server::plain_function, function_action_arg1, 
            boost::fusion::vector<typename detail::remove_qualifiers<T0>::type> 
        > base_type;

    public:
        plain_base_action1() 
        {}

        // construct an action from its arguments
        template <typename Arg0>
        plain_base_action1(Arg0 const& arg0) 
          : base_type(arg0) 
        {}

    public:
        typedef boost::mpl::false_ direct_execution;
        typedef void result_type;

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a base_action1 type. This is used by the \a applier in 
        /// case no continuation has been supplied.
        template <typename Arg0>
        static boost::function<threads::thread_function_type> 
        construct_thread_function(applier::applier& appl, 
            naming::address::address_type lva, Arg0 const& arg0) 
        {
            return boost::bind(F, _1, boost::ref(appl), arg0);
        }

        /// \brief This static \a construct_thread_function allows to construct 
        /// a proper thread function for a \a thread without having to 
        /// instantiate the \a base_action1 type. This is used by the \a applier in 
        /// case a continuation has been supplied
        template <typename Arg0>
        static boost::function<threads::thread_function_type> 
        construct_thread_function(continuation_type& cont,
            applier::applier& appl, naming::address::address_type lva, 
            Arg0 const& arg0) 
        {
            return base_type::construct_continuation_thread_function(
                boost::bind(F, _1, boost::ref(appl), arg0), appl, cont);
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
        typename T0, 
        threads::thread_state (*F)(
            threads::thread_self&, applier::applier&, T0)
    >
    class plain_action1 : public plain_base_action1<T0, F>
    {
    private:
        typedef plain_base_action1<T0, F> base_type;

    public:
        plain_action1()
        {}

        // construct an action from its arguments
        template <typename Arg0>
        plain_action1(Arg0 const& arg0) 
          : base_type(arg0) 
        {}

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_action1>();
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
        typename T0, 
        threads::thread_state (*F)(
            threads::thread_self&, applier::applier&, T0),
        void (*DirectF)(applier::applier&, T0)
    >
    class plain_direct_action1 : public plain_base_action1<T0, F>
    {
    private:
        typedef plain_base_action1<T0, F> base_type;

    public:
        plain_direct_action1()
        {}

        // construct an action from its arguments
        template <typename Arg0>
        plain_direct_action1(Arg0 const& arg0) 
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
            (*DirectF)(appl, arg0);
        }

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        char const* const get_action_name() const
        {
            return detail::get_action_name<plain_direct_action1>();
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
    #include <hpx/runtime/actions/plain_action_implementations.hpp>

///////////////////////////////////////////////////////////////////////////////
}}

#include <hpx/config/warnings_suffix.hpp>

#endif

