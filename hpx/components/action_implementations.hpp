//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_COMPONENTS_ACTION_IMPLEMENTATIONS_MAY_20_2008_1104AM)
#define HPX_COMPONENTS_ACTION_IMPLEMENTATIONS_MAY_20_2008_1104AM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/components/action_implementations.hpp"))                             \
    /**/
    
#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()
#define HPX_ACTION_ARGUMENT(z, n, data) BOOST_PP_COMMA_IF(n) this->get<n>()
#define HPX_REMOVE_QULIFIERS(z, n, data)                                      \
        BOOST_PP_COMMA_IF(n)                                                  \
        typename detail::remove_qualifiers<BOOST_PP_CAT(T, n)>::type          \
    /**/
#define HPX_GUID_ARGUMENT1(z, n, data) (typename BOOST_PP_CAT(T, n))
#define HPX_GUID_ARGUMENT2(z, n, data) (BOOST_PP_CAT(T, n))

    ///////////////////////////////////////////////////////////////////////////
    //  N parameter version
    template <
        typename Component, typename Result, int Action, 
        BOOST_PP_ENUM_PARAMS(N, typename T),
        threadmanager::thread_state(Component::*F)(
            threadmanager::px_thread_self&, applier::applier&, Result*,
            BOOST_PP_ENUM_PARAMS(N, T)) 
    >
    class BOOST_PP_CAT(result_action, N)
      : public action<
            Component, Action, 
            boost::fusion::vector<BOOST_PP_REPEAT(N, HPX_REMOVE_QULIFIERS, _)> 
        >
    {
    private:
        typedef action<
            Component, Action, 
            boost::fusion::vector<BOOST_PP_REPEAT(N, HPX_REMOVE_QULIFIERS, _)> 
        > base_type;

    public:
        BOOST_PP_CAT(result_action, N)() 
        {}

        // construct an action from its arguments
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(result_action, N)(
                BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg)) 
          : base_type(BOOST_PP_ENUM_PARAMS(N, arg)) 
        {}

    private:
        // The continuation_thread_function will be registered as the thread
        // function of a px_thread. It encapsulates the execution of the 
        // original function (given by func)
        template <typename Func>
        static threadmanager::thread_state 
        continuation_thread_function(
            threadmanager::px_thread_self& self, applier::applier& app, 
            components::continuation_type cont, boost::tuple<Func> func)
        {
            Result result;
            threadmanager::thread_state newstate = 
                boost::get<0>(func)(self, &result);
            cont->trigger_all(self, app, result);
            return newstate;
        }

        template <typename Func>
        static boost::function<threadmanager::thread_function_type>
        construct_continuation_thread_function(Func func, 
            applier::applier& appl, components::continuation_type cont) 
        {
            // we need to assign the address of the thread function to a 
            // variable to  help the compiler to deduce the function type
            threadmanager::thread_state (*f)(threadmanager::px_thread_self&, 
                    applier::applier&, components::continuation_type, 
                    boost::tuple<Func>) =
                &BOOST_PP_CAT(result_action, N)::continuation_thread_function;

            // The following bind constructs the wrapped thread function
            //   f:  is the wrapping thread function
            //  _1:  is a placeholder which will be replaced by the reference
            //       to px_thread_self
            //  app: reference to the applier (pre-bound second argument to f)
            // cont: continuation (pre-bound third argument to f)
            // func: wrapped function object (pre-bound forth argument to f)
            //       (this is embedded into a tuple because boost::bind can't
            //       pre-bind another bound function as an argument)
            return boost::bind(f, _1, boost::ref(appl), cont, 
                boost::make_tuple(func));
        }

    public:
        // This static construct_thread_function allows to construct 
        // a proper thread function for a px_thread without having to 
        // instantiate the action0 type. This is used by the applier in 
        // case no continuation has been supplied.
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static boost::function<threadmanager::thread_function_type> 
        construct_thread_function(applier::applier& appl, 
            naming::address::address_type lva, 
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg)) 
        {
            return boost::bind(F, reinterpret_cast<Component*>(lva), _1, 
                boost::ref(appl), reinterpret_cast<Result*>(NULL), 
                BOOST_PP_ENUM_PARAMS(N, arg));
        }

        // This static construct_thread_function allows to construct 
        // a proper thread function for a px_thread without having to 
        // instantiate the action0 type. This is used by the applier in 
        // case a continuation has been supplied
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static boost::function<threadmanager::thread_function_type> 
        construct_thread_function(components::continuation_type cont,
            applier::applier& appl, naming::address::address_type lva, 
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg)) 
        {
            return construct_continuation_thread_function(
                boost::bind(F, reinterpret_cast<Component*>(lva), _1, 
                    boost::ref(appl), _2, 
                    BOOST_PP_ENUM_PARAMS(N, arg)), appl, cont);
        }

    private:
        // This get_thread_function will be invoked to retrieve the thread 
        // function for an action which has to be invoked without continuations.
        boost::function<threadmanager::thread_function_type>
        get_thread_function(applier::applier& appl, 
            naming::address::address_type lva) const
        {
            return construct_thread_function(appl, lva, 
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, _));
        }

        // This get_thread_function will be invoked to retrieve the thread 
        // function for an action which has to be invoked with continuations.
        boost::function<threadmanager::thread_function_type>
        get_thread_function(components::continuation_type cont,
            applier::applier& appl, naming::address::address_type lva) const
        {
            return construct_thread_function(cont, appl, lva, 
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, _));
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

    //  N parameter version, no result type
    template <
        typename Component, int Action, BOOST_PP_ENUM_PARAMS(N, typename T),
        threadmanager::thread_state(Component::*F)(
            threadmanager::px_thread_self&, applier::applier&, 
            BOOST_PP_ENUM_PARAMS(N, T)) 
    >
    class BOOST_PP_CAT(action, N)
      : public action<
            Component, Action, 
            boost::fusion::vector<BOOST_PP_REPEAT(N, HPX_REMOVE_QULIFIERS, _)> 
        >
    {
    private:
        typedef action<
            Component, Action, 
            boost::fusion::vector<BOOST_PP_REPEAT(N, HPX_REMOVE_QULIFIERS, _)> 
        > base_type;

    public:
        BOOST_PP_CAT(action, N)() 
        {}

        // construct an action from its arguments
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        BOOST_PP_CAT(action, N)(
                BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg)) 
          : base_type(BOOST_PP_ENUM_PARAMS(N, arg)) 
        {}

    public:
        // This static construct_thread_function allows to construct 
        // a proper thread function for a px_thread without having to 
        // instantiate the action0 type. This is used by the applier in 
        // case no continuation has been supplied.
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static boost::function<threadmanager::thread_function_type> 
        construct_thread_function(applier::applier& appl, 
            naming::address::address_type lva, 
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg)) 
        {
            return boost::bind(F, reinterpret_cast<Component*>(lva), _1, 
                boost::ref(appl), BOOST_PP_ENUM_PARAMS(N, arg));
        }

        // This static construct_thread_function allows to construct 
        // a proper thread function for a px_thread without having to 
        // instantiate the action0 type. This is used by the applier in 
        // case a continuation has been supplied
        template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        static boost::function<threadmanager::thread_function_type> 
        construct_thread_function(components::continuation_type cont,
            applier::applier& appl, naming::address::address_type lva, 
            BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg)) 
        {
            return base_type::construct_continuation_thread_function(
                boost::bind(F, reinterpret_cast<Component*>(lva), _1, 
                    boost::ref(appl), BOOST_PP_ENUM_PARAMS(N, arg)), 
                    appl, cont);
        }

    private:
        boost::function<threadmanager::thread_function_type>
            get_thread_function(applier::applier& appl, 
                naming::address::address_type lva) const
        {
            return construct_thread_function(appl, lva, 
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, _));
        }

        ///
        boost::function<threadmanager::thread_function_type>
        get_thread_function(components::continuation_type cont,
            applier::applier& appl, naming::address::address_type lva) const
        {
            return construct_thread_function(cont, appl, lva, 
                BOOST_PP_REPEAT(N, HPX_ACTION_ARGUMENT, _));
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

#undef HPX_GUID_ARGUMENT1
#undef HPX_GUID_ARGUMENT2
#undef HPX_REMOVE_QULIFIERS
#undef HPX_ACTION_ARGUMENT
#undef N

#endif
