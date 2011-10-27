////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_4DCDC48F_9AC2_4D17_A6A7_FD60AEBB3D56)
#define HPX_4DCDC48F_9AC2_4D17_A6A7_FD60AEBB3D56

#include <bitset>

#include <boost/type_traits/remove_reference.hpp>
#include <boost/type_traits/remove_const.hpp>

#include <boost/mpl/at.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/arithmetic/dec.hpp>

#include <boost/serialization/access.hpp>

#include <hpx/exception.hpp>
#include <hpx/config/function.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/spinlock.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/actions/plain_action.hpp>

#if !defined HPX_CODELET_ARGUMENT_LIMIT
    #define HPX_CODELET_ARGUMENT_LIMIT HPX_ACTION_ARGUMENT_LIMIT
#endif


// TODO: error codes

namespace hpx { namespace lcos
{

template <
    typename Signature
>
struct codelet;

// TODO: specialize handle_gid for this class
template <
    std::size_t M
  , typename Signature
  , typename Argument
>
struct dependency;

template <
    std::size_t M
  , typename Signature
  , typename Argument
>
inline util::function<void(Argument const&)> make_dependency(
    naming::id_type const& id
    )
{
    return util::function<void(Argument const&)>(
        dependency<M, Signature, Argument>(id));
}

template <
    std::size_t M
  , typename Signature
  , typename Argument
>
inline void fulfill(
    naming::id_type const& id
  , Argument const& arg
    )
{
    dependency<M, Signature, Argument> d(id);

    d(arg);
}

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_CODELET_ARGUMENT_LIMIT, "hpx/lcos/codelet.hpp"))              \
    /**/
#include BOOST_PP_ITERATE()

}}

#endif

///////////////////////////////////////////////////////////////////////////////
// Preprocessor vertical repetition code (1st dimension)
///////////////////////////////////////////////////////////////////////////////
#elif BOOST_PP_ITERATION_DEPTH() == 1

#define N BOOST_PP_ITERATION()

template <
    typename Result
  , BOOST_PP_ENUM_PARAMS(N, typename T)
>
struct codelet<Result(BOOST_PP_ENUM_PARAMS(N, T))>
  : components::managed_component_base<
        codelet<Result(BOOST_PP_ENUM_PARAMS(N, T))>
    >
{
    typedef codelet<Result(BOOST_PP_ENUM_PARAMS(N, T))> this_type;

    typedef util::spinlock mutex_type;

    typedef BOOST_PP_CAT(boost::fusion::vector, N)<
        #define BOOST_PP_ITERATION_PARAMS_2                                   \
            (4, (0, BOOST_PP_DEC(N), "hpx/lcos/codelet.hpp", 1))              \
            /**/
        #include BOOST_PP_ITERATE()
    > arguments_type; 

    typedef std::bitset<N> arguments_state_type;

    typedef util::function<Result(BOOST_PP_ENUM_PARAMS(N, T))> action_type;

    typedef util::function<void(Result const&)> callback_type;

    enum actions
    {
        #define BOOST_PP_ITERATION_PARAMS_2                                   \
            (4, (0, BOOST_PP_DEC(N), "hpx/lcos/codelet.hpp", 2))              \
            /**/
        #include BOOST_PP_ITERATE()
    };

    #define BOOST_PP_ITERATION_PARAMS_2                                       \
        (4, (0, BOOST_PP_DEC(N), "hpx/lcos/codelet.hpp", 3))                  \
        /**/
    #include BOOST_PP_ITERATE()

    void initialize(
        action_type const& f
      , callback_type const& cb
        )
    {
        action_ = f;
        callback_ = cb;
    }

  private:
    mutex_type mutex_;    
    arguments_type arguments_;
    arguments_state_type arguments_state_;
    action_type action_;
    callback_type callback_;

    void call_if_ready_locked()
    {
        if (N == arguments_state_.count()) 
        {
            callback_(action_(
                #define BOOST_PP_ITERATION_PARAMS_2                           \
                    (4, (0, BOOST_PP_DEC(N), "hpx/lcos/codelet.hpp", 4))      \
                    /**/
                #include BOOST_PP_ITERATE()
                )); 
        }
    }
}; 

template <
    typename Signature
  , typename Argument
>
struct dependency<BOOST_PP_DEC(N), Signature, Argument>
{
  private:
    naming::id_type id_;

    friend class boost::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, unsigned const)
    {
        ar & id_;
    }

  public:
    dependency(
        naming::id_type const& id = naming::invalid_id
        )
      : id_(id)
    {}

    void operator()(
        Argument const& arg
        ) const
    {
        typedef typename codelet<Signature>::
            BOOST_PP_CAT(BOOST_PP_CAT(set, BOOST_PP_DEC(N)), _action) action_type;
        applier::apply<action_type>(id_, arg);
    }
};

#undef N

///////////////////////////////////////////////////////////////////////////////
// Preprocessor vertical repetition code (2nd dimension)
///////////////////////////////////////////////////////////////////////////////
#elif BOOST_PP_ITERATION_DEPTH() == 2

// strip references from types for the arguments_type typedef in codelet
#if BOOST_PP_ITERATION_FLAGS() == 1

#define M BOOST_PP_ITERATION()

        BOOST_PP_COMMA_IF(M) typename boost::remove_const<
            typename boost::remove_reference<BOOST_PP_CAT(T, M)>::type
        >::type

#undef M

// generate codelet action codes 
#elif BOOST_PP_ITERATION_FLAGS() == 2

#define M BOOST_PP_ITERATION()

        BOOST_PP_COMMA_IF(M) BOOST_PP_CAT(codelet_set, M)

#undef M

// generate codelet action typedefs and action definitions 
#elif BOOST_PP_ITERATION_FLAGS() == 3

#define M BOOST_PP_ITERATION()

    void BOOST_PP_CAT(set, M)(
        typename boost::mpl::at_c<arguments_type, M>::type const& arg
        )
    {
        mutex_type::scoped_lock l(mutex_);

        if (!arguments_state_[M])
        {
            boost::fusion::at_c<M>(arguments_) = arg;
            arguments_state_[M] = 1;
            call_if_ready_locked();
        }

        else
        {
            HPX_THROW_EXCEPTION(
                hpx::bad_parameter
              , "codelet::set" BOOST_PP_STRINGIZE(M)
              , "argument " BOOST_PP_STRINGIZE(M) " is already set");
        }
    }

    typedef hpx::actions::action1<
        this_type
      , BOOST_PP_CAT(codelet_set, M)
      , typename boost::mpl::at_c<arguments_type, M>::type const&
      , &this_type::BOOST_PP_CAT(set, M)
    > BOOST_PP_CAT(BOOST_PP_CAT(set, M), _action);
    /**/

#undef M

// generate codelet function call 
#elif BOOST_PP_ITERATION_FLAGS() == 4

#define M BOOST_PP_ITERATION()

                BOOST_PP_COMMA_IF(M) boost::fusion::at_c<M>(arguments_)

#undef M

#endif

#endif

