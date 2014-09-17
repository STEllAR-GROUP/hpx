//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Adapted from Jared Hoberock's is_call_possible here:
// https://github.com/jaredhoberock/is_call_possible

// Inspired by Roman Perepelitsa's presentation from comp.lang.c++.moderated
// based on the implementation here: http://www.rsdn.ru/forum/cpp/2759773.1.aspx

#if !defined(HPX_TRAITS_HAS_CALL_OPERATOR_APR_14_2013_0256PM)
#define HPX_TRAITS_HAS_CALL_OPERATOR_APR_14_2013_0256PM

#include <boost/preprocessor/comma_if.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/type_traits/detail/yes_no_type.hpp>
#include <boost/type_traits/add_reference.hpp>

namespace hpx { namespace traits { namespace detail
{
    // main template declaration
    template <typename T, typename Signature>
    class has_call_operator;

#define HPX_HAS_CALL_OPERATOR(Z, N, D)                                        \
    template <typename T, typename Result                                     \
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)>               \
    class has_call_operator<T, Result(BOOST_PP_ENUM_PARAMS(N, A))>            \
    {                                                                         \
        struct base_mixin                                                     \
        {                                                                     \
            Result operator()(BOOST_PP_ENUM_PARAMS(N, A));                    \
        };                                                                    \
        struct base : public T, public base_mixin {};                         \
        template <typename U, U t>  class helper{};                           \
        template <typename U>                                                 \
        static boost::type_traits::no_type                                    \
            deduce(U*, helper<Result (base_mixin::*)(                         \
                    BOOST_PP_ENUM_PARAMS(N, A)),                              \
                &U::operator()>* = 0);                                        \
        static boost::type_traits::yes_type deduce(...);                      \
    public:                                                                   \
        static const bool value = sizeof(boost::type_traits::yes_type) ==     \
            sizeof(deduce(static_cast<base*>(0)));                            \
    };                                                                        \
/**/

    BOOST_PP_REPEAT(HPX_FUNCTION_ARGUMENT_LIMIT, HPX_HAS_CALL_OPERATOR, _)

#undef HPX_HAS_CALL_OPERATOR

    template <typename T> class void_exp_result {};

    template <typename T, typename U>
    U const& operator,(U const&, void_exp_result<T>);

    template <typename T, typename U>
    U& operator,(U&, void_exp_result<T>);

    template <typename src_type, typename dest_type>
    struct clone_constness
    {
        typedef dest_type type;
    };

    template <typename src_type, typename dest_type>
    struct clone_constness<const src_type, dest_type>
    {
        typedef const dest_type type;
    };
}}}

namespace hpx { namespace traits
{
    template <typename T, typename Signature>
    struct has_call_operator
    {
    private:
        struct derived : public T
        {
            using T::operator();
            boost::type_traits::no_type operator()(...) const;
        };

        typedef typename detail::clone_constness<T, derived>::type derived_type;

        template <typename U, typename Result>
        struct return_value_check
        {
            static boost::type_traits::yes_type deduce(Result);
            static boost::type_traits::no_type deduce(...);
            static boost::type_traits::no_type deduce(boost::type_traits::no_type);
            static boost::type_traits::no_type deduce(detail::void_exp_result<T>);
        };

        template <typename U>
        struct return_value_check<U, void>
        {
            static boost::type_traits::yes_type deduce(...);
            static boost::type_traits::no_type deduce(boost::type_traits::no_type);
        };

        template <bool has_the_member_of_interest, typename F>
        struct impl
        {
            static const bool value = false;
        };

#define HPX_STATIC_REFERENCE_ARG(Z, NN, D)                                    \
    static typename boost::add_reference<BOOST_PP_CAT(A, NN)>::type           \
        BOOST_PP_CAT(arg, NN);                                                \
/**/

#define HPX_STATIC_ARG(Z, NN, D) BOOST_PP_COMMA_IF(NN) BOOST_PP_CAT(arg, NN)  \
/**/

#define HPX_HAS_CALL_OPERATOR_IMPL(Z, N, D)                                   \
    template <typename Result                                                 \
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)>               \
    struct impl<true, Result(BOOST_PP_ENUM_PARAMS(N, A))>                     \
    {                                                                         \
        static typename boost::add_reference<derived_type>::type test_me;     \
        BOOST_PP_REPEAT_ ## Z(N, HPX_STATIC_REFERENCE_ARG, _)                 \
        static const bool value = sizeof(                                     \
            return_value_check<T, Result>::deduce((                           \
                test_me.operator()(                                           \
                    BOOST_PP_REPEAT_ ## Z(N, HPX_STATIC_ARG, _)               \
                ), detail::void_exp_result<T>())                              \
            )) == sizeof(boost::type_traits::yes_type);                       \
    };                                                                        \
/**/

    BOOST_PP_REPEAT(HPX_FUNCTION_ARGUMENT_LIMIT, HPX_HAS_CALL_OPERATOR_IMPL, _)

#undef HPX_HAS_CALL_OPERATOR_IMPL
#undef HPX_STATIC_ARG
#undef HPX_STATIC_REFERENCE_ARG

    public:
        static const bool value = impl<
            detail::has_call_operator<T, Signature>::value, Signature
        >::value;
    };
}}

#endif
