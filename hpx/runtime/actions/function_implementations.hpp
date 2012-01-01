////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_64B0848B_3BB0_4564_9FD4_D963AE2E416C)
#define HPX_64B0848B_3BB0_4564_9FD4_D963AE2E416C

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/actions/function_implementations.hpp"))                      \
    /**/

#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()
#define HPX_ARG_TYPES(z, n, data)                                             \
        typedef BOOST_PP_CAT(T, n) BOOST_PP_CAT(BOOST_PP_CAT(arg, n), _type); \
    /**/
#define HPX_REMOVE_QUALIFIERS(z, n, data)                                     \
        BOOST_PP_COMMA_IF(n)                                                  \
        typename detail::remove_qualifiers<BOOST_PP_CAT(T, n)>::type          \
    /**/
#define HPX_PARAM_TYPES(z, n, data)                                           \
        BOOST_PP_COMMA_IF(n)                                                  \
        typename boost::call_traits<BOOST_PP_CAT(data, n)>::param_type        \
        BOOST_PP_CAT(BOOST_PP_CAT(data, n), _)                                \
    /**/
#define HPX_PARAM_ARGUMENT(z, n, data)                                       \
        BOOST_PP_COMMA_IF(n) BOOST_PP_CAT(BOOST_PP_CAT(data, n), _)          \
    /**/

template <typename Result, BOOST_PP_ENUM_PARAMS(N, typename T)>
struct function<Result(BOOST_PP_ENUM_PARAMS(N, T))>
{
    typedef typename detail::function_result<Result>::type result_type;
    typedef boost::fusion::vector<
        BOOST_PP_REPEAT(N, HPX_REMOVE_QUALIFIERS, _)
    > arguments_type;
    typedef signature<result_type, arguments_type> action_type;

    BOOST_PP_REPEAT(N, HPX_ARG_TYPES, _)

    enum { arity = N };

    function() {}

    function(action_type* f_)
    {
        if (f_)
            f.reset(f_);
    }

    function(boost::shared_ptr<action_type> const& f_)
    {
        if (f_)
            f = f_;
    }

    function(function const& other)
    {
        if (other.f)
            f = other.f;
    }

    function& operator=(action_type* f_)
    {
        if (f_)
            f.reset(f_);
        else
            clear();
        return *this;
    }

    function& operator=(boost::shared_ptr<action_type> const& f_)
    {
        if (f_)
            f = f_;
        else
            clear();
        return *this;
    }

    function& operator=(function const& other)
    {
        if (other.f)
            f = other.f;
        else
            clear();
        return *this;
    }

    void swap(function& other)
    { boost::swap(*this, other); }

    void clear()
    { f.reset(); }

    bool empty() const
    { return !f; }

    operator typename util::safe_bool<function>::result_type() const
    { return util::safe_bool<function>()(f); }

    bool operator!() const
    { return !f; }

    result_type operator()(
        BOOST_PP_REPEAT(N, HPX_PARAM_TYPES, T)
    ) const {
        if (HPX_UNLIKELY(!f))
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "function::operator()", "empty action was called");
        }

        return f->execute_function
            (0, BOOST_PP_REPEAT(N, HPX_PARAM_ARGUMENT, T));
    }

  private:
    boost::shared_ptr<action_type> f;

    friend class boost::serialization::access;

    template <typename Archive>
    void save(Archive& ar, const unsigned int) const
    {
        bool not_empty = f;
        ar << not_empty;
        if (not_empty)
            ar << f;
    }

    template <typename Archive>
    void load(Archive& ar, const unsigned int version)
    {
        if (version > HPX_FUNCTION_VERSION)
        {
            HPX_THROW_EXCEPTION(version_too_new,
                "function::load",
                "trying to load function with unknown version");
        }

        bool not_empty = false;
        ar >> not_empty;
        if (not_empty)
            ar >> f;
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

#undef HPX_PARAM_ARGUMENT
#undef HPX_PARAM_TYPES
#undef HPX_REMOVE_QUALIFIERS
#undef HPX_ARG_TYPES
#undef N

#endif

