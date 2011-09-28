////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach and Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_61E85C50_FA0E_4A4F_B581_8FAFC61BE00E)
#define HPX_61E85C50_FA0E_4A4F_B581_8FAFC61BE00E

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/tracking.hpp>

#include <hpx/config.hpp>
#include <hpx/state.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/util/safe_bool.hpp>
#include <hpx/util/unused.hpp>

#define HPX_FUNCTION_VERSION 0x10

namespace hpx { namespace actions
{

namespace detail
{

template <typename T>
struct function_result
    : boost::mpl::if_<boost::is_same<T, void>, util::unused_type, T>
 {};

}

template <typename Signature>
struct function;

template <typename Result>
struct function<Result()>
{
    typedef typename detail::function_result<Result>::type result_type;
    typedef boost::fusion::vector<> arguments_type;
    typedef signature<result_type, arguments_type> action_type;

    enum { arity = 0 };

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

    result_type operator()() const
    {
        if (HPX_UNLIKELY(!f))
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "function::operator()", "empty action was called");
        }

        return f->execute_function(0); 
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

// bring in the rest of the implementations
#include <hpx/runtime/actions/function_implementations.hpp>

}

using hpx::actions::function;

}

namespace boost { namespace serialization
{

template <typename Signature>
struct tracking_level<hpx::actions::function<Signature> >
{
    typedef mpl::integral_c_tag tag;
    typedef mpl::int_<track_never> type;
    BOOST_STATIC_CONSTANT(int, value = track_never);
};

template <typename Signature>
struct version<hpx::actions::function<Signature> >
{
    typedef mpl::integral_c_tag tag;
    typedef mpl::int_<HPX_FUNCTION_VERSION> type;
    BOOST_STATIC_CONSTANT(int, value = HPX_FUNCTION_VERSION);
};

}}

#endif // HPX_61E85C50_FA0E_4A4F_B581_8FAFC61BE00E

