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

#include <hpx/exception.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/future_value.hpp>
#include <hpx/util/safe_bool.hpp>

#define HPX_FUNCTION_VERSION 0x10

namespace hpx { namespace actions
{

template <typename Signature>
struct function;

template <typename Result>
struct function<Result()>
{
    typedef boost::fusion::vector<> arguments_type;
    typedef base_argument_action<arguments_type> action_type;

    typedef Result result_type;

    typedef Result sync_result_type;
    typedef lcos::future_value<Result> async_result_type;

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

    async_result_type eval_async() const
    {
        if (!f)
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "function::eval", "eval called on an empty function");
        }

        async_result_type r;
        continuation_type c = boost::make_shared<continuation>(r.get_gid()); 

        if (base_action::direct_action == f->get_action_type()
            || !is_system_running())
            f->get_thread_function(c, 0, arguments_type())
                (threads::thread_state_ex(threads::wait_signaled));

        else
        {
            threads::thread_init_data data;
            applier::get_applier().get_thread_manager().register_work
                ( f->get_thread_init_data(c, 0, data, arguments_type())
                , threads::thread_state(threads::pending));
        }

        return r;
    }

    sync_result_type eval_sync() const
    { return eval_async().get(); }

    sync_result_type operator()() const
    { return eval_async().get(); }

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

}}

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

