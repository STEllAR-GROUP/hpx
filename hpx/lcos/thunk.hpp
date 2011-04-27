//  Copyright (c) 2010-2011 Dylan Stark, Phillip LeBlanc
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_THUNK_JUL_11_2010_1025PM)
#define HPX_LCOS_THUNK_JUL_11_2010_1025PM

#include <hpx/hpx_fwd.hpp>

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
  template <typename Action>
  class thunk : public lcos::base_lco_with_value<typename Action::result_type>
  {
  //////////////////////////////////////////////////////////////////////////////
  public:
    typedef typename Action::result_type result_type;
    typedef boost::exception_ptr error_type;
    typedef boost::variant<result_type, error_type> feb_data_type;

  //////////////////////////////////////////////////////////////////////////////
  // Component setup
  public:
    typedef components::managed_component<detail::thunk<Action> > wrapping_type;

    enum action
    {
      thunk_trigger = 0,
      thunk_set = 1,
      thunk_get = 2,
    };

    enum { value = components::component_thunk };

    void set_gid(naming::id_type const& gid)
    {
      gid_ = gid;
    }

    naming::id_type const& get_gid(void) const
    {
      return gid_;
    }

  ////////////////////////////////////////////////////////////////////////////
  // LCO with value interface
  public:
    void set_event(void)
    {
      LLCO_(info) << "thunk::set_event()";
    }
   
    void set_result(result_type const& result)
    {
      set(result);
    }

    void set_error(error_type const& e)
    {
      data_.set(feb_data_type(e));
    }

    result_type get_value()
    {
      return get();
    }

  //////////////////////////////////////////////////////////////////////////////
  // Zero Thunk LCO implementation
  private:
    thunk* this_() { return this; }

    void invoke(naming::id_type const& gid)
    {
      hpx::applier::apply_c<Action>(this->get_gid(), gid);
    }

    template <typename Arg0>
    void invoke1(naming::id_type const& gid, Arg0 arg0)
    {
      hpx::applier::apply_c<Action>(this->get_gid(), gid, arg0);
    }

  public:
    thunk(naming::id_type const& target)
      : closure_(boost::bind(&thunk::invoke, this_(), target)),
        gid_(naming::invalid_id),
        was_triggered_(false)
    {}

    template <typename Arg0>
    thunk(naming::id_type const& target, Arg0 const& arg0)
      : closure_(boost::bind(&thunk::template invoke1<Arg0>, this_(), 
            target, arg0)),
        gid_(naming::invalid_id),
        was_triggered_(false)
    {}

    #include <hpx/lcos/thunk_constructors.hpp>

    void trigger(void)
    {
      // Trigger the action if not already done
      scoped_lock l(this);

      if (!was_triggered_)
      {
        closure_();
        was_triggered_ = true;
      }
    }

    void set(result_type value)
    {
      if (data_.is_empty())
        data_.set(feb_data_type(value));
    }

    result_type get(void)
    {
      // Trigger the action if not already done
      {
        scoped_lock l(this);
        if (!was_triggered_)
        {
          LLCO_(info) << "Triggering from get()";

          closure_();
          was_triggered_ = true;
        }
      }

      // Read FEB-protected data, this might yield
      feb_data_type d;
      data_.read(d);

      // Check for error before returning value
      if (1 == d.which())
      {
        error_type e = boost::get<error_type>(d);
        boost::rethrow_exception(e);
      }

      return boost::get<result_type>(d);
    }

  //////////////////////////////////////////////////////////////////////////////
  private:
    struct tag {};
    typedef hpx::util::spinlock_pool<tag> mutex_type;
    typedef typename mutex_type::scoped_lock scoped_lock;

    boost::function<void()> closure_;
    naming::id_type gid_;
    bool was_triggered_;
    util::full_empty<feb_data_type> data_;
  };
}}} // hpx::lcos::detail

namespace hpx { namespace lcos
{
  template <typename Thunk>
  class thunk_client
  {
  //////////////////////////////////////////////////////////////////////////////
  // Component setup
  public:
    typedef Thunk wrapped_type;
    typedef components::managed_component<wrapped_type> wrapping_type;

    naming::id_type get_gid(void)
    {
      naming::gid_type gid = impl_->get_base_gid();
      naming::strip_credit_from_gid(gid);
      return naming::id_type(gid, naming::id_type::unmanaged);
    }

  private:
    boost::shared_ptr<wrapping_type> impl_;

  //////////////////////////////////////////////////////////////////////////////
  // Thunk client interface
  public:
    thunk_client(naming::id_type target)
      : impl_(new wrapping_type(new wrapped_type(target)))
    {
      (*impl_)->set_gid(get_gid());
    }

    template <typename Arg0>
    thunk_client(naming::id_type target, Arg0 arg0)
      : impl_(new wrapping_type(new wrapped_type(target, arg0)))
    {
      (*impl_)->set_gid(get_gid());
    }

    #include <hpx/lcos/thunk_client_constructors.hpp>
  };
}}; // hpx::lcos

////////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_THUNK(thunk_type)                                         \
    typedef hpx::actions::action0<                                             \
      thunk_type,                                                              \
      thunk_type::thunk_trigger,                                               \
      &thunk_type::trigger                                                     \
    > BOOST_PP_CAT(thunk_type,_trigger_action);                                \
                                                                               \
    typedef hpx::actions::action1<                                             \
      thunk_type,                                                              \
      thunk_type::thunk_set,                                                   \
      thunk_type::result_type,                                                 \
      &thunk_type::set                                                         \
    > BOOST_PP_CAT(thunk_type,_set_action);                                    \
                                                                               \
    typedef hpx::actions::result_action0<                                      \
      thunk_type,                                                              \
      thunk_type::result_type,                                                 \
      thunk_type::thunk_get,                                                   \
      &thunk_type::get                                                         \
    > BOOST_PP_CAT(thunk_type,_get_action);                                    \
                                                                               \
    HPX_DEFINE_GET_ACTION_NAME(thunk_type);                                    \
    HPX_REGISTER_ACTION_EX(BOOST_PP_CAT(thunk_type,_trigger_action),           \
                           BOOST_PP_CAT(thunk_type,_trigger_action));          \
    HPX_REGISTER_ACTION_EX(BOOST_PP_CAT(thunk_type,_set_action),               \
                           BOOST_PP_CAT(thunk_type,_set_action));              \
    HPX_REGISTER_ACTION_EX(BOOST_PP_CAT(thunk_type,_get_action),               \
                           BOOST_PP_CAT(thunk_type,_get_action));              \
    HPX_DEFINE_GET_COMPONENT_TYPE(thunk_type);                                 \

#endif

