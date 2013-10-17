// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


    template <
        typename Action
         
      , typename Result
    >
    struct dataflow_impl<
        Action
         
        ,
        void , void , void , void , void , void , void , void , void , void , void , void , void , void , void , void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                 
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                     
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector0<>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init()
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            hpx::apply_c<Action>(get_gid(), action_id);
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0
        ,
        void , void , void , void , void , void , void , void , void , void , void , void , void , void , void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector1<A0>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(1);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1
        ,
        void , void , void , void , void , void , void , void , void , void , void , void , void , void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector2<A0 , A1>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(2);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2
        ,
        void , void , void , void , void , void , void , void , void , void , void , void , void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector3<A0 , A1 , A2>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(3);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3
        ,
        void , void , void , void , void , void , void , void , void , void , void , void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector4<A0 , A1 , A2 , A3>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(4);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4
        ,
        void , void , void , void , void , void , void , void , void , void , void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector5<A0 , A1 , A2 , A3 , A4>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(5);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4 , A5
        ,
        void , void , void , void , void , void , void , void , void , void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4 , A5
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4 , A5
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector6<A0 , A1 , A2 , A3 , A4 , A5>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | (1<< 5) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(6);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type()); set_slot< 5>( a5 , typename hpx::traits::is_dataflow<A5>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4 , A5 , A6
        ,
        void , void , void , void , void , void , void , void , void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4 , A5 , A6
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4 , A5 , A6
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector7<A0 , A1 , A2 , A3 , A4 , A5 , A6>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | (1<< 5) | (1<< 6) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(7);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type()); set_slot< 5>( a5 , typename hpx::traits::is_dataflow<A5>::type()); set_slot< 6>( a6 , typename hpx::traits::is_dataflow<A6>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7
        ,
        void , void , void , void , void , void , void , void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector8<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | (1<< 5) | (1<< 6) | (1<< 7) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(8);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type()); set_slot< 5>( a5 , typename hpx::traits::is_dataflow<A5>::type()); set_slot< 6>( a6 , typename hpx::traits::is_dataflow<A6>::type()); set_slot< 7>( a7 , typename hpx::traits::is_dataflow<A7>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8
        ,
        void , void , void , void , void , void , void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector9<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | (1<< 5) | (1<< 6) | (1<< 7) | (1<< 8) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(9);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type()); set_slot< 5>( a5 , typename hpx::traits::is_dataflow<A5>::type()); set_slot< 6>( a6 , typename hpx::traits::is_dataflow<A6>::type()); set_slot< 7>( a7 , typename hpx::traits::is_dataflow<A7>::type()); set_slot< 8>( a8 , typename hpx::traits::is_dataflow<A8>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9
        ,
        void , void , void , void , void , void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector10<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | (1<< 5) | (1<< 6) | (1<< 7) | (1<< 8) | (1<< 9) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(10);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type()); set_slot< 5>( a5 , typename hpx::traits::is_dataflow<A5>::type()); set_slot< 6>( a6 , typename hpx::traits::is_dataflow<A6>::type()); set_slot< 7>( a7 , typename hpx::traits::is_dataflow<A7>::type()); set_slot< 8>( a8 , typename hpx::traits::is_dataflow<A8>::type()); set_slot< 9>( a9 , typename hpx::traits::is_dataflow<A9>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10
        ,
        void , void , void , void , void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector11<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | (1<< 5) | (1<< 6) | (1<< 7) | (1<< 8) | (1<< 9) | (1<< 10) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(11);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type()); set_slot< 5>( a5 , typename hpx::traits::is_dataflow<A5>::type()); set_slot< 6>( a6 , typename hpx::traits::is_dataflow<A6>::type()); set_slot< 7>( a7 , typename hpx::traits::is_dataflow<A7>::type()); set_slot< 8>( a8 , typename hpx::traits::is_dataflow<A8>::type()); set_slot< 9>( a9 , typename hpx::traits::is_dataflow<A9>::type()); set_slot< 10>( a10 , typename hpx::traits::is_dataflow<A10>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11
        ,
        void , void , void , void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector12<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | (1<< 5) | (1<< 6) | (1<< 7) | (1<< 8) | (1<< 9) | (1<< 10) | (1<< 11) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(12);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type()); set_slot< 5>( a5 , typename hpx::traits::is_dataflow<A5>::type()); set_slot< 6>( a6 , typename hpx::traits::is_dataflow<A6>::type()); set_slot< 7>( a7 , typename hpx::traits::is_dataflow<A7>::type()); set_slot< 8>( a8 , typename hpx::traits::is_dataflow<A8>::type()); set_slot< 9>( a9 , typename hpx::traits::is_dataflow<A9>::type()); set_slot< 10>( a10 , typename hpx::traits::is_dataflow<A10>::type()); set_slot< 11>( a11 , typename hpx::traits::is_dataflow<A11>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12
        ,
        void , void , void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector13<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | (1<< 5) | (1<< 6) | (1<< 7) | (1<< 8) | (1<< 9) | (1<< 10) | (1<< 11) | (1<< 12) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(13);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type()); set_slot< 5>( a5 , typename hpx::traits::is_dataflow<A5>::type()); set_slot< 6>( a6 , typename hpx::traits::is_dataflow<A6>::type()); set_slot< 7>( a7 , typename hpx::traits::is_dataflow<A7>::type()); set_slot< 8>( a8 , typename hpx::traits::is_dataflow<A8>::type()); set_slot< 9>( a9 , typename hpx::traits::is_dataflow<A9>::type()); set_slot< 10>( a10 , typename hpx::traits::is_dataflow<A10>::type()); set_slot< 11>( a11 , typename hpx::traits::is_dataflow<A11>::type()); set_slot< 12>( a12 , typename hpx::traits::is_dataflow<A12>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13
        ,
        void , void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector14<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | (1<< 5) | (1<< 6) | (1<< 7) | (1<< 8) | (1<< 9) | (1<< 10) | (1<< 11) | (1<< 12) | (1<< 13) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(14);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type()); set_slot< 5>( a5 , typename hpx::traits::is_dataflow<A5>::type()); set_slot< 6>( a6 , typename hpx::traits::is_dataflow<A6>::type()); set_slot< 7>( a7 , typename hpx::traits::is_dataflow<A7>::type()); set_slot< 8>( a8 , typename hpx::traits::is_dataflow<A8>::type()); set_slot< 9>( a9 , typename hpx::traits::is_dataflow<A9>::type()); set_slot< 10>( a10 , typename hpx::traits::is_dataflow<A10>::type()); set_slot< 11>( a11 , typename hpx::traits::is_dataflow<A11>::type()); set_slot< 12>( a12 , typename hpx::traits::is_dataflow<A12>::type()); set_slot< 13>( a13 , typename hpx::traits::is_dataflow<A13>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14
        ,
        void , void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector15<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | (1<< 5) | (1<< 6) | (1<< 7) | (1<< 8) | (1<< 9) | (1<< 10) | (1<< 11) | (1<< 12) | (1<< 13) | (1<< 14) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(15);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type()); set_slot< 5>( a5 , typename hpx::traits::is_dataflow<A5>::type()); set_slot< 6>( a6 , typename hpx::traits::is_dataflow<A6>::type()); set_slot< 7>( a7 , typename hpx::traits::is_dataflow<A7>::type()); set_slot< 8>( a8 , typename hpx::traits::is_dataflow<A8>::type()); set_slot< 9>( a9 , typename hpx::traits::is_dataflow<A9>::type()); set_slot< 10>( a10 , typename hpx::traits::is_dataflow<A10>::type()); set_slot< 11>( a11 , typename hpx::traits::is_dataflow<A11>::type()); set_slot< 12>( a12 , typename hpx::traits::is_dataflow<A12>::type()); set_slot< 13>( a13 , typename hpx::traits::is_dataflow<A13>::type()); set_slot< 14>( a14 , typename hpx::traits::is_dataflow<A14>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15
        ,
        void , void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector16<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | (1<< 5) | (1<< 6) | (1<< 7) | (1<< 8) | (1<< 9) | (1<< 10) | (1<< 11) | (1<< 12) | (1<< 13) | (1<< 14) | (1<< 15) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(16);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type()); set_slot< 5>( a5 , typename hpx::traits::is_dataflow<A5>::type()); set_slot< 6>( a6 , typename hpx::traits::is_dataflow<A6>::type()); set_slot< 7>( a7 , typename hpx::traits::is_dataflow<A7>::type()); set_slot< 8>( a8 , typename hpx::traits::is_dataflow<A8>::type()); set_slot< 9>( a9 , typename hpx::traits::is_dataflow<A9>::type()); set_slot< 10>( a10 , typename hpx::traits::is_dataflow<A10>::type()); set_slot< 11>( a11 , typename hpx::traits::is_dataflow<A11>::type()); set_slot< 12>( a12 , typename hpx::traits::is_dataflow<A12>::type()); set_slot< 13>( a13 , typename hpx::traits::is_dataflow<A13>::type()); set_slot< 14>( a14 , typename hpx::traits::is_dataflow<A14>::type()); set_slot< 15>( a15 , typename hpx::traits::is_dataflow<A15>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16
        ,
        void , void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector17<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | (1<< 5) | (1<< 6) | (1<< 7) | (1<< 8) | (1<< 9) | (1<< 10) | (1<< 11) | (1<< 12) | (1<< 13) | (1<< 14) | (1<< 15) | (1<< 16) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(17);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type()); set_slot< 5>( a5 , typename hpx::traits::is_dataflow<A5>::type()); set_slot< 6>( a6 , typename hpx::traits::is_dataflow<A6>::type()); set_slot< 7>( a7 , typename hpx::traits::is_dataflow<A7>::type()); set_slot< 8>( a8 , typename hpx::traits::is_dataflow<A8>::type()); set_slot< 9>( a9 , typename hpx::traits::is_dataflow<A9>::type()); set_slot< 10>( a10 , typename hpx::traits::is_dataflow<A10>::type()); set_slot< 11>( a11 , typename hpx::traits::is_dataflow<A11>::type()); set_slot< 12>( a12 , typename hpx::traits::is_dataflow<A12>::type()); set_slot< 13>( a13 , typename hpx::traits::is_dataflow<A13>::type()); set_slot< 14>( a14 , typename hpx::traits::is_dataflow<A14>::type()); set_slot< 15>( a15 , typename hpx::traits::is_dataflow<A15>::type()); set_slot< 16>( a16 , typename hpx::traits::is_dataflow<A16>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17
        ,
        void , void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector18<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | (1<< 5) | (1<< 6) | (1<< 7) | (1<< 8) | (1<< 9) | (1<< 10) | (1<< 11) | (1<< 12) | (1<< 13) | (1<< 14) | (1<< 15) | (1<< 16) | (1<< 17) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(18);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type()); set_slot< 5>( a5 , typename hpx::traits::is_dataflow<A5>::type()); set_slot< 6>( a6 , typename hpx::traits::is_dataflow<A6>::type()); set_slot< 7>( a7 , typename hpx::traits::is_dataflow<A7>::type()); set_slot< 8>( a8 , typename hpx::traits::is_dataflow<A8>::type()); set_slot< 9>( a9 , typename hpx::traits::is_dataflow<A9>::type()); set_slot< 10>( a10 , typename hpx::traits::is_dataflow<A10>::type()); set_slot< 11>( a11 , typename hpx::traits::is_dataflow<A11>::type()); set_slot< 12>( a12 , typename hpx::traits::is_dataflow<A12>::type()); set_slot< 13>( a13 , typename hpx::traits::is_dataflow<A13>::type()); set_slot< 14>( a14 , typename hpx::traits::is_dataflow<A14>::type()); set_slot< 15>( a15 , typename hpx::traits::is_dataflow<A15>::type()); set_slot< 16>( a16 , typename hpx::traits::is_dataflow<A16>::type()); set_slot< 17>( a17 , typename hpx::traits::is_dataflow<A17>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18
        ,
        void
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector19<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | (1<< 5) | (1<< 6) | (1<< 7) | (1<< 8) | (1<< 9) | (1<< 10) | (1<< 11) | (1<< 12) | (1<< 13) | (1<< 14) | (1<< 15) | (1<< 16) | (1<< 17) | (1<< 18) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17 , A18 const & a18)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(19);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type()); set_slot< 5>( a5 , typename hpx::traits::is_dataflow<A5>::type()); set_slot< 6>( a6 , typename hpx::traits::is_dataflow<A6>::type()); set_slot< 7>( a7 , typename hpx::traits::is_dataflow<A7>::type()); set_slot< 8>( a8 , typename hpx::traits::is_dataflow<A8>::type()); set_slot< 9>( a9 , typename hpx::traits::is_dataflow<A9>::type()); set_slot< 10>( a10 , typename hpx::traits::is_dataflow<A10>::type()); set_slot< 11>( a11 , typename hpx::traits::is_dataflow<A11>::type()); set_slot< 12>( a12 , typename hpx::traits::is_dataflow<A12>::type()); set_slot< 13>( a13 , typename hpx::traits::is_dataflow<A13>::type()); set_slot< 14>( a14 , typename hpx::traits::is_dataflow<A14>::type()); set_slot< 15>( a15 , typename hpx::traits::is_dataflow<A15>::type()); set_slot< 16>( a16 , typename hpx::traits::is_dataflow<A16>::type()); set_slot< 17>( a17 , typename hpx::traits::is_dataflow<A17>::type()); set_slot< 18>( a18 , typename hpx::traits::is_dataflow<A18>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
    template <
        typename Action
        , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19
      , typename Result
    >
    struct dataflow_impl<
        Action
        , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19
        
        
      , Result
    >
        : ::hpx::lcos::base_lco_with_value<
              typename traits::promise_remote_result<Result>::type
            , typename Action::result_type
          >
    {
        typedef
            typename traits::promise_remote_result<Result>::type
            result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19
            >
            wrapped_type;
        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19
                >
            >
            wrapping_type;
        typedef
            passed_args_transforms<
                boost::mpl::vector20<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19>
            >
            passed_args;
        typedef typename passed_args::results_type args_type;
        typedef typename passed_args::slot_to_args_map slot_to_args_map;
        
        static const boost::uint32_t
            slots_completed = ((1<< 0) | (1<< 1) | (1<< 2) | (1<< 3) | (1<< 4) | (1<< 5) | (1<< 6) | (1<< 7) | (1<< 8) | (1<< 9) | (1<< 10) | (1<< 11) | (1<< 12) | (1<< 13) | (1<< 14) | (1<< 15) | (1<< 16) | (1<< 17) | (1<< 18) | (1<< 19) | 0);
        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
            , slots_set(0)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }
        void init(A0 const & a0 , A1 const & a1 , A2 const & a2 , A3 const & a3 , A4 const & a4 , A5 const & a5 , A6 const & a6 , A7 const & a7 , A8 const & a8 , A9 const & a9 , A10 const & a10 , A11 const & a11 , A12 const & a12 , A13 const & a13 , A14 const & a14 , A15 const & a15 , A16 const & a16 , A17 const & a17 , A18 const & a18 , A19 const & a19)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(20);
            set_slot< 0>( a0 , typename hpx::traits::is_dataflow<A0>::type()); set_slot< 1>( a1 , typename hpx::traits::is_dataflow<A1>::type()); set_slot< 2>( a2 , typename hpx::traits::is_dataflow<A2>::type()); set_slot< 3>( a3 , typename hpx::traits::is_dataflow<A3>::type()); set_slot< 4>( a4 , typename hpx::traits::is_dataflow<A4>::type()); set_slot< 5>( a5 , typename hpx::traits::is_dataflow<A5>::type()); set_slot< 6>( a6 , typename hpx::traits::is_dataflow<A6>::type()); set_slot< 7>( a7 , typename hpx::traits::is_dataflow<A7>::type()); set_slot< 8>( a8 , typename hpx::traits::is_dataflow<A8>::type()); set_slot< 9>( a9 , typename hpx::traits::is_dataflow<A9>::type()); set_slot< 10>( a10 , typename hpx::traits::is_dataflow<A10>::type()); set_slot< 11>( a11 , typename hpx::traits::is_dataflow<A11>::type()); set_slot< 12>( a12 , typename hpx::traits::is_dataflow<A12>::type()); set_slot< 13>( a13 , typename hpx::traits::is_dataflow<A13>::type()); set_slot< 14>( a14 , typename hpx::traits::is_dataflow<A14>::type()); set_slot< 15>( a15 , typename hpx::traits::is_dataflow<A15>::type()); set_slot< 16>( a16 , typename hpx::traits::is_dataflow<A16>::type()); set_slot< 17>( a17 , typename hpx::traits::is_dataflow<A17>::type()); set_slot< 18>( a18 , typename hpx::traits::is_dataflow<A18>::type()); set_slot< 19>( a19 , typename hpx::traits::is_dataflow<A19>::type());
        }
        void finalize()
        {
            data_type d;
            result.read(d);
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }
        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
            BOOST_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }
        typedef typename Action::result_type remote_result;
        
        
        
        void set_value(BOOST_RV_REF(remote_result) r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(boost::move(r));
            forward_results(tmp);
        }
        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }
            
            
            
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp = r;
                hpx::apply<action_type>(t[i], boost::move(tmp));
            }
        }
        
        
        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";
            lcos::local::spinlock::scoped_lock l(mtx);
            if(!result.is_empty())
            {
                data_type d;
                result.read(d);
                l.unlock();
                if(d.stores_error())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    BOOST_ASSERT(d.stores_value()); 
                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }
        
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename util::decay<A>::type
                dataflow_type;
            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;
            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;
            component_type * c = new component_type(this, boost::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A) a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = boost::forward<A>(a);
            maybe_apply<Slot>();
        };
        
        template <
            int Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(BOOST_FWD_REF(A), boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };
        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<Slot);
                    apply_it = (slots_set == slots_completed);
                }
            }
            if(apply_it)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , slots
                );
                update_fired_count();
            }
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::maybe_apply(): "
                << get_gid()
                << " args set: "
                << slots_set
                << "("
                << slots_completed
                << ")"
                << "\n"
                ;
        }
        void set_event()
        {
            this->set_value_nonvirt(remote_result());
        }
        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }
        naming::id_type get_gid() const
        {
            return
                naming::id_type(
                    naming::detail::get_stripped_gid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }
        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }
    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;
        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }
        components::managed_component<dataflow_impl>* back_ptr_;
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
        hpx::lcos::detail::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;
        lcos::local::spinlock & mtx;
    };
