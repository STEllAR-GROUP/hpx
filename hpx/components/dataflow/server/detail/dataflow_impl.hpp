//  Copyright (c) 2011-2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_SERVER_DETAIL_DATAFLOW_IMPL_HPP
#define HPX_LCOS_DATAFLOW_SERVER_DETAIL_DATAFLOW_IMPL_HPP

#include <hpx/config.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/next.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/utility/enable_if.hpp>
#include <hpx/components/dataflow/is_dataflow.hpp>
#include <hpx/components/dataflow/server/detail/dataflow_slot.hpp>
#include <hpx/components/dataflow/server/detail/apply_helper.hpp>
#include <hpx/components/dataflow/server/detail/dataflow_impl_helpers.hpp>
#include <hpx/components/dataflow/server/detail/component_wrapper.hpp>

#include <hpx/util/decay.hpp>
#include <hpx/util/demangle_helper.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/detail/full_empty_memory.hpp>

namespace hpx { namespace lcos { namespace server { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // data structure holding all counters for full_empty entries
    struct dataflow_counter_data
    {
        dataflow_counter_data()
          : constructed_(0), initialized_(0), fired_(0), destructed_(0)
        {}

        boost::int64_t constructed_;
        boost::int64_t initialized_;
        boost::int64_t fired_;
        boost::int64_t destructed_;
        lcos::local::spinlock mtx_;
    };

    /// counter function declarations

    HPX_COMPONENT_EXPORT boost::int64_t get_initialized_count(bool);
    HPX_COMPONENT_EXPORT boost::int64_t get_constructed_count(bool);
    HPX_COMPONENT_EXPORT boost::int64_t get_fired_count(bool);
    HPX_COMPONENT_EXPORT boost::int64_t get_destructed_count(bool);
    HPX_COMPONENT_EXPORT void update_constructed_count();
    HPX_COMPONENT_EXPORT void update_initialized_count();
    HPX_COMPONENT_EXPORT void update_fired_count();
    HPX_COMPONENT_EXPORT void update_destructed_count();

    // call this to register all counter types for dataflow objects
    void register_counter_types();

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Action
      , typename R, typename ...Ts
    >
    struct dataflow_impl;

    template <
        typename Action
      , typename R
    >
    struct dataflow_impl<
        Action
      , R()
    >
        : ::hpx::lcos::base_lco_with_value<
              typename Action::result_type
            , typename Action::remote_result_type
          >
    {
        typedef
            typename traits::promise_remote_result<R>::type
            result_type;

        typedef util::detail::value_or_error<result_type> data_type;

        typedef
            hpx::lcos::base_lco_with_value<
                typename Action::result_type
              , typename Action::remote_result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
              , typename traits::promise_local_result<
                    typename Action::result_type>::type()
            >
            wrapped_type;

        typedef
            components::managed_component<wrapped_type>
            wrapping_type;

        typedef
            passed_args_transforms<
                util::tuple<>
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
                    HPX_ASSERT(d.stores_value()); // This should never be empty

                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], std::move(r));
                }
            }
        }

        ~dataflow_impl()
        {
            HPX_ASSERT(!result.is_empty());
            HPX_ASSERT(targets.empty());
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }

        typedef typename Action::remote_result_type remote_result;

        // This is called by our action after it executed. The argument is what
        // has been calculated by the action. The result has to be sent to all
        // connected dataflow instances.
        void set_value(remote_result && r)
        {
            remote_result tmp(r);
            result.set(std::move(r));
            forward_results(tmp);
        }

        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }

            // Note: lco::set_value is a direct action, for this reason,
            //       the following loop will not be parallelized if the
            //       targets are local (which is ok)
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp =  r;
                hpx::apply<action_type>(t[i], std::move(tmp));
            }
        }

        // This is called when some dataflow object connects to this one (i.e.
        // requests to receive the output of this dataflow instance).
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
                    HPX_ASSERT(d.stores_value()); // This should never be empty

                    typedef typename lco_type::set_value_action action_type;
                    result_type r =  d.get_value();
                    hpx::apply<action_type>(target, std::move(r));
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

        result_type get_value(error_code& ec = throws)
        {
            HPX_ASSERT(false);
            static result_type default_;
            return default_;
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
            HPX_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }

    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;

        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            HPX_ASSERT(0 == back_ptr_);
            HPX_ASSERT(bp);
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
      , typename R
      , typename ...Ts
    >
    struct dataflow_impl<
        Action
      , R(Ts...)
    >
        : ::hpx::lcos::base_lco_with_value<
              typename Action::result_type
            , typename Action::remote_result_type
          >
    {
        typedef
            typename traits::promise_remote_result<R>::type
            result_type;

        typedef util::detail::value_or_error<result_type> data_type;

        typedef
            hpx::lcos::base_lco_with_value<
                typename Action::result_type
              , typename Action::remote_result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
              , typename traits::promise_local_result<
                    typename Action::result_type>::type(Ts...)
            >
            wrapped_type;

        typedef
            components::managed_component<wrapped_type>
            wrapping_type;

        typedef
            passed_args_transforms<
                util::tuple<Ts...>
            >
            passed_args;

        typedef typename passed_args::results_type args_type;

        typedef typename passed_args::slot_to_args_map slot_to_args_map;

        // generate the bitset for checking if all slots have fired
        static const boost::uint32_t slots_completed = (1u << sizeof...(Ts)) - 1;

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

        template <std::size_t ...Is>
        void init_impl(util::detail::pack_c<std::size_t, Is...>, Ts&&... vs)
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
            future_slots.reserve(sizeof...(Is));

            int const sequencer[] = {
                (set_slot<Is>(
                    vs
                  , typename hpx::traits::is_dataflow<Ts>::type()), 0)...
            };
        }

        void init(Ts&&... vs)
        {
            init_impl(
                typename util::detail::make_index_pack<sizeof...(Ts)>::type()
              , std::forward<Ts>(vs)...);
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
                    HPX_ASSERT(d.stores_value()); // This should never be empty

                    typedef typename lco_type::set_value_action action_type;
                    result_type r = d.get_value();
                    hpx::apply<action_type>(t[i], std::move(r));
                }
            }
        }

        ~dataflow_impl()
        {
            HPX_ASSERT(!result.is_empty());
            HPX_ASSERT(targets.empty());
            HPX_ASSERT(slots_set == slots_completed);
            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }

        typedef typename Action::remote_result_type remote_result;

        // This is called by our action after it executed. The argument is what
        // has been calculated by the action. The result has to be sent to all
        // connected dataflow instances.
        void set_value(remote_result && r)
        {
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            future_slots.clear();
            remote_result tmp(r);
            result.set(std::move(r));
            forward_results(tmp);
        }

        void forward_results(remote_result & r)
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);
            }

            // Note: lco::set_value is a direct action, for this reason,
            //       the following loop will not be parallelized if the
            //       targets are local (which is ok)
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_value_action action_type;
                result_type tmp =  r;
                hpx::apply<action_type>(t[i], std::move(tmp));
            }
        }

        // This is called when some dataflow object connects to this one (i.e.
        // requests to receive the output of this dataflow instance).
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
                    HPX_ASSERT(d.stores_value()); // This should never be empty

                    typedef typename lco_type::set_value_action action_type;
                    result_type r =  d.get_value();
                    hpx::apply<action_type>(target, std::move(r));
                }
            }
            else
            {
                targets.push_back(target);
            }
        }

        // Setting the slot for future values
        template <std::size_t Slot, typename A>
        void set_slot(A && a, boost::mpl::true_)
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

            component_type * c = new component_type(this, std::forward<A>(a));
            future_slots.push_back(c);
            (*c)->connect_();
        };

        // Setting the slot for immediate values
        template <
            std::size_t Slot
          , typename A
        >
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(A && a, boost::mpl::false_)
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(slots) = std::forward<A>(a);
            maybe_apply<Slot>();
        };
        // Setting the slot for immediate values
        template <
            std::size_t Slot
          , typename A
        >
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_slot(A &&, boost::mpl::false_)
        {
            maybe_apply<Slot>();
        };

        template <std::size_t Slot>
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

        result_type get_value(error_code& ec = throws)
        {
            HPX_ASSERT(false);
            static result_type default_;
            return default_;
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
            HPX_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }

    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;

        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            HPX_ASSERT(0 == back_ptr_);
            HPX_ASSERT(bp);
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
}}}}

namespace hpx { namespace traits
{
    template <
        typename Action
      , typename R
      , typename ...Ts
    >
    struct component_type_database<
        lcos::server::detail::dataflow_impl<
            Action
          , R(Ts...)
        >
    >
    {
        typedef
            typename boost::mpl::if_<
                boost::is_void<R>
              , hpx::util::unused_type
              , R
            >::type
            result_type;

        static components::component_type get()
        {
            return component_type_database<
                lcos::base_lco_with_value<
                    typename Action::result_type
                  , typename Action::remote_result_type
                >
            >::get();
        }

        static void set(components::component_type t)
        {
            component_type_database<
                lcos::base_lco_with_value<
                    typename Action::result_type
                  , typename Action::remote_result_type
                >
            >::set(t);
        }
    };
}}

#endif
