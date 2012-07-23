//  Copyright (c) 2011-2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

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
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/utility/enable_if.hpp>
#include <hpx/components/dataflow/is_dataflow.hpp>
#include <hpx/components/dataflow/server/detail/dataflow_slot.hpp>
#include <hpx/components/dataflow/server/detail/apply_helper.hpp>
#include <hpx/components/dataflow/server/detail/dataflow_impl_helpers.hpp>
#include <hpx/components/dataflow/server/detail/component_wrapper.hpp>

#include <hpx/util/demangle_helper.hpp>
#include <hpx/lcos/local/spinlock.hpp>

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
    HPX_COMPONENT_EXPORT extern dataflow_counter_data dataflow_counter_data_;

    // call this to register all counter types for dataflow objects
    void register_counter_types();

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Action
      , BOOST_PP_ENUM_PARAMS_WITH_A_DEFAULT(HPX_ACTION_ARGUMENT_LIMIT, typename A, void)
      , typename Result = typename traits::promise_local_result<
                typename Action::result_type>::type
      , typename Enable = void
    >
    struct dataflow_impl;

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            0                                                                   \
          , HPX_ACTION_ARGUMENT_LIMIT                                           \
          , <hpx/components/dataflow/server/detail/dataflow_impl.hpp>           \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

}}}}

namespace hpx { namespace traits
{
    template <
        typename Action
      , BOOST_PP_ENUM_PARAMS(HPX_ACTION_ARGUMENT_LIMIT, typename A)
      , typename Result
    >
    struct component_type_database<
        lcos::server::detail::dataflow_impl<
            Action
          , BOOST_PP_ENUM_PARAMS(HPX_ACTION_ARGUMENT_LIMIT, A), Result
        >
    >
    {
        typedef
            typename boost::mpl::if_<
                boost::is_void<Result>
              , hpx::util::unused_type
              , Result
            >::type
            result_type;

        static components::component_type get()
        {
            return component_type_database<
                lcos::base_lco_with_value<
                    result_type
                  , typename Action::result_type
                >
            >::get();
        }

        static void set(components::component_type t)
        {
            component_type_database<
                lcos::base_lco_with_value<
                    result_type
                  , typename Action::result_type
                >
            >::set(t);
        }
    };
}}


#endif

#else

#define N BOOST_PP_ITERATION()
#define REMAINDER BOOST_PP_SUB(HPX_ACTION_ARGUMENT_LIMIT, N)

    template <
        typename Action
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
      , typename Result
    >
    struct dataflow_impl<
        Action
        BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, A)
        BOOST_PP_COMMA_IF(REMAINDER)
        BOOST_PP_ENUM_PARAMS(
            REMAINDER
          , void BOOST_PP_INTERCEPT
        )
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

        typedef util::value_or_error<result_type> data_type;

        typedef
            hpx::lcos::base_lco_with_value<
                result_type
              , typename Action::result_type
            >
            lco_type;
        typedef
            dataflow_impl<
                Action
                BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, A)
            >
            wrapped_type;

        typedef
            components::managed_component<
                dataflow_impl<
                    Action
                    BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, A)
                >
            >
            wrapping_type;

        typedef
            passed_args_transforms<
                BOOST_PP_CAT(boost::mpl::vector, N)<BOOST_PP_ENUM_PARAMS(N, A)>
            >
            passed_args;

        typedef typename passed_args::results_type args_type;

        typedef typename passed_args::slot_to_args_map slot_to_args_map;

#if N > 0
        // generate the bitset for checking if all slots have fired
#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
            (1<<N) |                                                            \
    /**/
        static const boost::uint32_t
            slots_completed = (BOOST_PP_REPEAT(N, HPX_LCOS_DATAFLOW_M0, _) 0);
#undef HPX_LCOS_DATAFLOW_M0
#endif

        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
#if N > 0
            , slots_set(0)
#endif
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }


        void init(BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a))
        {
            LLCO_(info)
                << "hpx::lcos::server::detail::dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
#if N == 0
            hpx::apply_c<Action>(get_gid(), action_id);
#endif
#if N > 0
            future_slots.reserve(N);
#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
            set_slot<N>(                                                        \
                BOOST_PP_CAT(a, N)                                              \
              , typename hpx::traits::is_dataflow<BOOST_PP_CAT(A, N)>::type()); \
    /**/

            BOOST_PP_REPEAT(N, HPX_LCOS_DATAFLOW_M0, _)
#undef HPX_LCOS_DATAFLOW_M0
#endif
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
                if(!d.stores_value())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(t[i], d.get_error());
                }
                else
                {
                    typedef typename lco_type::set_value_action action_type;
                    result_type r =  d.get_value();
                    hpx::apply<action_type>(t[i], boost::move(r));
                }
            }
        }

        ~dataflow_impl()
        {
            BOOST_ASSERT(!result.is_empty());
            BOOST_ASSERT(targets.empty());
#if N > 0
            BOOST_ASSERT(slots_set == slots_completed);
#endif
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
#if N > 0
            /*
            BOOST_FOREACH(detail::component_wrapper_base *p, future_slots)
            {
                delete p;
            }
            */
#endif
            remote_result tmp(r);
            forward_results(tmp);
            result.set(boost::move(r));
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

            if(!result.is_empty())
            {
                data_type d;
                result.read(d);

                if(!d.stores_value())
                {
                    typedef typename lco_type::set_exception_action action_type;
                    hpx::apply<action_type>(target, d.get_error());
                }
                else
                {
                    typedef typename lco_type::set_value_action action_type;
                    result_type r =  d.get_value();
                    hpx::apply<action_type>(target, boost::move(r));
                }
            }
            else
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                targets.push_back(target);
            }
        }

#if N > 0
        // Setting the slot for future values
        template <int Slot, typename A>
        void set_slot(BOOST_FWD_REF(A) a, boost::mpl::true_)
        {
            typedef
                typename boost::remove_const<
                    typename util::detail::remove_reference<
                        A
                    >::type
                >::type
                dataflow_type;

            typedef
                dataflow_slot<dataflow_type, Slot, dataflow_impl>
                dataflow_slot_type;

            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;

            component_type * c = new component_type(this, boost::forward<A>(a));
            (*c)->connect_();
            future_slots.push_back(c);
        };

        // Setting the slot for immediate values
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
        // Setting the slot for immediate values
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

                lcos::local::spinlock::scoped_lock l(mtx);
                ++dataflow_counter_data_.fired_;
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
#endif

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
                    naming::strip_credit_from_cgid(get_base_gid())
                  , naming::id_type::unmanaged
                );
        }

        naming::gid_type get_base_gid() const
        {
            BOOST_ASSERT(back_ptr_);
            return back_ptr_->get_base_gid();
        }

    private:
        template <typename, typename>
        friend class components::managed_component;

        void set_back_ptr(components::managed_component<dataflow_impl>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }

        components::managed_component<dataflow_impl>* back_ptr_;

#if N > 0
        args_type slots;
        boost::uint32_t slots_set;
        std::vector<detail::component_wrapper_base *> future_slots;
#endif

        util::full_empty<data_type> result;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;

        lcos::local::spinlock & mtx;
    };

#undef REMAINDER
#undef N

#endif
