//  Copyright (c) 2011 Thomas Heller
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
          : constructed_(0), initialized_(0), fired_(0)
        {}

        boost::int64_t constructed_;
        boost::int64_t initialized_;
        boost::int64_t fired_;
        lcos::local::spinlock mtx_;
    };
    extern HPX_COMPONENT_EXPORT dataflow_counter_data dataflow_counter_data_;

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

#define HPX_RV_REF_ARGS(z, n, _)                                              \
        BOOST_PP_COMMA_IF(n)                                                  \
            BOOST_RV_REF(BOOST_PP_CAT(A, n)) BOOST_PP_CAT(a, n)              \
    /**/

#define HPX_FORWARD_ARGS(z, n, _)                                             \
        BOOST_PP_COMMA_IF(n)                                                  \
            boost::forward<BOOST_PP_CAT(A, n)>(BOOST_PP_CAT(a, n))            \
    /**/

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

#undef HPX_FWD_ARGS
#undef HPX_FORWARD_ARGS

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
            args_completed = (BOOST_PP_REPEAT(N, HPX_LCOS_DATAFLOW_M0, _) 0);
#undef HPX_LCOS_DATAFLOW_M0
#endif

        dataflow_impl(
            naming::id_type const & id
          , lcos::local::spinlock & mtx
          , std::vector<naming::id_type> & t
        )
            : back_ptr_(0)
#if N > 0
            , args_set(0)
#endif
            , result_set(false)
            , targets(t)
            , action_id(id)
            , mtx(mtx)
        {
        }

#if N > 0
        void init(BOOST_PP_REPEAT(N, HPX_RV_REF_ARGS, _))
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
            typedef                                                             \
                component_wrapper<                                              \
                    dataflow_slot<                                              \
                        BOOST_PP_CAT(A, N)                                      \
                      , N, wrapped_type                                         \
                    >                                                           \
                >                                                               \
                BOOST_PP_CAT(component_type, N);                                \
                                                                                \
            BOOST_PP_CAT(component_type, N) * BOOST_PP_CAT(w, N)                \
                = new BOOST_PP_CAT(component_type, N)(                          \
                    this                                                        \
                  , boost::move(BOOST_PP_CAT(a, N))                             \
                );                                                              \
                                                                                \
            arg_ids[N] = BOOST_PP_CAT(w, N);                                    \
            (*BOOST_PP_CAT(w, N))->connect();                                   \
    /**/

            BOOST_PP_REPEAT(N, HPX_LCOS_DATAFLOW_M0, _)
#undef HPX_LCOS_DATAFLOW_M0
        }
#endif

        void init(BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a))
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
#if N == 0
            hpx::applier::apply_c<Action>(get_gid(), action_id);
#endif
#if N > 0
#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
            typedef                                                             \
                component_wrapper<                                              \
                    dataflow_slot<                                              \
                        BOOST_PP_CAT(A, N)                                      \
                      , N, wrapped_type                                         \
                    >                                                           \
                >                                                               \
                BOOST_PP_CAT(component_type, N);                                \
                                                                                \
            BOOST_PP_CAT(component_type, N) * BOOST_PP_CAT(w, N)                \
                = new BOOST_PP_CAT(component_type, N)(                          \
                    this                                                        \
                  , BOOST_PP_CAT(a, N)                                          \
                );                                                              \
                                                                                \
            arg_ids[N] = BOOST_PP_CAT(w, N);                                    \
            (*BOOST_PP_CAT(w, N))->connect();                                   \
    /**/

            BOOST_PP_REPEAT(N, HPX_LCOS_DATAFLOW_M0, _)
#undef HPX_LCOS_DATAFLOW_M0
#endif
        }

        ~dataflow_impl()
        {
            forward_results();
            BOOST_ASSERT(result_set);
            BOOST_ASSERT(targets.size() == 0);
#if N > 0
            BOOST_ASSERT(args_set == args_completed);
#endif
#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
            delete arg_ids[N];                                                  \
    /**/
            //BOOST_PP_REPEAT(N, HPX_LCOS_DATAFLOW_M0, _)
#undef HPX_LCOS_DATAFLOW_M0

            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }

        /*
        void finalize()
        {
            int time = 0;
#if N > 0
            bool wait = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                wait = (args_set == args_completed);
            }
            if(wait)
#endif
            {
                while(true)
                {
                    {
                        lcos::local::spinlock::scoped_lock l(mtx);
                        if(result_set)
                            break;
                    }
                    threads::suspend(boost::posix_time::microseconds(++time * 10));
                }
            }
        }
        */

        typedef typename Action::result_type remote_result;

        void set_result(BOOST_RV_REF(remote_result) r)
        {
            BOOST_ASSERT(!result_set);
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_result(): set_result: "
                << targets.size()
                ;
            {
                lcos::local::spinlock::scoped_lock l(mtx);

                result = r;
                result_set = true;
            }

            forward_results();
        }

        void forward_results()
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(result_set == false) return;
                std::swap(targets, t);
            }

            // Note: lco::set_result is a direct action, for this reason,
            //       the following loop will not be parallelized if the
            //       targets are local (which is ok)
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_result_action action_type;
                applier::apply<action_type>(t[i], boost::forward<remote_result>(result));
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

            if(result_set)
            {
                typedef typename lco_type::set_result_action action_type;
                l.unlock();
                applier::apply<action_type>(target, boost::forward<remote_result>(result));
            }
            else
            {
                targets.push_back(target);
            }
        }

#if N > 0
        template <int Slot, typename T>
        typename boost::enable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_arg(
            BOOST_FWD_REF(T) value
        )
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(args) = boost::forward<T>(value);
            maybe_apply<Slot>();
        }

        template <int Slot>
        typename boost::disable_if<
            typename boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >::type
        >::type
        set_arg(hpx::util::unused_type)
        {
            maybe_apply<Slot>();
        }

        template <int Slot>
        void maybe_apply()
        {
            bool apply_it = false;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(args_set != args_completed)
                {
                    args_set |= (1<<Slot);
                    apply_it = (args_set == args_completed);
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
                  , args
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
                << args_set
                << "("
                << args_completed
                << ")"
                << "\n"
                ;
        }
#endif

        void set_event()
        {
            this->set_result_nonvirt(remote_result());
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
        args_type args;
        boost::uint32_t args_set;
#endif

        remote_result result;
        bool result_set;
        std::vector<naming::id_type> & targets;
        naming::id_type action_id;

#if N > 0
        boost::array<component_wrapper_base *, N> arg_ids;
#endif
        lcos::local::spinlock & mtx;
    };

#undef REMAINDER
#undef N

#endif
