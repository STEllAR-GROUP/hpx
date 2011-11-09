
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
#include <examples/bright_future/dataflow/is_dataflow.hpp>
#include <examples/bright_future/dataflow/server/detail/dataflow_slot.hpp>
#include <examples/bright_future/dataflow/server/detail/apply_helper.hpp>
#include <examples/bright_future/dataflow/server/detail/dataflow_impl_helpers.hpp>
#include <examples/bright_future/dataflow/server/detail/component_wrapper.hpp>

#include <hpx/util/demangle_helper.hpp>

namespace hpx { namespace lcos { namespace server { namespace detail {
    template <
        typename Action
      , BOOST_PP_ENUM_PARAMS_WITH_A_DEFAULT(HPX_ACTION_ARGUMENT_LIMIT, typename A, void)
      , typename Result = void
      , typename Enable = void
    >
    struct dataflow_impl;

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            0                                                                   \
          , HPX_ACTION_ARGUMENT_LIMIT                                           \
          , <examples/bright_future/dataflow/server/detail/dataflow_impl.hpp> \
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
            typename boost::mpl::if_<
                boost::is_void<Result>
              , hpx::util::unused_type
              , Result
            >::type
          , typename Action::result_type
        >
    {
        typedef
            typename boost::mpl::if_<
                boost::is_void<Result>
              , hpx::util::unused_type
              , Result
            >::type
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

        // generate the bitset for checking if all slots have fired
#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
            (1<<N) |                                                            \
    /**/
        static const boost::uint32_t
            args_completed = (BOOST_PP_REPEAT(N, HPX_LCOS_DATAFLOW_M0, _) 0);
#undef HPX_LCOS_DATAFLOW_M0

        dataflow_impl(
            naming::id_type const & id
        )
            : back_ptr_(0)
            , args_set(0)
            , result_set(false)
            , action_id(id)
        {
        }

        void init(BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a))
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::init(): "
                << get_gid()
                ;
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
            (*BOOST_PP_CAT(w, N))->connect();                                   \
            arg_ids[N] = BOOST_PP_CAT(w, N);                                    \
    /**/

            BOOST_PP_REPEAT(N, HPX_LCOS_DATAFLOW_M0, _)

#undef HPX_LCOS_DATAFLOW_M0
#else
            hpx::applier::apply_c<Action>(get_gid(), action_id);
#endif
        }

        ~dataflow_impl()
        {
            BOOST_ASSERT(args_set == args_completed);
#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
            delete arg_ids[N];                                                  \
    /**/
            BOOST_PP_REPEAT(N, HPX_LCOS_DATAFLOW_M0, _)
#undef HPX_LCOS_DATAFLOW_M0

            LLCO_(info)
                << "~dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::dataflow_impl(): "
                << get_gid()
                ;
        }

        typedef typename Action::result_type remote_result;

        void set_result(remote_result const & r)
        {
            BOOST_ASSERT(!result_set);
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_result(): set_result: "
                << targets.size()
                ;

            std::vector<promise<void> > lazy_results;
            std::vector<naming::id_type> t;

            {
                typename hpx::util::spinlock::scoped_lock l(mtx);
                std::swap(targets, t);

                result = r;
                result_set = true;
            }

            // Note: lco::set_result is a direct action, for this reason,
            //       the following loop will not be parallelized if the
            //       targets are local (which is ok)
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename lco_type::set_result_action action_type;
                lazy_results.push_back(async<action_type>(t[i], result));
            }
            wait(lazy_results);
        }

        void connect(naming::id_type const & target)
        {
            LLCO_(info)
                << "dataflow_impl<"
                << hpx::actions::detail::get_action_name<Action>()
                << ">::set_target() of "
                << get_gid()
                << " ";

            typename hpx::util::spinlock::scoped_lock l(mtx);
            if(result_set)
            {
                typedef typename lco_type::set_result_action action_type;
                promise<void> p = async<action_type>(target, result);
                l.unlock();

                wait(p);
            }
            else
            {
                targets.push_back(target);
            }
        }

#if N > 0
        template <int Slot>
        typename boost::enable_if<
            boost::mpl::has_key<slot_to_args_map, boost::mpl::int_<Slot> >
        >::type
        set_arg(
            typename boost::fusion::result_of::value_at<
                args_type
              , typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >::type const & value
        )
        {
            boost::fusion::at<
                typename boost::mpl::at<
                    slot_to_args_map
                  , boost::mpl::int_<Slot>
                >::type
            >(args) = value;
            maybe_apply<Slot>();
        }

        template <int Slot>
        void set_arg(hpx::util::unused_type)
        {
            maybe_apply<Slot>();
        }

        template <int Slot>
        void maybe_apply()
        {
            typename hpx::util::spinlock::scoped_lock l(mtx);
            args_set |= (1<<Slot);
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
            if(args_set == args_completed)
            {
                apply_helper<
                    boost::fusion::result_of::size<args_type>::value
                  , Action
                >()(
                    get_gid()
                  , action_id
                  , args
                );
            }
            arg_ids[Slot] = 0;
        }
#endif


        void set_event()
        {
            //if(boost::is_void<Result>::value)
            {
                this->set_result_nonvirt(remote_result());
            }
            //else
            {
                //BOOST_ASSERT(false);
            }
        }

        result_type get_value()
        {
            BOOST_ASSERT(false);
            return result_type();
        }

        naming::id_type get_gid() const
        {
            return naming::id_type(get_base_gid(), naming::id_type::unmanaged);
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

        args_type args;
        boost::uint32_t args_set;

        hpx::util::spinlock mtx;
        remote_result result;
        bool result_set;
        std::vector<naming::id_type> targets;
        naming::id_type action_id;

#if N > 0
        boost::array<component_wrapper_base *, N> arg_ids;
#endif
    };

#undef N

#endif
