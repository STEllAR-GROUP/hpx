//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_SERVER_DETAIL_DATAFLOW_ARG_HPP
#define HPX_LCOS_DATAFLOW_SERVER_DETAIL_DATAFLOW_ARG_HPP

#include <hpx/components/dataflow/dataflow_base_fwd.hpp>
#include <hpx/components/dataflow/dataflow_fwd.hpp>

namespace hpx { namespace lcos { namespace server { namespace detail
{
    template <typename T, int Slot, typename SinkAction, typename Enable = void>
    struct dataflow_slot
        : base_lco_with_value<T, T>
    {
        typedef hpx::lcos::server::detail::dataflow_slot<T, Slot, SinkAction> wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;

        typedef T result_type;
        typedef T remote_result;

        template <typename TT>
        dataflow_slot(SinkAction * back, BOOST_FWD_REF(TT) t)
            : back_ptr_(0)
            , back(back)
            , t(boost::forward<TT>(t))
        {}

        void set_result(BOOST_RV_REF(remote_result) r)
        {
            BOOST_ASSERT(false);
        }

        void connect()
        {
            back->template set_arg<Slot>(boost::move(t));
        }

        void set_event()
        {
        }

        T get_value()
        {
            BOOST_ASSERT(false);
            return T();
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

        void set_back_ptr(components::managed_component<dataflow_slot>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }

        components::managed_component<dataflow_slot>* back_ptr_;
        SinkAction *back;
        T t;
    };

    template <typename T, int Slot, typename SinkAction>
    struct dataflow_slot<T, Slot, SinkAction, typename boost::enable_if<hpx::traits::is_dataflow<T> >::type>
        : hpx::lcos::base_lco_with_value<
            typename boost::mpl::if_<
                boost::is_void<typename T::result_type>
              , hpx::util::unused_type
              , typename T::result_type
            >::type
          , typename T::remote_result_type
        >
    {
        typedef
            typename boost::mpl::if_<
                boost::is_void<typename T::result_type>
              , hpx::util::unused_type
              , typename T::result_type
            >::type
            result_type;

        typedef T dataflow_type;
        typedef
            hpx::lcos::server::detail::dataflow_slot<
                dataflow_type
              , Slot
              , SinkAction
            >
            wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;

        typedef typename T::remote_result_type remote_result;

        dataflow_slot(SinkAction * back, dataflow_type const & flow)
            : back_ptr_(0)
            , dataflow_sink(back)
            , dataflow_source(flow)
        {
            BOOST_ASSERT(dataflow_source.get_gid());
        }

        ~dataflow_slot()
        {
            LLCO_(info)
                << "~dataflow_slot<"
                << util::type_id<T>::typeid_.type_id()
                << ", " << Slot
                << hpx::actions::detail::get_action_name<SinkAction>()
                << ">::dataflow_slot(): "
                << get_gid();
        }

        void set_result(BOOST_RV_REF(remote_result) r)
        {
            LLCO_(info)
                << "dataflow_slot<"
                << util::type_id<T>::typeid_.type_id()
                << ", " << Slot
                << hpx::actions::detail::get_action_name<SinkAction>()
                << ">::set_result(): "
                << get_gid();
            dataflow_sink
                ->template set_arg<Slot>(
                    traits::get_remote_result<result_type, remote_result>
                        ::call(r)
                );
            //dataflow_source.invalidate();
        }

        void connect()
        {
            /*
            LLCO_(info)
                << "dataflow_slot<"
                << util::type_id<T>::typeid_.type_id()
                << ", " << Slot
                << hpx::actions::detail::get_action_name<SinkAction>()
                << ">::connect() from "
                << get_gid();

            typedef
                typename dataflow_type::stub_type::server_type::connect_action
                action_type;
            
            BOOST_ASSERT(dataflow_source.get_gid());

            applier::apply<action_type>(dataflow_source.get_gid(), get_gid());
            */
        }

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

        void set_back_ptr(components::managed_component<dataflow_slot>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }

        components::managed_component<dataflow_slot>* back_ptr_;

        SinkAction * dataflow_sink;
        dataflow_type dataflow_source;
    };

    template <typename T, typename SinkAction>
    struct dataflow_slot<T, -1, SinkAction, typename boost::enable_if<hpx::traits::is_dataflow<T> >::type>
        : hpx::lcos::base_lco_with_value<
            typename boost::mpl::if_<
                boost::is_void<typename T::result_type>
              , hpx::util::unused_type
              , typename T::result_type
            >::type
          , typename T::remote_result_type
        >
    {
        typedef
            typename boost::mpl::if_<
                boost::is_void<typename T::result_type>
              , hpx::util::unused_type
              , typename T::result_type
            >::type
            result_type;

        typedef T dataflow_type;
        typedef
            hpx::lcos::server::detail::dataflow_slot<
                dataflow_type
              , -1
              , SinkAction
            >
            wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;

        typedef typename T::remote_result_type remote_result;

        dataflow_slot(SinkAction * back, dataflow_type const & flow, unsigned slot)
            : back_ptr_(0)
            , dataflow_sink(back)
            , dataflow_source(flow)
            , slot(slot)
        {
            BOOST_ASSERT(dataflow_source.get_gid());
        }

        ~dataflow_slot()
        {
            LLCO_(info)
                << "~dataflow_slot<"
                << util::type_id<T>::typeid_.type_id()
                << ", " << slot << ", "
                << hpx::actions::detail::get_action_name<SinkAction>()
                << ">::dataflow_slot() dynamic: "
                << get_gid();
        }

        void set_result(BOOST_RV_REF(remote_result) r)
        {
            LLCO_(info)
                << "dataflow_slot<"
                << util::type_id<T>::typeid_.type_id()
                << ", " << slot << ", "
                << hpx::actions::detail::get_action_name<SinkAction>()
                << ">::set_result() dynamic: "
                << get_gid();
            dataflow_sink->set_slot(slot);
            //dataflow_source.invalidate();
        }

        void connect()
        {
            LLCO_(info)
                << "dataflow_slot<"
                << util::type_id<T>::typeid_.type_id()
                << ", " << slot << ", "
                << hpx::actions::detail::get_action_name<SinkAction>()
                << ">::connect() dynamic: "
                << get_gid();
            /*
            typedef
                typename dataflow_type::stub_type::server_type::connect_action
                action_type;

            BOOST_ASSERT(dataflow_source.get_gid());

            applier::apply<action_type>(dataflow_source.get_gid(), get_gid());
            */
        }

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

        void set_back_ptr(components::managed_component<dataflow_slot>* bp)
        {
            BOOST_ASSERT(0 == back_ptr_);
            BOOST_ASSERT(bp);
            back_ptr_ = bp;
        }

        components::managed_component<dataflow_slot>* back_ptr_;

        SinkAction * dataflow_sink;
        dataflow_type dataflow_source;
        unsigned slot;
    };
}}}}

namespace hpx { namespace traits
{
    template <
        typename T
      , int Slot
      , typename SinkAction
    >
    struct component_type_database<lcos::server::detail::dataflow_slot<T, Slot, SinkAction> >
    {
        typedef lcos::server::detail::dataflow_slot<T, Slot, SinkAction> dataflow_slot;
        typedef typename dataflow_slot::result_type result_type;
        typedef typename dataflow_slot::remote_result remote_result_type;

        static components::component_type get()
        {
            return
                component_type_database<
                    lcos::base_lco_with_value<result_type, remote_result_type>
                >::get();
        }

        static void set(components::component_type t)
        {
            component_type_database<
                lcos::base_lco_with_value<result_type, remote_result_type>
            >::set(t);
        }
    };
}}
#endif
