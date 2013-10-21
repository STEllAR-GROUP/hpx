//  Copyright (c) 2011-2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_SERVER_DETAIL_DATAFLOW_SLOT_HPP
#define HPX_LCOS_DATAFLOW_SERVER_DETAIL_DATAFLOW_SLOT_HPP

#include <hpx/components/dataflow/dataflow_base_fwd.hpp>
#include <hpx/components/dataflow/dataflow_fwd.hpp>

namespace hpx { namespace lcos { namespace server { namespace detail
{
    template <typename T, int Slot, typename SinkAction>
    struct dataflow_slot
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

        void set_value(BOOST_RV_REF(remote_result) r)
        {
            LLCO_(info)
                << "dataflow_slot<"
                << util::type_id<T>::typeid_.type_id()
                << ", " << Slot
                << hpx::actions::detail::get_action_name<SinkAction>()
                << ">::set_value(): "
                << get_gid();
            dataflow_sink
                ->template set_slot<Slot>(
                    traits::get_remote_result<result_type, remote_result>
                        ::call(r)
                  , boost::mpl::false_()
                );
        }

        void connect_()
        {
            LLCO_(info)
                << "dataflow_slot<"
                << util::type_id<T>::typeid_.type_id()
                << ", " << Slot
                << hpx::actions::detail::get_action_name<SinkAction>()
                << ">::connect() from "
                << get_gid();
            
            BOOST_ASSERT(get_gid());

            if(dataflow_source.valid())
            {
                dataflow_source.connect(get_gid());
            }
            else
            {
                //this->set_value_nonvirt(remote_result());
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
