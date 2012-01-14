//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_BASE_VOID_HPP
#define HPX_LCOS_DATAFLOW_BASE_VOID_HPP

#include <hpx/components/dataflow/dataflow_base_fwd.hpp>
#include <hpx/components/dataflow/dataflow_base_impl.hpp>
#include <hpx/components/dataflow/stubs/dataflow.hpp>

namespace hpx { namespace lcos
{
    template <>
    struct dataflow_base<void>
    {
        typedef traits::promise_remote_result<void>::type remote_result_type;
        typedef void result_type;
        typedef
            components::client_base<
                dataflow_base<void>
              , stubs::dataflow
            >
            base_type;

        typedef stubs::dataflow stub_type;

        dataflow_base()
        {}

        virtual ~dataflow_base()
        {
        }

        dataflow_base(promise<naming::id_type, naming::gid_type> const & promise)
            : impl(new detail::dataflow_base_impl(promise))
        {}

#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
        template <BOOST_PP_ENUM_PARAMS(N, typename A)>                          \
        dataflow_base(                                                          \
            promise<naming::id_type, naming::gid_type> const & promise          \
          , BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)                        \
        )                                                                       \
            : impl(                                                             \
                new detail::dataflow_base_impl(                                 \
                    promise                                                     \
                  , BOOST_PP_ENUM_PARAMS(N, a)                                  \
                )                                                               \
            )                                                                   \
        {                                                                       \
        }                                                                       \
    /**/
        BOOST_PP_REPEAT_FROM_TO(
            1
          , BOOST_PP_SUB(HPX_ACTION_ARGUMENT_LIMIT, 3)
          , HPX_LCOS_DATAFLOW_M0
          , _
        )

#undef HPX_LCOS_DATAFLOW_M0

        promise<void> get_promise() const
        {
            promise<void> p;
            connect(p.get_gid());
            return p;
        }

        void get() const
        {
            promise<void> p;
            connect(p.get_gid());
            p.get();
        }

        void invalidate()
        {
            impl->invalidate();
        }

        naming::id_type get_gid() const
        {
            return impl->get_gid();
        }

        void connect(naming::id_type const & target) const
        {
            stub_type::connect(impl->get_gid(), target);
        }

        boost::shared_ptr<detail::dataflow_base_impl> impl;

    private:
        friend class boost::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & impl;
        }
    };
}}

#endif
