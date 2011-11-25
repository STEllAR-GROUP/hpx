
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXAMPLES_BRIGHT_FUTURE_DATAFLOW_BASE_IMPL_HPP
#define EXAMPLES_BRIGHT_FUTURE_DATAFLOW_BASE_IMPL_HPP

#include <hpx/hpx_fwd.hpp>
#include <examples/bright_future/dataflow/is_dataflow.hpp>

namespace hpx { namespace lcos { namespace detail {
    struct dataflow_base_impl
    {
        dataflow_base_impl()
        {}

        ~dataflow_base_impl()
        {}
        
        dataflow_base_impl(
            lcos::promise<naming::id_type, naming::gid_type> const & promise
        )
            : gid_promise(promise)
        {}

        template <typename T>
        void keep_arg_alive(T const & t, boost::mpl::false_)
        {}

        template <typename T>
        void keep_arg_alive(std::vector<T> const & t, boost::mpl::false_)
        {
            BOOST_FOREACH(T const & tt, t)
            {
                keep_arg_alive(
                    tt
                  , typename traits::is_dataflow<T>::type()
                );
            }
        }
        template <typename T>
        void keep_arg_alive(T const & t, boost::mpl::true_)
        {
            args.push_back(t.impl);
        }

#define HPX_LCOS_DATAFLOW_M0(Z, N, D)                                           \
        keep_arg_alive(                                                         \
            BOOST_PP_CAT(a, N)                                                  \
          , typename traits::is_dataflow<BOOST_PP_CAT(A, N)>::type()            \
        );
    /**/
#define HPX_LCOS_DATAFLOW_M1(Z, N, D)                                           \
        template <BOOST_PP_ENUM_PARAMS(N, typename A)>                          \
        dataflow_base_impl(                                                     \
            lcos::promise<naming::id_type, naming::gid_type> const & promise    \
          , BOOST_PP_ENUM_BINARY_PARAMS(N, A, const & a)                        \
        )                                                                       \
            : gid_promise(promise)                                              \
        {                                                                       \
            BOOST_PP_REPEAT(N, HPX_LCOS_DATAFLOW_M0, _)                         \
        }                                                                       \
    /**/
        BOOST_PP_REPEAT_FROM_TO(
            1
          , BOOST_PP_SUB(HPX_ACTION_ARGUMENT_LIMIT, 3)
          , HPX_LCOS_DATAFLOW_M1
          , _
        )

#undef HPX_LCOS_DATAFLOW_M0
#undef HPX_LCOS_DATAFLOW_M1

        void invalidate()
        {
            //gid_promise.reset();
            //std::vector<boost::shared_ptr<dataflow_base_impl> > t;
            //std::swap(t, args);
        }

        naming::id_type get_gid() const
        {
            return gid_promise.get();
        }

    protected:
        lcos::promise<naming::id_type, naming::gid_type> gid_promise;

    private:
        std::vector<boost::shared_ptr<dataflow_base_impl> > args;
        friend class boost::serialization::access;

        template <typename Archive>
        void load(Archive & ar, unsigned)
        {
            naming::id_type id;
            //ar & args;
            ar & id;
            gid_promise.set_local_data(0, id);
        }

        template <typename Archive>
        void save(Archive & ar, unsigned) const
        {
            //ar & args;
            //BOOST_ASSERT(this->get_gid());
            naming::id_type id = this->get_gid();
            ar & id;
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER();
    };
}}}

#endif
