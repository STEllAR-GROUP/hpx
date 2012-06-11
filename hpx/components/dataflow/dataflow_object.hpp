//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_DATAFLOW_OBJECT_HPP
#define HPX_COMPONENTS_DATAFLOW_OBJECT_HPP

#include <hpx/components/dataflow/dataflow.hpp>
#include <hpx/components/remote_object/object.hpp>

namespace hpx { namespace components
{
    namespace detail
    {
        template <typename T,
            typename Enable = typename hpx::traits::is_dataflow<T>::type>
        struct dataflow_result
        {
            typedef T type;
        };

        template <typename T>
        struct dataflow_result<T, boost::mpl::true_>
        {
            typedef typename T::result_type type;
        };
    }

    template <typename T>
    struct dataflow_object
    {
        dataflow_object() {}
        dataflow_object(dataflow_object const & o) : gid_(o.gid_) {}
        explicit dataflow_object(object<T> const & o) : gid_(o.gid_) {}
        explicit dataflow_object(naming::id_type const & gid) : gid_(gid) {}

        dataflow_object & operator=(dataflow_object const & o)
        {
            gid_ = o.gid_;
            return *this;
        }

        dataflow_object & operator=(object<T> const & o)
        {
            gid_ = o.gid_;
            return *this;
        }

        dataflow_object & operator=(naming::id_type const & gid)
        {
            gid_ = gid;
            return *this;
        }

        naming::id_type gid_;

        template <typename F>
        lcos::dataflow_base<
            typename boost::result_of<typename boost::remove_const<
                typename hpx::util::detail::remove_reference<F>::type
            >::type(T &)>::type
        >
        apply(BOOST_FWD_REF(F) f) const
        {
            typedef
                server::remote_object_apply_action1<
                    remote_object::invoke_apply_fun<
                        T
                      , typename boost::remove_const<
                            typename hpx::util::detail::remove_reference<F>::type
                        >::type
                    >
                >
                apply_action;

            return lcos::dataflow<apply_action>(gid_
                  , boost::move(remote_object::invoke_apply_fun<T, F>(boost::forward<F>(f)))
                );
        }

        template <typename F, typename D>
        lcos::dataflow_base<
            typename boost::result_of<typename boost::remove_const<
                typename hpx::util::detail::remove_reference<F>::type
            >::type(T &)>::type
        >
        apply(BOOST_FWD_REF(F) f, BOOST_FWD_REF(D) d) const
        {
            typedef
                server::remote_object_apply_action1<
                    remote_object::invoke_apply_fun<
                        T
                      , typename boost::remove_const<
                            typename hpx::util::detail::remove_reference<F>::type
                        >::type
                    >
                >
                apply_action;


            return lcos::dataflow<apply_action>(gid_
                  , boost::move(remote_object::invoke_apply_fun<T, F>(boost::forward<F>(f)))
                  , boost::forward<D>(d)
                );
        }

        template <typename F, typename A>
        lcos::dataflow_base<
            typename boost::result_of<typename boost::remove_const<
                typename hpx::util::detail::remove_reference<F>::type
            >::type(T &, A)>::type
        >
        apply2(BOOST_FWD_REF(F) f, BOOST_FWD_REF(A) a) const
        {
            typedef
                server::remote_object_apply_action2<
                    remote_object::invoke_apply_fun<
                        T
                      , typename boost::remove_const<
                            typename hpx::util::detail::remove_reference<F>::type
                        >::type
                    >
                  , typename detail::dataflow_result<A>::type
                >
                apply_action;


            return lcos::dataflow<apply_action>(gid_
                  , boost::move(remote_object::invoke_apply_fun<T, F>(boost::forward<F>(f)))
                  , boost::forward<A>(a)
                );
        }

        template <typename F, typename A0, typename A1>
        lcos::dataflow_base<
            typename boost::result_of<typename boost::remove_const<
                typename hpx::util::detail::remove_reference<F>::type
            >::type(T &)>::type
        >
        apply3(BOOST_FWD_REF(F) f, A0 const & a0, A1 const & a1) const
        {
            typedef
                server::remote_object_apply_action2<
                    remote_object::invoke_apply_fun<
                        T
                      , typename boost::remove_const<
                            typename hpx::util::detail::remove_reference<F>::type
                        >::type
                    >
                  , typename detail::dataflow_result<A0>::type
                >
                apply_action;


            return lcos::dataflow<apply_action>(gid_
                  , boost::move(remote_object::invoke_apply_fun<T, F>(boost::forward<F>(f)))
                  , a0
                  , a1
                );
        }

        template <typename F, typename A0, typename A1, typename A2>
        lcos::dataflow_base<
            typename boost::result_of<typename boost::remove_const<
                typename hpx::util::detail::remove_reference<F>::type
            >::type(T &)>::type
        >
        apply4(BOOST_FWD_REF(F) f, A0 const & a0, A1 const & a1, A2 const & a2) const
        {
            typedef
                server::remote_object_apply_action2<
                    remote_object::invoke_apply_fun<
                        T
                      , typename boost::remove_const<
                            typename hpx::util::detail::remove_reference<F>::type
                        >::type
                    >
                  , typename detail::dataflow_result<A0>::type
                >
                apply_action;


            return lcos::dataflow<apply_action>(gid_
                  , boost::move(remote_object::invoke_apply_fun<T, F>(boost::forward<F>(f)))
                  , a0
                  , a1
                  , a2
                );
        }

        template <typename F, typename A0, typename A1, typename A2, typename A3>
        lcos::dataflow_base<
            typename boost::result_of<typename boost::remove_const<
                typename hpx::util::detail::remove_reference<F>::type
            >::type(T &)>::type
        >
        apply5(BOOST_FWD_REF(F) f, A0 const & a0, A1 const & a1, A2 const & a2, A3 const & a3) const
        {
            typedef
                server::remote_object_apply_action2<
                    remote_object::invoke_apply_fun<
                        T
                      , typename boost::remove_const<
                            typename hpx::util::detail::remove_reference<F>::type
                        >::type
                    >
                  , typename detail::dataflow_result<A0>::type
                >
                apply_action;


            return lcos::dataflow<apply_action>(gid_
                  , boost::move(remote_object::invoke_apply_fun<T, F>(boost::forward<F>(f)))
                  , a0
                  , a1
                  , a2
                  , a3
                );
        }

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            ar & gid_;
        }
    };
}}

#endif
