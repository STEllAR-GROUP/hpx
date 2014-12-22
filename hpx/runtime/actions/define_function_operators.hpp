//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_FUNCTION_OPERATORS_MAY_09_2012_0420PM)
#define HPX_RUNTIME_ACTIONS_FUNCTION_OPERATORS_MAY_09_2012_0420PM

    ///////////////////////////////////////////////////////////////////////////
    template <typename LocalResult, typename ...Ts>
    struct sync_invoke
    {
        BOOST_FORCEINLINE static LocalResult call(
            boost::mpl::false_, BOOST_SCOPED_ENUM(launch) policy,
            naming::id_type const& id, Ts&&... vs, error_code& ec = throws)
        {
            return hpx::async<basic_action>(policy, id,
                std::forward<Ts>(vs)...).get(ec);
        }

        BOOST_FORCEINLINE static LocalResult call(
            boost::mpl::true_, BOOST_SCOPED_ENUM(launch) policy,
            naming::id_type const& id, Ts&&... vs, error_code& = throws)
        {
            return hpx::async<basic_action>(policy, id,
                std::forward<Ts>(vs)...);
        }
    };

    template <typename ...Ts>
    BOOST_FORCEINLINE typename boost::enable_if_c<
        util::detail::pack<Ts...>::size == arity, 
        local_result_type
    >::type operator()(BOOST_SCOPED_ENUM(launch) policy,
        naming::id_type const& id, error_code& ec, Ts&&... vs) const
    {
        return util::void_guard<local_result_type>(),
            sync_invoke<local_result_type, Ts...>::call(
                is_future_pred(), policy, id,
                std::forward<Ts>(vs)..., ec);
    }

    template <typename ...Ts>
    BOOST_FORCEINLINE typename boost::enable_if_c<
        util::detail::pack<Ts...>::size == arity, 
        local_result_type
    >::type operator()(naming::id_type const& id, error_code& ec, Ts&&... vs) const
    {
        return (*this)(launch::all, id, ec, std::forward<Ts>(vs)...);
    }

    template <typename ...Ts>
    BOOST_FORCEINLINE typename boost::enable_if_c<
        util::detail::pack<Ts...>::size == arity, 
        local_result_type
    >::type operator()(BOOST_SCOPED_ENUM(launch) policy,
        naming::id_type const& id, Ts&&... vs) const
    {
        return (*this)(launch::all, id, throws, std::forward<Ts>(vs)...);
    }

    template <typename ...Ts>
    BOOST_FORCEINLINE typename boost::enable_if_c<
        util::detail::pack<Ts...>::size == arity, 
        local_result_type
    >::type operator()(naming::id_type const& id, Ts&&... vs) const
    {
        return (*this)(launch::all, id, throws, std::forward<Ts>(vs)...);
    }

#endif
