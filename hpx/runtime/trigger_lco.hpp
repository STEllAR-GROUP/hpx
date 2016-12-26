//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/trigger_lco.hpp

#if !defined(HPX_RUNTIME_TRIGGER_LCO_JUN_22_2015_0618PM)
#define HPX_RUNTIME_TRIGGER_LCO_JUN_22_2015_0618PM

#include <hpx/config.hpp>
#include <hpx/lcos_fwd.hpp>
#include <hpx/runtime/actions/continuation_fwd.hpp>
#include <hpx/runtime/actions/action_priority.hpp>
#include <hpx/runtime/applier/detail/apply_implementations_fwd.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/decay.hpp>

#include <boost/exception_ptr.hpp>

#include <type_traits>
#include <utility>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Trigger the LCO referenced by the given id
    ///
    /// \param id [in] This represents the id of the LCO which should be
    ///                triggered.
    /// \param addr [in] This represents the addr of the LCO which should be
    ///                triggered.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    HPX_API_EXPORT void trigger_lco_event(naming::id_type const& id,
        naming::address && addr, bool move_credits = true);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Trigger the LCO referenced by the given id
    ///
    /// \param id [in] This represents the id of the LCO which should be
    ///                triggered.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    inline void trigger_lco_event(naming::id_type const& id,
        bool move_credits = true)
    {
        trigger_lco_event(id, naming::address(), move_credits);
    }

    /// \brief Trigger the LCO referenced by the given id
    ///
    /// \param id   [in] This represents the id of the LCO which should be
    ///                  triggered.
    /// \param addr [in] This represents the addr of the LCO which should be
    ///                triggered.
    /// \param cont [in] This represents the LCO to trigger after completion.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    HPX_API_EXPORT void trigger_lco_event(naming::id_type const& id,
        naming::address && addr, naming::id_type const& cont,
        bool move_credits = true);

    /// \brief Trigger the LCO referenced by the given id
    ///
    /// \param id   [in] This represents the id of the LCO which should be
    ///                  triggered.
    /// \param cont [in] This represents the LCO to trigger after completion.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    inline void trigger_lco_event(naming::id_type const& id,
        naming::id_type const& cont, bool move_credits = true)
    {
        trigger_lco_event(id, naming::address(), cont, move_credits);
    }

    /// \brief Set the result value for the LCO referenced by the given id
    ///
    /// \param id [in] This represents the id of the LCO which should
    ///                receive the given value.
    /// \param addr [in] This represents the addr of the LCO which should be
    ///                triggered.
    /// \param t  [in] This is the value which should be sent to the LCO.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    template <typename Result>
    void set_lco_value(naming::id_type const& id,
        naming::address && addr, Result && t, bool move_credits = true);

    /// \brief Set the result value for the LCO referenced by the given id
    ///
    /// \param id [in] This represents the id of the LCO which should
    ///                receive the given value.
    /// \param t  [in] This is the value which should be sent to the LCO.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    template <typename Result>
    typename std::enable_if<
        !std::is_same<typename util::decay<Result>::type, naming::address>::value
    >::type
    set_lco_value(naming::id_type const& id, Result && t, bool move_credits = true)
    {
        set_lco_value(id, naming::address(), std::forward<Result>(t), move_credits);
    }

    /// \brief Set the result value for the LCO referenced by the given id
    ///
    /// \param id   [in] This represents the id of the LCO which should
    ///                  receive the given value.
    /// \param addr [in] This represents the addr of the LCO which should be
    ///                triggered.
    /// \param t    [in] This is the value which should be sent to the LCO.
    /// \param cont [in] This represents the LCO to trigger after completion.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    template <typename Result>
    void set_lco_value(naming::id_type const& id,
        naming::address && addr, Result && t, naming::id_type const& cont,
        bool move_credits = true);

    /// \brief Set the result value for the LCO referenced by the given id
    ///
    /// \param id   [in] This represents the id of the LCO which should
    ///                  receive the given value.
    /// \param t    [in] This is the value which should be sent to the LCO.
    /// \param cont [in] This represents the LCO to trigger after completion.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    template <typename Result>
    typename std::enable_if<
        !std::is_same<typename util::decay<Result>::type, naming::address>::value
    >::type
    set_lco_value(naming::id_type const& id, Result && t,
        naming::id_type const& cont, bool move_credits = true)
    {
        set_lco_value(id, naming::address(), std::forward<Result>(t), cont,
            move_credits);
    }

    /// \brief Set the error state for the LCO referenced by the given id
    ///
    /// \param id [in] This represents the id of the LCO which should
    ///                receive the error value.
    /// \param addr [in] This represents the addr of the LCO which should be
    ///                triggered.
    /// \param e  [in] This is the error value which should be sent to
    ///                the LCO.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    HPX_API_EXPORT void set_lco_error(naming::id_type const& id,
        naming::address && addr, boost::exception_ptr const& e,
        bool move_credits = true);

    /// \brief Set the error state for the LCO referenced by the given id
    ///
    /// \param id [in] This represents the id of the LCO which should
    ///                receive the error value.
    /// \param addr [in] This represents the addr of the LCO which should be
    ///                triggered.
    /// \param e  [in] This is the error value which should be sent to
    ///                the LCO.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    HPX_API_EXPORT void set_lco_error(naming::id_type const& id,
        naming::address && addr, boost::exception_ptr && e,
        bool move_credits = true);

    /// \brief Set the error state for the LCO referenced by the given id
    ///
    /// \param id [in] This represents the id of the LCO which should
    ///                receive the error value.
    /// \param e  [in] This is the error value which should be sent to
    ///                the LCO.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    inline void set_lco_error(naming::id_type const& id,
        boost::exception_ptr const& e, bool move_credits = true)
    {
        set_lco_error(id, naming::address(), e, move_credits);
    }

    /// \brief Set the error state for the LCO referenced by the given id
    ///
    /// \param id [in] This represents the id of the LCO which should
    ///                receive the error value.
    /// \param e  [in] This is the error value which should be sent to
    ///                the LCO.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    inline void set_lco_error(naming::id_type const& id,
        boost::exception_ptr && e, bool move_credits = true)
    {
        set_lco_error(id, naming::address(), std::move(e), move_credits);
    }

    /// \brief Set the error state for the LCO referenced by the given id
    ///
    /// \param id   [in] This represents the id of the LCO which should
    ///                  receive the error value.
    /// \param addr [in] This represents the addr of the LCO which should be
    ///                triggered.
    /// \param e    [in] This is the error value which should be sent to
    ///                  the LCO.
    /// \param cont [in] This represents the LCO to trigger after completion.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    HPX_API_EXPORT void set_lco_error(naming::id_type const& id,
        naming::address && addr, boost::exception_ptr const& e,
        naming::id_type const& cont, bool move_credits = true);

    /// \brief Set the error state for the LCO referenced by the given id
    ///
    /// \param id   [in] This represents the id of the LCO which should
    ///                  receive the error value.
    /// \param addr [in] This represents the addr of the LCO which should be
    ///                triggered.
    /// \param e    [in] This is the error value which should be sent to
    ///                  the LCO.
    /// \param cont [in] This represents the LCO to trigger after completion.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    HPX_API_EXPORT void set_lco_error(naming::id_type const& id,
        naming::address && addr, boost::exception_ptr && e,
        naming::id_type const& cont, bool move_credits = true);

    /// \brief Set the error state for the LCO referenced by the given id
    ///
    /// \param id   [in] This represents the id of the LCO which should
    ///                  receive the error value.
    /// \param e    [in] This is the error value which should be sent to
    ///                  the LCO.
    /// \param cont [in] This represents the LCO to trigger after completion.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    inline void set_lco_error(naming::id_type const& id,
        boost::exception_ptr const& e, naming::id_type const& cont,
        bool move_credits = true)
    {
        set_lco_error(id, naming::address(), e, cont, move_credits);
    }

    /// \brief Set the error state for the LCO referenced by the given id
    ///
    /// \param id   [in] This represents the id of the LCO which should
    ///                  receive the error value.
    /// \param e    [in] This is the error value which should be sent to
    ///                  the LCO.
    /// \param cont [in] This represents the LCO to trigger after completion.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    inline void set_lco_error(naming::id_type const& id,
        boost::exception_ptr && e, naming::id_type const& cont,
        bool move_credits = true)
    {
        set_lco_error(id, naming::address(), std::move(e), cont, move_credits);
    }

    //////////////////////////////////////////////////////////////////////////
    // forward declare the required overload of apply.
    template <typename Action, typename ...Ts>
    bool apply(naming::id_type const& gid, Ts&&... vs);

    template <
        typename Component, typename Signature, typename Derived,
        typename Cont, typename ...Ts>
    bool apply_continue(
        hpx::actions::basic_action<Component, Signature, Derived>,
        Cont&& cont, naming::id_type const& gid, Ts&&... vs);

    template <typename Component, typename Signature, typename Derived,
        typename ...Ts>
    inline bool
    apply_c(hpx::actions::basic_action<Component, Signature, Derived>,
        naming::id_type const& contgid, naming::id_type const& gid,
        Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    // handling special case of triggering an LCO
    ///////////////////////////////////////////////////////////////////////////

    /// \cond NOINTERNAL
    namespace detail
    {
        template <typename T>
        struct make_rvalue_impl
        {
            typedef T && type;

            template <typename U>
            HPX_FORCEINLINE static T && call(U& u)
            {
                return std::move(u);
            }
        };

        template <typename T>
        struct make_rvalue_impl<T const>
        {
            typedef T type;

            template <typename U>
            HPX_FORCEINLINE static T call(U const& u)
            {
                return u;
            }
        };

        template <typename T>
        struct make_rvalue_impl<T&>
        {
            typedef T type;

            HPX_FORCEINLINE static T call(T& u)
            {
                return u;
            }
        };

        template <typename T>
        struct make_rvalue_impl<T const&>
        {
            typedef T type;

            HPX_FORCEINLINE static T call(T const& u)
            {
                return u;
            }
        };

        template <typename T>
        HPX_FORCEINLINE typename detail::make_rvalue_impl<T>::type
        make_rvalue(typename std::remove_reference<T>::type& v)
        {
            return detail::make_rvalue_impl<T>::call(v);
        }

        template <typename T>
        HPX_FORCEINLINE typename detail::make_rvalue_impl<T>::type
        make_rvalue(typename std::remove_reference<T>::type&& v)
        {
            return detail::make_rvalue_impl<T>::call(v);
        }
    }
    /// \endcond

    template <typename Result>
    void set_lco_value(naming::id_type const& id, naming::address && addr,
        Result && t, bool move_credits)
    {
        typedef typename util::decay<Result>::type remote_result_type;
        typedef typename traits::promise_local_result<
                remote_result_type
            >::type local_result_type;
        typedef typename lcos::base_lco_with_value<
                local_result_type, remote_result_type
            >::set_value_action set_value_action;

        if (move_credits &&
            id.get_management_type() != naming::id_type::unmanaged)
        {
            naming::id_type target(id.get_gid(),
                naming::id_type::managed_move_credit);
            id.make_unmanaged();

            detail::apply_impl<set_value_action>(target, std::move(addr),
                actions::action_priority<set_value_action>(),
                detail::make_rvalue<Result>(t));
        }
        else
        {
            detail::apply_impl<set_value_action>(id, std::move(addr),
                actions::action_priority<set_value_action>(),
                detail::make_rvalue<Result>(t));
        }
    }

    template <typename Result>
    void set_lco_value(naming::id_type const& id, naming::address && addr,
        Result && t, naming::id_type const& cont, bool move_credits)
    {
        typedef typename util::decay<Result>::type remote_result_type;
        typedef typename traits::promise_local_result<
                remote_result_type
            >::type local_result_type;
        typedef typename lcos::base_lco_with_value<
                local_result_type, remote_result_type
            >::set_value_action set_value_action;

        if (move_credits &&
            id.get_management_type() != naming::id_type::unmanaged)
        {
            naming::id_type target(id.get_gid(),
                naming::id_type::managed_move_credit);
            id.make_unmanaged();

            detail::apply_impl<set_value_action>(
                actions::typed_continuation<
                    local_result_type, remote_result_type>(cont),
                target, std::move(addr),
                detail::make_rvalue<Result>(t));
        }
        else
        {
            detail::apply_impl<set_value_action>(
                actions::typed_continuation<
                    local_result_type, remote_result_type>(cont),
                id, std::move(addr),
                detail::make_rvalue<Result>(t));
        }
    }
}

#endif
