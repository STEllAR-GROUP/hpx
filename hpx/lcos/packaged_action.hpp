//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_PACKAGED_ACTION_JUN_27_2008_0420PM)
#define HPX_LCOS_PACKAGED_ACTION_JUN_27_2008_0420PM

#include <hpx/config.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/applier/apply_callback.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/util/protect.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/traits/component_type_is_compatible.hpp>
#include <hpx/traits/component_supports_migration.hpp>
#include <hpx/traits/action_was_object_migrated.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/type_traits/is_same.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    /// A packaged_action can be used by a single \a thread to invoke a
    /// (remote) action and wait for the result. The result is expected to be
    /// sent back to the packaged_action using the LCO's set_event action
    ///
    /// A packaged_action is one of the simplest synchronization primitives
    /// provided by HPX. It allows to synchronize on a eager evaluated remote
    /// operation returning a result of the type \a Result.
    ///
    /// \tparam Action   The template parameter \a Action defines the action
    ///                  to be executed by this packaged_action instance. The
    ///                  arguments \a arg0,... \a argN are used as parameters
    ///                  for this action.
    /// \tparam Result   The template parameter \a Result defines the type this
    ///                  packaged_action is expected to return from its associated
    ///                  future \a packaged_action#get_future.
    /// \tparam DirectExecute The template parameter \a DirectExecute is an
    ///                  optimization aid allowing to execute the action
    ///                  directly if the target is local (without spawning a
    ///                  new thread for this). This template does not have to be
    ///                  supplied explicitly as it is derived from the template
    ///                  parameter \a Action.
    ///
    /// \note            The action executed using the packaged_action as a
    ///                  continuation must return a value of a type convertible
    ///                  to the type as specified by the template parameter
    ///                  \a Result.
    template <typename Action, typename Result, typename DirectExecute>
    class packaged_action;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    class packaged_action<Action, Result, boost::mpl::false_>
      : public promise<Result,
            typename hpx::actions::extract_action<Action>::remote_result_type>
    {
    protected:
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef promise<Result, typename action_type::remote_result_type> base_type;

        static void parcel_write_handler(
            boost::intrusive_ptr<typename base_type::wrapping_type> impl,
            boost::system::error_code const& ec, parcelset::parcel const& p)
        {
            // any error in the parcel layer will be stored in the future object
            if (ec)
            {
                boost::exception_ptr exception =
                    HPX_GET_EXCEPTION(ec,
                        "packaged_action::parcel_write_handler",
                        parcelset::dump_parcel(p));
                (*impl)->set_exception(exception);
            }
        }

        template <typename Callback>
        static void parcel_write_handler_cb(Callback const& cb,
            boost::intrusive_ptr<typename base_type::wrapping_type> impl,
            boost::system::error_code const& ec, parcelset::parcel const& p)
        {
            // any error in the parcel layer will be stored in the future object
            if (ec)
            {
                boost::exception_ptr exception =
                    HPX_GET_EXCEPTION(ec,
                        "packaged_action::parcel_write_handler",
                        parcelset::dump_parcel(p));
                (*impl)->set_exception(exception);
            }

            // invoke user supplied callback
            cb(ec, p);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ... Ts>
        void do_apply(naming::address && addr, naming::id_type const& id,
            threads::thread_priority priority, Ts&&... vs)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            LLCO_(info) << "packaged_action::do_apply(" //-V128
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", " << id << ") args(" << sizeof...(Ts) << ")";

            naming::id_type cont_id(this->get_id());
            naming::detail::set_dont_store_in_cache(cont_id);

            using util::placeholders::_1;
            using util::placeholders::_2;

            auto f = util::bind(&packaged_action::parcel_write_handler,
                this->impl_, _1, _2);

            if (addr)
            {
                hpx::apply_p_cb<action_type>(
                    actions::typed_continuation<result_type>(
                        std::move(cont_id), this->resolve()),
                    std::move(addr), id, priority, std::move(f),
                    std::forward<Ts>(vs)...);
            }
            else
            {
                hpx::apply_p_cb<action_type>(
                    actions::typed_continuation<result_type>(
                        std::move(cont_id), this->resolve()),
                    id, priority, std::move(f),
                    std::forward<Ts>(vs)...);
            }
        }

        template <typename ... Ts>
        void do_apply(naming::id_type const& id,
            threads::thread_priority priority, Ts&&... vs)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            LLCO_(info) << "packaged_action::do_apply(" //-V128
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", " << id << ") args(" << sizeof...(Ts) << ")";

            naming::id_type cont_id(this->get_id());
            naming::detail::set_dont_store_in_cache(cont_id);

            using util::placeholders::_1;
            using util::placeholders::_2;

            hpx::apply_p_cb<action_type>(
                actions::typed_continuation<result_type>(
                    std::move(cont_id), this->resolve()),
                id, priority,
                util::bind(
                    &packaged_action::parcel_write_handler,
                    this->impl_, _1, _2),
                std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename ... Ts>
        void do_apply_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback && cb, Ts&&... vs)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            LLCO_(info) << "packaged_action::do_apply_cb(" //-V128
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", " << id << ") args(" << sizeof...(Ts) << ")";

            naming::id_type cont_id(this->get_id());
            naming::detail::set_dont_store_in_cache(cont_id);

            using util::placeholders::_1;
            using util::placeholders::_2;

            typedef typename util::decay<Callback>::type callback_type;

            auto f = util::bind(
                &packaged_action::parcel_write_handler_cb<callback_type>,
                util::protect(std::forward<Callback>(cb)), this->impl_, _1, _2);

            if (addr)
            {
                hpx::apply_p_cb<action_type>(
                    actions::typed_continuation<result_type>(
                        std::move(cont_id), this->resolve()),
                    std::move(addr), id, priority, std::move(cb),
                    std::forward<Ts>(vs)...);
            }
            else
            {
                hpx::apply_p_cb<action_type>(
                    actions::typed_continuation<result_type>(
                        std::move(cont_id), this->resolve()),
                    id, priority, std::move(cb),
                    std::forward<Ts>(vs)...);
            }
        }

        template <typename Callback, typename ... Ts>
        void do_apply_cb(naming::id_type const& id,
            threads::thread_priority priority, Callback && cb, Ts&&... vs)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            LLCO_(info) << "packaged_action::do_apply_cb(" //-V128
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", " << id << ") args(" << sizeof...(Ts) << ")";

            naming::id_type cont_id(this->get_id());
            naming::detail::set_dont_store_in_cache(cont_id);

            using util::placeholders::_1;
            using util::placeholders::_2;

            typedef typename util::decay<Callback>::type callback_type;

            hpx::apply_p_cb<action_type>(
                actions::typed_continuation<result_type>(
                    std::move(cont_id), this->resolve()),
                id, priority,
                util::bind(
                    &packaged_action::parcel_write_handler_cb<callback_type>,
                    util::protect(std::forward<Callback>(cb)),
                    this->impl_, _1, _2),
                std::forward<Ts>(vs)...);
        }

    public:
        // Construct a (non-functional) instance of an \a packaged_action. To
        // use this instance its member function \a apply needs to be directly
        // called.
        packaged_action() {}

        ///////////////////////////////////////////////////////////////////////
        template <typename ...Ts>
        void apply(naming::id_type const& id, Ts &&... vs)
        {
            do_apply(id, actions::action_priority<action_type>(),
                std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        void apply(naming::address && addr, naming::id_type const& id,
            Ts &&... vs)
        {
            do_apply(std::move(addr), id,
                actions::action_priority<action_type>(),
                std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename ...Ts>
        void apply_cb(naming::id_type const& id, Callback && cb, Ts &&... vs)
        {
            do_apply_cb(id, actions::action_priority<action_type>(),
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename ...Ts>
        void apply_cb(naming::address && addr, naming::id_type const& id,
            Callback && cb, Ts &&... vs)
        {
            do_apply_cb(std::move(addr), id,
                actions::action_priority<action_type>(),
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        void apply_p(naming::id_type const& id,
            threads::thread_priority priority, Ts &&... vs)
        {
            do_apply(id, priority, std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        void apply_p(naming::address && addr, naming::id_type const& id,
            threads::thread_priority priority, Ts &&... vs)
        {
            do_apply(std::move(addr), id, priority, std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename ...Ts>
        void apply_p_cb(naming::id_type const& id,
            threads::thread_priority priority, Callback && cb, Ts &&... vs)
        {
            do_apply_cb(id, priority, std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename ...Ts>
        void apply_p_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback && cb, Ts&&... vs)
        {
            do_apply_cb(std::move(addr), id, priority,
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ...Ts>
        void apply_deferred(naming::address && addr,
            naming::id_type const& id, Ts &&... vs)
        {
            LLCO_(info) << "packaged_action::apply_deferred(" //-V128
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", " << id << ") args(" << sizeof...(Ts) << ")";

            naming::id_type cont_id(this->get_id());
            naming::detail::set_dont_store_in_cache(cont_id);

            using util::placeholders::_1;
            using util::placeholders::_2;

            auto f = hpx::functional::apply_c_p_cb<action_type>(
                cont_id, std::move(addr), id,
                actions::action_priority<action_type>(),
                util::bind(
                    &packaged_action::parcel_write_handler, this->impl_, _1, _2
                ),
                std::forward<Ts>(vs)...);

            this->base_type::set_task(std::move(f));
        }

        template <typename Callback, typename ...Ts>
        void apply_deferred_cb(naming::address && addr,
            naming::id_type const& id, Callback && cb, Ts &&... vs)
        {
            LLCO_(info) << "packaged_action::apply_deferred(" //-V128
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", " << id << ") args(" << sizeof...(Ts) << ")";

            naming::id_type cont_id(this->get_id());
            naming::detail::set_dont_store_in_cache(cont_id);

            using util::placeholders::_1;
            using util::placeholders::_2;

            typedef typename util::decay<Callback>::type callback_type;

            auto f = hpx::functional::apply_c_p_cb<action_type>(
                cont_id, std::move(addr), id,
                actions::action_priority<action_type>(),
                util::bind(
                     &packaged_action::parcel_write_handler_cb<callback_type>,
                     util::protect(std::forward<Callback>(cb)),
                     this->impl_, _1, _2
                ),
                std::forward<Ts>(vs)...);

            this->base_type::set_task(std::move(f));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    class packaged_action<Action, Result, boost::mpl::true_>
      : public packaged_action<Action, Result, boost::mpl::false_>
    {
        typedef typename packaged_action<
                Action, Result, boost::mpl::false_
            >::action_type action_type;

    public:
        /// Construct a (non-functional) instance of an \a packaged_action. To
        /// use this instance its member function \a apply needs to be directly
        /// called.
        packaged_action() {}

        ///////////////////////////////////////////////////////////////////////
        template <typename ...Ts>
        void apply(naming::id_type const& id, Ts &&... vs)
        {
            std::pair<bool, components::pinned_ptr> r;

            naming::address addr;
            if (agas::is_local_address_cached(id, addr))
            {
                typedef typename Action::component_type component_type;
                HPX_ASSERT(traits::component_type_is_compatible<
                    component_type>::call(addr));

                if (traits::component_supports_migration<component_type>::call())
                {
                    r = traits::action_was_object_migrated<Action>::call(
                            id, addr.address_);
                    if (!r.first)
                    {
                        // local, direct execution
                        (*this->impl_)->set_data(action_type::execute_function(
                            addr.address_, std::forward<Ts>(vs)...));
                        return;
                    }
                }
                else
                {
                    // local, direct execution
                    (*this->impl_)->set_data(action_type::execute_function(
                        addr.address_, std::forward<Ts>(vs)...));
                    return;
                }
            }

            // remote execution
            this->do_apply(id, actions::action_priority<action_type>(),
                std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        void apply(naming::address && addr, naming::id_type const& id,
            Ts &&... vs)
        {
            std::pair<bool, components::pinned_ptr> r;

            if (addr.locality_ == hpx::get_locality())
            {
                typedef typename Action::component_type component_type;
                HPX_ASSERT(traits::component_type_is_compatible<
                    component_type>::call(addr));

                if (traits::component_supports_migration<component_type>::call())
                {
                    r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                    if (!r.first)
                    {
                        // local, direct execution
                        (*this->impl_)->set_data(action_type::execute_function(
                            addr.address_, std::forward<Ts>(vs)...));
                        return;
                    }
                }
                else
                {
                    // local, direct execution
                    (*this->impl_)->set_data(action_type::execute_function(
                        addr.address_, std::forward<Ts>(vs)...));
                    return;
                }
            }

            // remote execution
            this->do_apply(std::move(addr), id,
                actions::action_priority<action_type>(),
                std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename ...Ts>
        void apply_cb(naming::id_type const& id, Callback && cb, Ts&&... vs)
        {
            std::pair<bool, components::pinned_ptr> r;

            naming::address addr;
            if (agas::is_local_address_cached(id, addr))
            {
                typedef typename Action::component_type component_type;
                HPX_ASSERT(traits::component_type_is_compatible<
                    component_type>::call(addr));

                if (traits::component_supports_migration<component_type>::call())
                {
                    r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                    if (!r.first)
                    {
                        // local, direct execution
                        (*this->impl_)->set_data(action_type::execute_function(
                            addr.address_, std::forward<Ts>(vs)...));

                        // invoke callback
                        cb(boost::system::error_code(), parcelset::parcel());
                        return;
                    }
                }
                else
                {
                    // local, direct execution
                    (*this->impl_)->set_data(action_type::execute_function(
                        addr.address_, std::forward<Ts>(vs)...));

                    // invoke callback
                    cb(boost::system::error_code(), parcelset::parcel());
                    return;
                }
            }

            // remote execution
            this->do_apply_cb(id, actions::action_priority<action_type>(),
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename ...Ts>
        void apply_cb(naming::address && addr, naming::id_type const& id,
            Callback && cb, Ts &&... vs)
        {
            std::pair<bool, components::pinned_ptr> r;

            if (addr.locality_ == hpx::get_locality())
            {
                typedef typename Action::component_type component_type;
                HPX_ASSERT(traits::component_type_is_compatible<
                    component_type>::call(addr));

                if (traits::component_supports_migration<component_type>::call())
                {
                    r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                    if (!r.first)
                    {
                        // local, direct execution
                        (*this->impl_)->set_data(action_type::execute_function(
                            addr.address_, std::forward<Ts>(vs)...));

                        // invoke callback
                        cb(boost::system::error_code(), parcelset::parcel());
                        return;
                    }
                }
                else
                {
                    // local, direct execution
                    (*this->impl_)->set_data(action_type::execute_function(
                        addr.address_, std::forward<Ts>(vs)...));

                    // invoke callback
                    cb(boost::system::error_code(), parcelset::parcel());
                    return;
                }
            }

            // remote execution
            this->do_apply_cb(std::move(addr), id,
                actions::action_priority<action_type>(),
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }
    };
}}

#endif
