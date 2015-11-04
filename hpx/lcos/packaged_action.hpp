//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_PACKAGED_ACTION_JUN_27_2008_0420PM)
#define HPX_LCOS_PACKAGED_ACTION_JUN_27_2008_0420PM

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/applier/apply_callback.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/protect.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/traits/component_type_is_compatible.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/static_assert.hpp>
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
    private:
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef promise<Result, typename action_type::remote_result_type> base_type;

        struct profiler_tag {};

        static void parcel_write_handler(
            boost::intrusive_ptr<typename base_type::wrapping_type> impl,
            boost::system::error_code const& ec, parcelset::parcel const& p)
        {
            // any error in the parcel layer will be stored in the future object
            if (ec) {
                boost::exception_ptr exception =
                    hpx::detail::get_exception(hpx::exception(ec),
                        "packaged_action::parcel_write_handler",
                        __FILE__, __LINE__, parcelset::dump_parcel(p));
                (*impl)->set_exception(exception);
            }
        }

        template <typename Callback>
        static void parcel_write_handler_cb(Callback const& cb,
            boost::intrusive_ptr<typename base_type::wrapping_type> impl,
            boost::system::error_code const& ec, parcelset::parcel const& p)
        {
            // any error in the parcel layer will be stored in the future object
            if (ec) {
                boost::exception_ptr exception =
                    hpx::detail::get_exception(hpx::exception(ec),
                        "packaged_action::parcel_write_handler",
                        __FILE__, __LINE__, parcelset::dump_parcel(p));
                (*impl)->set_exception(exception);
            }

            // invoke user supplied callback
            cb(ec, p);
        }

    public:
        /// Construct a (non-functional) instance of an \a packaged_action. To use
        /// this instance its member function \a apply needs to be directly
        /// called.
        packaged_action()
          : apply_logger_("packaged_action")
        {
            LLCO_(info) << "packaged_action::packaged_action("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ") args(0)";
        }

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        template <typename ...Ts>
        void apply(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
            Ts&&... vs)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

            naming::id_type cont_id(this->get_id());
            naming::detail::set_dont_store_in_cache(cont_id);

            hpx::apply_c_cb<action_type>(cont_id, gid,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1, util::placeholders::_2),
                std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        void apply(BOOST_SCOPED_ENUM(launch) policy, naming::address&& addr,
            naming::id_type const& gid, Ts&&... vs)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

            naming::id_type cont_id(this->get_id());
            naming::detail::set_dont_store_in_cache(cont_id);

            hpx::apply_c_cb<action_type>(cont_id, std::move(addr), gid,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1, util::placeholders::_2),
                std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename ...Ts>
        void apply_cb(BOOST_SCOPED_ENUM(launch) policy,
            naming::id_type const& gid, Callback && cb, Ts&&... vs)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

            typedef typename util::decay<Callback>::type callback_type;

            naming::id_type cont_id(this->get_id());
            naming::detail::set_dont_store_in_cache(cont_id);

            hpx::apply_c_cb<action_type>(cont_id, gid,
                util::bind(
                    &packaged_action::parcel_write_handler_cb<callback_type>,
                    util::protect(std::forward<Callback>(cb)), this->impl_,
                    util::placeholders::_1, util::placeholders::_2),
                std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename ...Ts>
        void apply_cb(BOOST_SCOPED_ENUM(launch) policy, naming::address&& addr,
            naming::id_type const& gid, Callback && cb, Ts&&... vs)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

            typedef typename util::decay<Callback>::type callback_type;

            naming::id_type cont_id(this->get_id());
            naming::detail::set_dont_store_in_cache(cont_id);

            hpx::apply_c_cb<action_type>(cont_id, std::move(addr), gid,
                util::bind(
                    &packaged_action::parcel_write_handler_cb<callback_type>,
                    util::protect(std::forward<Callback>(cb)), this->impl_,
                    util::placeholders::_1, util::placeholders::_2),
                std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        void apply_p(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
            threads::thread_priority priority, Ts&&... vs)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

            naming::id_type cont_id(this->get_id());
            naming::detail::set_dont_store_in_cache(cont_id);

            hpx::apply_c_p_cb<action_type>(cont_id, gid, priority,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1, util::placeholders::_2),
                std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        void apply_p(BOOST_SCOPED_ENUM(launch) policy, naming::address&& addr,
            naming::id_type const& gid, threads::thread_priority priority,
            Ts&&... vs)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

            naming::id_type cont_id(this->get_id());
            naming::detail::set_dont_store_in_cache(cont_id);

            hpx::apply_c_p_cb<action_type>(cont_id, std::move(addr),
                gid, priority,
                util::bind(&packaged_action::parcel_write_handler,
                    this->impl_, util::placeholders::_1, util::placeholders::_2),
                std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename ...Ts>
        void apply_p_cb(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
            threads::thread_priority priority, Callback && cb, Ts&&... vs)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

            typedef typename util::decay<Callback>::type callback_type;

            naming::id_type cont_id(this->get_id());
            naming::detail::set_dont_store_in_cache(cont_id);

            hpx::apply_c_p_cb<action_type>(cont_id, gid, priority,
                util::bind(
                    &packaged_action::parcel_write_handler_cb<callback_type>,
                    util::protect(std::forward<Callback>(cb)), this->impl_,
                    util::placeholders::_1, util::placeholders::_2),
                std::forward<Ts>(vs)...);
        }

        template <typename Callback, typename ...Ts>
        void apply_p_cb(BOOST_SCOPED_ENUM(launch) policy, naming::address&& addr,
            naming::id_type const& gid, threads::thread_priority priority,
            Callback && cb, Ts&&... vs)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

            typedef typename util::decay<Callback>::type callback_type;

            naming::id_type cont_id(this->get_id());
            naming::detail::set_dont_store_in_cache(cont_id);

            hpx::apply_c_p_cb<action_type>(cont_id, std::move(addr),
                gid, priority,
                util::bind(
                    &packaged_action::parcel_write_handler_cb<callback_type>,
                    util::protect(std::forward<Callback>(cb)), this->impl_,
                    util::placeholders::_1, util::placeholders::_2),
                std::forward<Ts>(vs)...);
        }

        /// Construct a new \a packaged_action instance. The \a thread
        /// supplied to the function \a packaged_action#get will be
        /// notified as soon as the result of the operation associated with
        /// this packaged_action instance has been returned.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        ///
        /// \note         The result of the requested operation is expected to
        ///               be returned as the first parameter using a
        ///               \a base_lco#set_value action. Any error has to be
        ///               reported using a \a base_lco::set_exception action. The
        ///               target for either of these actions has to be this
        ///               packaged_action instance (as it has to be sent along
        ///               with the action as the continuation parameter).
        template <typename ...Ts>
        packaged_action(naming::id_type const& gid, Ts&&... vs)
          : apply_logger_("packaged_action::apply")
        {
            LLCO_(info) << "packaged_action::packaged_action(" //-V128
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", "
                        << gid
                        << ") args(" << sizeof...(Ts) << ")";
            apply(launch::all, gid, std::forward<Ts>(vs)...);
        }

        template <typename ...Ts>
        packaged_action(naming::gid_type const& gid,
                threads::thread_priority priority, Ts&&... vs)
          : apply_logger_("packaged_action::apply")
        {
            LLCO_(info) << "packaged_action::packaged_action(" //-V128
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", "
                        << gid
                        << ") args(" << sizeof...(Ts) << ")";
            apply_p(launch::all, naming::id_type(gid, naming::id_type::unmanaged),
                priority, std::forward<Ts>(vs)...);
        }

        util::block_profiler<profiler_tag> apply_logger_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Result>
    class packaged_action<Action, Result, boost::mpl::true_>
      : public promise<Result,
          typename hpx::actions::extract_action<Action>::remote_result_type>
    {
    private:
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef promise<Result, typename action_type::remote_result_type> base_type;

        struct profiler_tag {};

        static void parcel_write_handler(
            boost::intrusive_ptr<typename base_type::wrapping_type> impl,
            boost::system::error_code const& ec, parcelset::parcel const& p)
        {
            // any error in the parcel layer will be stored in the future object
            if (ec) {
                boost::exception_ptr exception =
                    hpx::detail::get_exception(hpx::exception(ec),
                        "packaged_action::parcel_write_handler",
                        __FILE__, __LINE__, parcelset::dump_parcel(p));
                (*impl)->set_exception(exception);
            }
        }

        template <typename Callback>
        static void parcel_write_handler_cb(Callback const& cb,
            boost::intrusive_ptr<typename base_type::wrapping_type> impl,
            boost::system::error_code const& ec, parcelset::parcel const& p)
        {
            // any error in the parcel layer will be stored in the future object
            if (ec) {
                boost::exception_ptr exception =
                    hpx::detail::get_exception(hpx::exception(ec),
                        "packaged_action::parcel_write_handler",
                        __FILE__, __LINE__, parcelset::dump_parcel(p));
                (*impl)->set_exception(exception);
            }

            // invoke user supplied callback
            cb(ec, p);
        }

    public:
        /// Construct a (non-functional) instance of an \a packaged_action. To use
        /// this instance its member function \a apply needs to be directly
        /// called.
        packaged_action()
          : apply_logger_("packaged_action_direct::apply")
        {
            LLCO_(info) << "packaged_action::packaged_action("
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ") args(0)";
        }

        /// The apply function starts the asynchronous operations encapsulated
        /// by this eager future.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        template <typename ...Ts>
        void apply(BOOST_SCOPED_ENUM(launch) /*policy*/,
            naming::id_type const& gid, Ts&&... vs)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

            naming::address addr;
            if (agas::is_local_address_cached(gid, addr)) {
                // local, direct execution
                HPX_ASSERT(traits::component_type_is_compatible<
                    typename Action::component_type>::call(addr));

                (*this->impl_)->set_data(action_type::execute_function(
                    addr.address_, std::forward<Ts>(vs)...));
            }
            else {
                // remote execution
                naming::id_type cont_id(this->get_id());
                naming::detail::set_dont_store_in_cache(cont_id);

                hpx::applier::detail::apply_c_cb<action_type>(
                    std::move(addr), cont_id, gid,
                    util::bind(&packaged_action::parcel_write_handler,
                        this->impl_, util::placeholders::_1, util::placeholders::_2),
                    std::forward<Ts>(vs)...);
            }
        }

        template <typename ...Ts>
        void apply(BOOST_SCOPED_ENUM(launch) /*policy*/, naming::address&& addr,
            naming::id_type const& gid, Ts&&... vs)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

            if (addr.locality_ == hpx::get_locality()) {
                // local, direct execution
                HPX_ASSERT(traits::component_type_is_compatible<
                    typename Action::component_type>::call(addr));

                (*this->impl_)->set_data(action_type::execute_function(
                    addr.address_, std::forward<Ts>(vs)...));
            }
            else {
                // remote execution
                naming::id_type cont_id(this->get_id());
                naming::detail::set_dont_store_in_cache(cont_id);

                hpx::applier::detail::apply_c_cb<action_type>(
                    std::move(addr), cont_id, gid,
                    util::bind(&packaged_action::parcel_write_handler,
                        this->impl_, util::placeholders::_1, util::placeholders::_2),
                    std::forward<Ts>(vs)...);
            }
        }

        template <typename Callback, typename ...Ts>
        void apply_cb(BOOST_SCOPED_ENUM(launch) /*policy*/,
            naming::id_type const& gid, Callback && cb, Ts&&... vs)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

            naming::address addr;
            if (agas::is_local_address_cached(gid, addr)) {
                // local, direct execution
                HPX_ASSERT(traits::component_type_is_compatible<
                    typename Action::component_type>::call(addr));

                (*this->impl_)->set_data(action_type::execute_function(
                    addr.address_, std::forward<Ts>(vs)...));

                // invoke callback
                cb(boost::system::error_code(), parcelset::parcel());
            }
            else {
                // remote execution
                typedef typename util::decay<Callback>::type callback_type;

                naming::id_type cont_id(this->get_id());
                naming::detail::set_dont_store_in_cache(cont_id);

                hpx::applier::detail::apply_c_cb<action_type>(
                    std::move(addr), cont_id, gid,
                    util::bind(
                        &packaged_action::parcel_write_handler_cb<callback_type>,
                        util::protect(std::forward<Callback>(cb)), this->impl_,
                        util::placeholders::_1, util::placeholders::_2),
                    std::forward<Ts>(vs)...);
            }
        }

        template <typename Callback, typename ...Ts>
        void apply_cb(BOOST_SCOPED_ENUM(launch) /*policy*/, naming::address&& addr,
            naming::id_type const& gid, Callback && cb, Ts&&... vs)
        {
            util::block_profiler_wrapper<profiler_tag> bp(apply_logger_);

            if (addr.locality_ == hpx::get_locality()) {
                // local, direct execution
                HPX_ASSERT(traits::component_type_is_compatible<
                    typename Action::component_type>::call(addr));

                (*this->impl_)->set_data(action_type::execute_function(
                    addr.address_, std::forward<Ts>(vs)...));

                // invoke callback
                cb(boost::system::error_code(), parcelset::parcel());
            }
            else {
                // remote execution
                typedef typename util::decay<Callback>::type callback_type;

                naming::id_type cont_id(this->get_id());
                naming::detail::set_dont_store_in_cache(cont_id);

                hpx::applier::detail::apply_c_cb<action_type>(
                    std::move(addr), cont_id, gid,
                    util::bind(
                        &packaged_action::parcel_write_handler_cb<callback_type>,
                        util::protect(std::forward<Callback>(cb)), this->impl_,
                        util::placeholders::_1, util::placeholders::_2),
                    std::forward<Ts>(vs)...);
            }
        }

        /// Construct a new \a packaged_action instance. The \a thread
        /// supplied to the function \a packaged_action#get will be
        /// notified as soon as the result of the operation associated with
        /// this packaged_action instance has been returned.
        ///
        /// \param gid    [in] The global id of the target component to use to
        ///               apply the action.
        ///
        /// \note         The result of the requested operation is expected to
        ///               be returned as the first parameter using a
        ///               \a base_lco#set_value action. Any error has to be
        ///               reported using a \a base_lco::set_exception action. The
        ///               target for either of these actions has to be this
        ///               packaged_action instance (as it has to be sent along
        ///               with the action as the continuation parameter).
        template <typename ...Ts>
        packaged_action(naming::id_type const& gid, Ts&&... vs)
          : apply_logger_("packaged_action_direct::apply")
        {
            LLCO_(info) << "packaged_action::packaged_action(" //-V128
                        << hpx::actions::detail::get_action_name<action_type>()
                        << ", "
                        << gid
                        << ") args(" << sizeof...(Ts) << ")";
            apply(launch::all, gid, std::forward<Ts>(vs)...);
        }

        util::block_profiler<profiler_tag> apply_logger_;
    };
}}

#endif
