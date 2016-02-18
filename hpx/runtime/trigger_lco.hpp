//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/trigger_lco.hpp

#if !defined(HPX_RUNTIME_TRIGGER_LCO_JUN_22_2015_0618PM)
#define HPX_RUNTIME_TRIGGER_LCO_JUN_22_2015_0618PM

#include <hpx/config.hpp>
#include <hpx/config/export_definitions.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/decay.hpp>

#include <boost/exception_ptr.hpp>

#include <utility>
#include <type_traits>

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
    template <typename T>
    void set_lco_value(naming::id_type const& id,
        naming::address && addr, T && t, bool move_credits = true);

    /// \brief Set the result value for the LCO referenced by the given id
    ///
    /// \param id [in] This represents the id of the LCO which should
    ///                receive the given value.
    /// \param t  [in] This is the value which should be sent to the LCO.
    /// \param move_credits [in] If this is set to \a true then it is ok to
    ///                     send all credits in \a id along with the generated
    ///                     message. The default value is \a true.
    template <typename T>
    typename std::enable_if<
        !std::is_same<typename util::decay<T>::type, naming::address>::value
    >::type
    set_lco_value(naming::id_type const& id,
        T && t, bool move_credits = true)
    {
        set_lco_value(id, naming::address(), std::forward<T>(t), move_credits);
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
    template <typename T>
    void set_lco_value(naming::id_type const& id,
        naming::address && addr, T && t, naming::id_type const& cont,
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
    template <typename T>
    void set_lco_value(naming::id_type const& id, T && t,
        naming::id_type const& cont, bool move_credits = true)
    {
        set_lco_value(
            id, naming::address(), std::forward<T>(t), cont, move_credits);
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

    /// \copydoc hpx::set_lco_error(naming::id_type const& id,
    /// naming::address && addr, boost::exception_ptr const& e,
    /// bool move_credits)
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

    /// \copydoc hpx::set_lco_error(naming::id_type const& id,
    /// boost::exception_ptr const& e, bool move_credits)
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

    /// \copydoc hpx::set_lco_error(naming::id_type const& id,
    ///  naming::address && addr boost::exception_ptr const& e,
    ///  naming::id_type const& cont, bool move_credits)
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

    /// \copydoc hpx::set_lco_error(naming::id_type const& id,
    ///  boost::exception_ptr const& e, naming::id_type const& cont, bool move_credits)
    inline void set_lco_error(naming::id_type const& id,
        boost::exception_ptr && e, naming::id_type const& cont,
        bool move_credits = true)
    {
        set_lco_error(id, naming::address(), std::move(e), cont, move_credits);
    }
}

#endif
