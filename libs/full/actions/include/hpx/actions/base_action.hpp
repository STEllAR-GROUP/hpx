//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file base_action.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/traits/action_continuation.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/actions/actions_fwd.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>

#include <hpx/actions_base/actions_base_fwd.hpp>
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/actions_base/detail/action_factory.hpp>
#include <hpx/components_base/pinned_ptr.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/coroutines/thread_id_type.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/serialization/traits/needs_automatic_registration.hpp>
#include <hpx/serialization/traits/polymorphic_traits.hpp>
#include <hpx/serialization/tuple.hpp>

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
#include <hpx/modules/itt_notify.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions {
    ///////////////////////////////////////////////////////////////////////////
    // The \a base_action class is an abstract class used as the base class
    // for all action types. It's main purpose is to allow polymorphic
    // serialization of action instances through a unique_ptr.
    struct base_action
    {
        /// Destructor
        virtual ~base_action();

        /// The function \a get_component_type returns the \a component_type
        /// of the component this action belongs to.
        virtual int get_component_type() const = 0;

        /// The function \a get_action_name returns the name of this action
        /// (mainly used for debugging and logging purposes).
        virtual char const* get_action_name() const = 0;

        /// The function \a get_action_type returns whether this action needs
        /// to be executed in a new thread or directly.
        virtual action_flavor get_action_type() const = 0;

        virtual bool has_continuation() const = 0;

        /// The \a get_thread_function constructs a proper thread function for
        /// a \a thread, encapsulating the functionality and the arguments
        /// of the action it is called for.
        ///
        /// \param lva    [in] This is the local virtual address of the
        ///               component the action has to be invoked on.
        ///
        /// \returns      This function returns a proper thread function usable
        ///               for a \a thread.
        virtual threads::thread_function_type get_thread_function(
            naming::id_type&& target, naming::address_type lva,
            naming::component_type comptype) = 0;

        /// return the id of the locality of the parent thread
        virtual std::uint32_t get_parent_locality_id() const = 0;

        /// Return the thread id of the parent thread
        virtual threads::thread_id_type get_parent_thread_id() const = 0;

        /// Return the thread phase of the parent thread
        virtual std::uint64_t get_parent_thread_phase() const = 0;

        /// Return the thread priority this action has to be executed with
        virtual threads::thread_priority get_thread_priority() const = 0;

        /// Return the thread stacksize this action has to be executed with
        virtual threads::thread_stacksize get_thread_stacksize() const = 0;

        /// Return whether the embedded action is part of termination detection
        virtual bool does_termination_detection() const = 0;

        /// Perform thread initialization
        virtual void schedule_thread(naming::gid_type const& target,
            naming::address_type lva, naming::component_type comptype,
            std::size_t num_thread) = 0;

        /// Return whether the given object was migrated
        virtual std::pair<bool, components::pinned_ptr> was_object_migrated(
            hpx::naming::gid_type const&, naming::address_type) = 0;

        /// Return a pointer to the filter to be used while serializing an
        /// instance of this action type.
        virtual serialization::binary_filter* get_serialization_filter(
            parcelset::parcel const& p) const = 0;

        /// Return a pointer to the message handler to be used for this action.
        virtual parcelset::policies::message_handler* get_message_handler(
            parcelset::parcelhandler* ph, parcelset::locality const& loc,
            parcelset::parcel const& p) const = 0;

        virtual void load(serialization::input_archive& ar) = 0;
        virtual void save(serialization::output_archive& ar) = 0;

        virtual void load_schedule(serialization::input_archive& ar,
            naming::gid_type&& target, naming::address_type lva,
            naming::component_type comptype, std::size_t num_thread,
            bool& deferred_schedule) = 0;

        /// The function \a get_serialization_id returns the id which has been
        /// associated with this action (mainly used for serialization purposes).
        virtual std::uint32_t get_action_id() const = 0;

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
        /// The function \a get_action_name_itt returns the name of this action
        /// as a ITT string_handle
        virtual util::itt::string_handle const& get_action_name_itt() const = 0;
#endif
    };

    ///////////////////////////////////////////////////////////////////////////
    struct base_action_data : base_action
    {
        base_action_data() = default;

        base_action_data(threads::thread_priority priority,
            threads::thread_stacksize stacksize);

        /// Return the locality of the parent thread
        std::uint32_t get_parent_locality_id() const override;

        /// Return the thread id of the parent thread
        threads::thread_id_type get_parent_thread_id() const override;

        /// Return the phase of the parent thread
        std::uint64_t get_parent_thread_phase() const override;

        /// Return the thread priority this action has to be executed with
        threads::thread_priority get_thread_priority() const override;

        /// Return the thread stacksize this action has to be executed with
        threads::thread_stacksize get_thread_stacksize() const override;

    private:
        static std::uint32_t get_locality_id();

    protected:
        // serialization support
        void load_base(hpx::serialization::input_archive& ar);
        void save_base(hpx::serialization::output_archive& ar);

    protected:
        threads::thread_priority priority_;
        threads::thread_stacksize stacksize_;

#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
        std::uint32_t parent_locality_;
        threads::thread_id_type parent_id_;
        std::uint64_t parent_phase_;
#endif
    };
}}    // namespace hpx::actions

///////////////////////////////////////////////////////////////////////////////
// serialization support for basic_action
namespace hpx { namespace serialization {
    template <typename Archive, typename Component, typename R,
        typename... Args, typename Derived>
    HPX_FORCEINLINE void serialize(Archive& /* ar */,
        ::hpx::actions::basic_action<Component, R(Args...), Derived>& /* t */,
        unsigned int const /* version */ = 0)
    {
    }
}}    // namespace hpx::serialization

#include <hpx/config/warnings_suffix.hpp>

// Helper macro for action serialization, each of the defined actions needs to
// be registered with the serialization library
#define HPX_DEFINE_GET_ACTION_NAME(action)                                     \
    HPX_DEFINE_GET_ACTION_NAME_(action, action)                                \
    /**/

#define HPX_DEFINE_GET_ACTION_NAME_(action, actionname)                        \
    HPX_DEFINE_GET_ACTION_NAME_ITT(action, actionname)                         \
    namespace hpx { namespace actions { namespace detail {                     \
                template <>                                                    \
                HPX_ALWAYS_EXPORT char const* get_action_name<action>()        \
                {                                                              \
                    return HPX_PP_STRINGIZE(actionname);                       \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /**/
/**/

///////////////////////////////////////////////////////////////////////////////
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
#define HPX_DEFINE_GET_ACTION_NAME_ITT(action, actionname)                     \
    namespace hpx { namespace actions { namespace detail {                     \
                template <>                                                    \
                HPX_ALWAYS_EXPORT util::itt::string_handle const&              \
                get_action_name_itt<action>()                                  \
                {                                                              \
                    static util::itt::string_handle sh(                        \
                        HPX_PP_STRINGIZE(actionname));                         \
                    return sh;                                                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /**/

#define HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID_ITT(action)            \
    namespace hpx { namespace actions { namespace detail {                     \
                template <>                                                    \
                HPX_ALWAYS_EXPORT util::itt::string_handle const&              \
                get_action_name_itt<action>();                                 \
            }                                                                  \
        }                                                                      \
    }                                                                          \
/**/
#else    // HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
#define HPX_DEFINE_GET_ACTION_NAME_ITT(action, actionname)          /**/
#define HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID_ITT(action) /**/
#endif    // HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)

#define HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID(action)                \
    HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID_ITT(action)                \
    namespace hpx { namespace actions { namespace detail {                     \
                template <>                                                    \
                HPX_ALWAYS_EXPORT char const* get_action_name<action>();       \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    HPX_REGISTER_ACTION_EXTERN_DECLARATION(action)                             \
                                                                               \
    namespace hpx { namespace traits {                                         \
            template <>                                                        \
            struct is_action<action> : std::true_type                          \
            {                                                                  \
            };                                                                 \
            template <>                                                        \
            struct needs_automatic_registration<action> : std::false_type      \
            {                                                                  \
            };                                                                 \
        }                                                                      \
    }                                                                          \
    /**/

#define HPX_REGISTER_ACTION_DECLARATION_2(action, actionname)                  \
    HPX_REGISTER_ACTION_DECLARATION_NO_DEFAULT_GUID(action)                    \
    /**/

#if defined(HPX_MSVC) || defined(HPX_MINGW)
#define HPX_REGISTER_ACTION_2(action, actionname)                              \
    HPX_DEFINE_GET_ACTION_NAME_(action, actionname)                            \
    HPX_REGISTER_ACTION_INVOCATION_COUNT(action)                               \
    HPX_REGISTER_PER_ACTION_DATA_COUNTER_TYPES(action)                         \
    namespace hpx { namespace actions {                                        \
            template struct HPX_ALWAYS_EXPORT transfer_action<action>;         \
            template struct HPX_ALWAYS_EXPORT                                  \
                transfer_continuation_action<action>;                          \
        }                                                                      \
    }                                                                          \
/**/
#define HPX_REGISTER_ACTION_EXTERN_DECLARATION(action) /**/

#else    // defined(HPX_MSVC) || defined(HPX_MINGW)

#define HPX_REGISTER_ACTION_2(action, actionname)                              \
    HPX_DEFINE_GET_ACTION_NAME_(action, actionname)                            \
    HPX_REGISTER_ACTION_INVOCATION_COUNT(action)                               \
    HPX_REGISTER_PER_ACTION_DATA_COUNTER_TYPES(action)                         \
    namespace hpx { namespace actions {                                        \
            template struct transfer_action<action>;                           \
            template struct transfer_continuation_action<action>;              \
        }                                                                      \
    }                                                                          \
/**/
#define HPX_REGISTER_ACTION_EXTERN_DECLARATION(action)                         \
    namespace hpx { namespace actions {                                        \
            extern template struct HPX_ALWAYS_IMPORT transfer_action<action>;  \
            extern template struct HPX_ALWAYS_IMPORT                           \
                transfer_continuation_action<action>;                          \
        }                                                                      \
    }                                                                          \
    /**/

#endif    // defined(HPX_MSVC) || defined(HPX_MINGW)

HPX_TRAITS_SERIALIZED_WITH_ID(hpx::actions::base_action)
HPX_TRAITS_SERIALIZED_WITH_ID(hpx::actions::base_action_data)

#endif    // HPX_HAVE_NETWORKING
