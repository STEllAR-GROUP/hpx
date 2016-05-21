////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2014-2015 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef HPX_RUNTIME_AGAS_REQUEST_HPP
#define HPX_RUNTIME_AGAS_REQUEST_HPP

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/runtime/agas/gva.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/util/function.hpp>

#include <cstdint>
#include <memory>
#include <string>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace agas
{

// TODO: Ensure that multiple invocations of get_data get optimized into the
// same jump table.
struct HPX_EXPORT request
{
  private:
    struct request_data;

  public:
    typedef hpx::util::function<
        void(std::string const&, naming::gid_type const&)
    > iterate_names_function_type;

    typedef hpx::util::function<
        void(std::string const&, components::component_type)
    > iterate_types_function_type;

    request();

    request(
        namespace_action_code type_
      , naming::gid_type const& lower_
      , naming::gid_type const& upper_
      , std::int64_t count_
        );

    request(
        namespace_action_code type_
      , naming::gid_type const& gid_
      , std::uint64_t count_
        );

    // REVIEW: Should the GVA here be a resolved address?
    request(
        namespace_action_code type_
      , naming::gid_type const& gid_
      , gva const& gva_
      , naming::gid_type locality_
        );

    request(
        namespace_action_code type_
      , naming::gid_type const& gid_
        );

    request(
        namespace_action_code type_
      , parcelset::endpoints_type const & endpoints_
      , std::uint64_t count_
      , std::uint32_t num_threads_
      , naming::gid_type prefix_ = naming::gid_type()
        );

    request(
        namespace_action_code type_
      , components::component_type ctype_
        );

    request(
        namespace_action_code type_
      , std::string const& name_
      , std::uint32_t prefix_
        );

    request(
        namespace_action_code type_
      , std::string const& name_
      , naming::gid_type const& gid_
        );

    request(
        namespace_action_code type_
      , std::string const& name_
        );

    request(
        namespace_action_code type_
      , iterate_names_function_type const& f_
        );

    request(
        namespace_action_code type_
      , iterate_types_function_type const& f_
        );

    explicit request(
        namespace_action_code type_
        );

    request(
        namespace_action_code type_
      , std::string const& name
      , namespace_action_code evt
      , bool call_for_past_events
      , hpx::id_type result_lco
        );

    ~request();

    ///////////////////////////////////////////////////////////////////////////
    // copy constructor
    request(
        request const& other
        );
    // move constructor
    request(
        request&& other
        );

    // copy assignment
    request& operator=(
        request const& other
        );

    // move assignment
    request& operator=(
        request&& other
        );

    gva get_gva(
        error_code& ec = throws
        ) const;

    std::uint64_t get_count(
        error_code& ec = throws
        ) const;

    std::int64_t get_credit(
        error_code& ec = throws
        ) const;

    components::component_type get_component_type(
        error_code& ec = throws
        ) const;

    std::uint32_t get_locality_id(
        error_code& ec = throws
        ) const;

    naming::gid_type get_locality(
        error_code& ec = throws
        ) const;

    iterate_names_function_type get_iterate_names_function(
        error_code& ec = throws
        ) const;

    iterate_types_function_type get_iterate_types_function(
        error_code& ec = throws
        ) const;

    parcelset::endpoints_type get_endpoints(
        error_code& ec = throws
    ) const;

    naming::gid_type get_gid(
        error_code& ec = throws
        ) const;

    naming::gid_type get_lower_bound(
        error_code& ec = throws
        ) const;

    naming::gid_type get_upper_bound(
        error_code& ec = throws
        ) const;

    std::string get_name(
        error_code& ec = throws
        ) const;

    namespace_action_code get_action_code() const
    {
        return mc;
    }

    std::string get_statistics_counter_name(
        error_code& ec = throws
        ) const;

    std::uint32_t get_num_threads(
        error_code& ec = throws
        ) const;

    naming::gid_type get_suggested_prefix(
        error_code& ec = throws
        ) const;

    namespace_action_code get_on_event_event(
        error_code& ec = throws
        ) const;

    bool get_on_event_call_for_past_event(
        error_code& ec = throws
        ) const;

    hpx::id_type get_on_event_result_lco(
        error_code& ec = throws
        ) const;

  private:
    friend class hpx::serialization::access;


    void save(
        serialization::output_archive& ar
      , const unsigned int
        ) const;

    void load(
        serialization::input_archive& ar
      , const unsigned int
        );

    HPX_SERIALIZATION_SPLIT_MEMBER()

    namespace_action_code mc; //-V707
    std::unique_ptr<request_data> data;
};

}}

HPX_UTIL_REGISTER_FUNCTION_DECLARATION(
    void(std::string const&, hpx::naming::gid_type const&)
  , hpx::util::function<void(std::string const&, hpx::naming::gid_type const&)>
  , request_iterate_names_function_type
)

HPX_UTIL_REGISTER_FUNCTION_DECLARATION(
    void(std::string const&, hpx::components::component_type)
  , hpx::util::function<void(std::string const&, hpx::components::component_type)>
  , request_iterate_types_function_type
)

#include <hpx/config/warnings_suffix.hpp>

#endif // HPX_AB01A9FE_45BE_43EF_B9AD_05B701B06685

