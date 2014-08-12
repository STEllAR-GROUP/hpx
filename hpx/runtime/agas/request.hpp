////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_AB01A9FE_45BE_43EF_B9AD_05B701B06685)
#define HPX_AB01A9FE_45BE_43EF_B9AD_05B701B06685

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/function.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/runtime/agas/gva.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/traits/serialize_as_future.hpp>

#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/tracking.hpp>

// The number of types that the request's variant can represent.
#define HPX_AGAS_REQUEST_SUBTYPES 15

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
      , boost::int64_t count_
        );

    request(
        namespace_action_code type_
      , naming::gid_type const& gid_
      , boost::uint64_t count_
        );

    // REVIEW: Should the GVA here be a resolved address?
    request(
        namespace_action_code type_
      , naming::gid_type const& gid_
      , gva const& gva_
      , boost::uint32_t locality_
        );

    request(
        namespace_action_code type_
      , naming::gid_type const& gid_
        );

    request(
        namespace_action_code type_
      , naming::locality const& locality_
      , boost::uint64_t count_
      , boost::uint32_t num_threads_
      , naming::gid_type prefix_ = naming::gid_type()
        );

    request(
        namespace_action_code type_
      , naming::locality const& locality_
        );

    request(
        namespace_action_code type_
      , components::component_type ctype_
        );

    request(
        namespace_action_code type_
      , std::string const& name_
      , boost::uint32_t prefix_
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

    request(
        namespace_action_code type_
      , parcelset::parcel const& p
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

    ///////////////////////////////////////////////////////////////////////////
    // copy constructor
    request(
        request const& other
        );

    // copy assignment
    request& operator=(
        request const& other
        );

    gva get_gva(
        error_code& ec = throws
        ) const;

    boost::uint64_t get_count(
        error_code& ec = throws
        ) const;

    boost::int64_t get_credit(
        error_code& ec = throws
        ) const;

    components::component_type get_component_type(
        error_code& ec = throws
        ) const;

    boost::uint32_t get_locality_id(
        error_code& ec = throws
        ) const;

    iterate_names_function_type get_iterate_names_function(
        error_code& ec = throws
        ) const;

    iterate_types_function_type get_iterate_types_function(
        error_code& ec = throws
        ) const;

    parcelset::parcel get_parcel(
        error_code& ec = throws
        ) const;

    naming::locality get_locality(
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

    boost::uint32_t get_num_threads(
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
    friend class boost::serialization::access;


    void save(
        hpx::util::portable_binary_oarchive& ar
      , const unsigned int
        ) const;

    void load(
        hpx::util::portable_binary_iarchive& ar
      , const unsigned int
        );

    BOOST_SERIALIZATION_SPLIT_MEMBER()

    namespace_action_code mc;
    boost::shared_ptr<request_data> data;
};

}}

namespace hpx { namespace traits
{
    template <>
    struct serialize_as_future<hpx::agas::request>
      : boost::mpl::true_
    {
        static void call(hpx::agas::request& r)
        {
            if (r.get_action_code() == hpx::agas::primary_ns_route)
            {
                hpx::parcelset::parcel p = r.get_parcel();
                serialize_as_future<hpx::parcelset::parcel>::call(p);
            }
        }
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

#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
BOOST_CLASS_VERSION(hpx::agas::request, HPX_AGAS_VERSION)
BOOST_CLASS_TRACKING(hpx::agas::request, boost::serialization::track_never)
#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif

#endif // HPX_AB01A9FE_45BE_43EF_B9AD_05B701B06685

