////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_FB40C7A4_33B0_4C64_A16B_2A3FEEB237ED)
#define HPX_FB40C7A4_33B0_4C64_A16B_2A3FEEB237ED

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/traits/get_remote_result.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/runtime/agas/gva.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>

#include <boost/variant.hpp>
#include <boost/mpl/at.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/tracking.hpp>

#include <numeric>

namespace hpx { namespace agas
{

// TODO: Ensure that multiple invocations of get_data get optimized into the
// same jump table.
struct HPX_EXPORT response
{
  private:
    struct response_data;
  public:
    response();

    response(
        namespace_action_code type_
      , naming::gid_type lower_
      , naming::gid_type upper_
      , boost::uint32_t prefix_
      , error status_ = success
        );

    response(
        namespace_action_code type_
      , naming::gid_type const& gidbase_
      , gva const& gva_
      , boost::uint32_t locality_id_
      , error status_ = success
        );

    response(
        namespace_action_code type_
      , gva const& gva_
      , boost::uint32_t locality_id_
      , error status_ = success
        );

    response(
        namespace_action_code type_
      , components::component_type ctype_
      , error status_ = success
        );

    response(
        namespace_action_code type_
      , std::vector<boost::uint32_t> const& prefixes_
      , error status_ = success
        );

    response(
        namespace_action_code type_
      , naming::gid_type gid_
      , error status_ = success
        );

    response(
        namespace_action_code type_
      , boost::uint32_t prefix_
      , error status_ = success
        );

    response(
        namespace_action_code type_
      , std::string const& name_
      , error status_ = success
        );

    explicit response(
        namespace_action_code type_
      , error status_ = success
        );

    response(
        namespace_action_code type_
      , std::vector<naming::locality> const& localities_
      , error status_ = success
        );

    response(
        namespace_action_code type_
      , boost::int64_t added_credits_
      , error status_ = success
        );

    ///////////////////////////////////////////////////////////////////////////
    // copy constructor
    response(
        response const& other
        );

    // copy assignment
    response& operator=(
        response const& other
        );

    gva get_gva(
        error_code& ec = throws
        ) const;

    std::vector<boost::uint32_t> get_localities(
        error_code& ec = throws
        ) const;

    std::vector<naming::locality> get_resolved_localities(
        error_code& ec = throws
        ) const;

    boost::uint32_t get_num_localities(
        error_code& ec = throws
        ) const;

    boost::uint32_t get_num_overall_threads(
        error_code& ec = throws
        ) const;

    std::vector<boost::uint32_t> get_num_threads(
        error_code& ec = throws
        ) const;

    components::component_type get_component_type(
        error_code& ec = throws
        ) const;

    boost::uint32_t get_locality_id(
        error_code& ec = throws
        ) const;

    // primary_ns_change_credit_one
    boost::int64_t get_added_credits(
        error_code& ec = throws
        ) const;

    naming::gid_type get_base_gid(
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

    std::string get_component_typename(
        error_code& ec = throws
        ) const;

    naming::gid_type get_statistics_counter(
        error_code& ec = throws
        ) const;

    namespace_action_code get_action_code() const
    {
        return mc;
    }

    error get_status() const
    {
        return status;
    }

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
    error status;

    // FIXME: std::unique_ptr doesn't seem to work with incomplete types
    boost::shared_ptr<response_data> data;
};

}

namespace traits
{

// TODO: verification of namespace_action_code
template <>
struct get_remote_result<naming::id_type, agas::response>
{
    static naming::id_type call(
        agas::response const& rep
        )
    {
        switch(rep.get_action_code()) {
        case agas::symbol_ns_unbind:
        case agas::symbol_ns_resolve:
        case agas::primary_ns_statistics_counter:
        case agas::component_ns_statistics_counter:
        case agas::symbol_ns_statistics_counter:
            {
                naming::gid_type raw_gid = rep.get_gid();

                if (naming::detail::has_credits(raw_gid))
                    return naming::id_type(raw_gid, naming::id_type::managed);

                return naming::id_type(raw_gid, naming::id_type::unmanaged);
            }

        case agas::primary_ns_resolve_gid:
            {
                // return the wrapped locality_id
                return naming::get_id_from_locality_id(rep.get_locality_id());
            }
            break;

        default:
            break;
        }

        HPX_THROW_EXCEPTION(bad_parameter,
            "get_remote_result<naming::id_type, agas::response>::call",
            "unexpected action code in result conversion");
        return naming::invalid_id;
    }
};

// TODO: verification of namespace_action_code
template <>
struct get_remote_result<bool, agas::response>
{
    static bool call(
        agas::response const& rep
        )
    {
        return success == rep.get_status();
    }
};

template <>
struct get_remote_result<boost::uint32_t, agas::response>
{
    static boost::uint32_t call(
        agas::response const& rep
        )
    {
        switch(rep.get_action_code()) {
        case agas::locality_ns_num_localities:
        case agas::component_ns_num_localities:
            return rep.get_num_localities();

        case agas::locality_ns_num_threads:
            return rep.get_num_overall_threads();

        default:
            break;
        }
        HPX_THROW_EXCEPTION(bad_parameter,
            "get_remote_result<boost::uint32_t, agas::response>::call",
            "unexpected action code in result conversion");
        return 0;
    }
};

template <>
struct get_remote_result<boost::int64_t, agas::response>
{
    static boost::int64_t call(
        agas::response const& rep
        )
    {
        switch(rep.get_action_code()) {
        case agas::primary_ns_increment_credit:
            return rep.get_added_credits();

        default:
            break;
        }

        HPX_THROW_EXCEPTION(bad_parameter,
            "get_remote_result<boost::int64_t, agas::response>::call",
            "unexpected action code in result conversion");
        return 0;
    }
};

template <>
struct get_remote_result<std::vector<boost::uint32_t>, agas::response>
{
    static std::vector<boost::uint32_t> call(
        agas::response const& rep
        )
    {
        switch(rep.get_action_code()) {
        case agas::locality_ns_num_threads:
            return rep.get_num_threads();

        default:
            break;
        }
        HPX_THROW_EXCEPTION(bad_parameter,
            "get_remote_result<std::vector<boost::uint32_t>, agas::response>::call",
            "unexpected action code in result conversion");
        return std::vector<boost::uint32_t>();
    }
};

template <>
struct get_remote_result<std::vector<naming::locality>, agas::response>
{
    static std::vector<naming::locality> call(
        agas::response const& rep
        )
    {
        return rep.get_resolved_localities();
    }
};

}}

#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
BOOST_CLASS_VERSION(hpx::agas::response, HPX_AGAS_VERSION)
BOOST_CLASS_TRACKING(hpx::agas::response, boost::serialization::track_never)
#if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#if defined(HPX_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#pragma GCC diagnostic pop
#endif
#endif

HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    hpx::agas::response,
    agas_response_type)

HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    std::vector<hpx::agas::response>,
    agas_response_vector_type)

namespace hpx { namespace agas { namespace create_result_ns {
    typedef
        hpx::lcos::base_lco_with_value<bool, hpx::agas::response>
        base_lco_bool_response_type;
    typedef
        hpx::lcos::base_lco_with_value<hpx::naming::id_type, hpx::agas::response>
        base_lco_id_type_response_type;
}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::create_result_ns::base_lco_bool_response_type::set_value_action,
    set_value_action_agas_bool_response_type)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::create_result_ns::base_lco_id_type_response_type::set_value_action,
    set_value_action_agas_id_type_response_type)

#endif // HPX_FB40C7A4_33B0_4C64_A16B_2A3FEEB237ED

