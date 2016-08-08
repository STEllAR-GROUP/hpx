////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Thomas Heller
//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// make inspect happy: hpxinspect:nodeprecatedinclude hpxinspect:nodeprecatedname

#include <hpx/runtime/agas/request.hpp>

#include <hpx/error_code.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/runtime/agas/gva.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/mpl/at.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/variant.hpp>

#include <cstdint>
#include <string>
#include <utility>

// The number of types that the request's variant can represent.
#define HPX_AGAS_REQUEST_SUBTYPES 13

namespace hpx { namespace agas
{
#if defined(HPX_MSVC)
#pragma warning (push)
#pragma warning (disable: 4521)
#endif
    struct request::request_data
    {
        request_data()
          : data(hpx::util::make_tuple())
        {}

        request_data(request_data&& other)
          : data(std::move(other.data))
        {}

        request_data(request_data const& other)
          : data(other.data)
        {}

        request_data(request_data& other)
          : data(other.data)
        {}

        template <typename Tuple>
        explicit request_data(Tuple&& tuple)
          : data(std::forward<Tuple>(tuple))
        {}

        template <typename Tuple>
        request_data& operator=(Tuple&& tuple)
        {
            data = std::forward<Tuple>(tuple);
            return *this;
        }

        int which() const
        {
            return data.which();
        }

        enum subtype
        {
            subtype_gid_gid_credit          = 0x0
          , subtype_gid_count               = 0x1
          , subtype_gid_gva_prefix          = 0x2
          , subtype_gid                     = 0x3
          , subtype_locality_count          = 0x4
          , subtype_ctype                   = 0x5
          , subtype_name_prefix             = 0x6
          , subtype_name_gid                = 0x7
          , subtype_name                    = 0x8
          , subtype_iterate_names_function  = 0x9
          , subtype_iterate_types_function  = 0xa
          , subtype_void                    = 0xb
          , subtype_name_evt_id             = 0xc
          // update HPX_AGAS_REQUEST_SUBTYPES above if you add more entries
        };

        // The types listed for any of the services represent the argument types
        // for this particular service.
        // The order of the variant types is significant, and should not be changed
        typedef boost::variant<
            // 0x0
            // primary_ns_increment_credit
            // primary_ns_decrement_credit
            util::tuple<
                naming::gid_type // lower
              , naming::gid_type // upper
              , std::int64_t   // credit
            >
            // 0x1
            // primary_ns_unbind_gid
          , util::tuple<
                naming::gid_type // gid
              , std::uint64_t  // count
            >
            // 0x2
            // primary_ns_bind_gid
          , util::tuple<
                naming::gid_type // gid
              , gva              // resolved address
              , naming::gid_type // locality
            >
            // 0x3
            // primary_ns_resolve_gid
            // primary_ns_change_credit_one
            // primary_ns_free
            // primary_ns_resolve_locality
            // primary_ns_begin_migration
            // primary_ns_end_migration
          , util::tuple<
                naming::gid_type // gid
            >
            // 0x4
            // primary_ns_allocate
          , util::tuple<
                parcelset::endpoints_type // endpoints
              , std::uint64_t           // count
              , std::uint32_t           // num_threads
              , naming::gid_type          // suggested prefix
            >
            // 0x5
            // component_ns_resolve_id
            // component_ns_get_component_type
          , util::tuple<
                components::component_type // ctype
            >
            // 0x6
            // component_ns_bind_prefix
          , util::tuple<
                std::string     // name
              , std::uint32_t // prefix
            >
            // 0x7
            // symbol_ns_bind
          , util::tuple<
                std::string      // name
              , naming::gid_type // gid
            >
            // 0x8
            // component_ns_bind_name
            // component_ns_unbind_name
            // symbol_ns_resolve
            // symbol_ns_unbind
            // component_ns_statistics
            // primary_ns_statistics
            // symbol_ns_statistics
          , util::tuple<
                std::string // name
            >
            // 0x9
            // symbol_ns_iterate_names
          , util::tuple<
                iterate_names_function_type // f
            >
            // 0xa
            // component_ns_iterate_types
          , util::tuple<
                iterate_types_function_type // f
            >
            // 0xb
            // primary_ns_localities
            // primary_ns_resolved_localities
          , util::tuple<
            >
            // 0xc
            // symbol_ns_on_event
          , util::tuple<
                std::string
              , namespace_action_code
              , bool
              , hpx::id_type
            >
        > data_type;

        // {{{ variant helper TODO: consolidate with helpers in response
        template <
            subtype Type
          , int N
        >
        typename util::tuple_element<
            N
          , typename boost::mpl::at_c<
                typename data_type::types, Type
            >::type
        >::type
        get_data(
            error_code& ec = throws
            ) const
        { // {{{
            typedef typename boost::mpl::at_c<
                typename data_type::types, Type
            >::type vector_type;

            typedef typename util::tuple_element<
                N, vector_type
            >::type return_type;

            switch (data.which())
            {
                case Type:
                {
                    vector_type const* v = boost::get<vector_type>(&data);

                    if (!v)
                    {
                        HPX_THROWS_IF(ec, invalid_data
                          , "request::get_data"
                          , "internal data corruption");
                        return return_type();
                    }

                    if (&ec != &throws)
                        ec = make_success_code();

                    return util::get<N>(*v);
                }

                default: {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "request::get_data",
                        "invalid operation for request type");
                    return return_type();
                }
            };
        } // }}}
        // }}}

        data_type data;
    };
#if defined(HPX_MSVC)
#pragma warning (pop)
#endif

    request::request()
        : mc(invalid_request)
        , data(new request_data(util::make_tuple()))
    {}

    request::request(
        namespace_action_code type_
      , naming::gid_type const& lower_
      , naming::gid_type const& upper_
      , std::int64_t count_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(lower_, upper_, count_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , naming::gid_type const& gid_
      , std::uint64_t count_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(gid_, count_)))
    {
        // TODO: verification of namespace_action_code
    }

    // REVIEW: Should the GVA here be a resolved address?
    request::request(
        namespace_action_code type_
      , naming::gid_type const& gid_
      , gva const& gva_
      , naming::gid_type locality_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(gid_, gva_, locality_)))
    {
        HPX_ASSERT(type_ == primary_ns_bind_gid);
    }

    // primary_ns_resolve_gid, primary_ns_change_credit_one
    request::request(
        namespace_action_code type_
      , naming::gid_type const& gid_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(gid_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , parcelset::endpoints_type const & endpoints_
      , std::uint64_t count_
      , std::uint32_t num_threads_
      , naming::gid_type prefix_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(endpoints_,
          count_, num_threads_, prefix_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , components::component_type ctype_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(ctype_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , std::string const& name_
      , std::uint32_t prefix_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(name_, prefix_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , std::string const& name_
      , naming::gid_type const& gid_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(name_, gid_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , std::string const& name_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(name_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , iterate_names_function_type const& f_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(f_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , iterate_types_function_type const& f_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(f_)))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
        )
      : mc(type_)
      , data(new request_data(util::make_tuple()))
    {
        // TODO: verification of namespace_action_code
    }

    request::request(
        namespace_action_code type_
      , std::string const& name
      , namespace_action_code evt
      , bool call_for_past_events
      , hpx::id_type result_lco
        )
      : mc(type_)
      , data(new request_data(util::make_tuple(name,
          evt, call_for_past_events, result_lco)))
    {
        HPX_ASSERT(type_ == symbol_ns_on_event);
    }

    request::~request()
    {}

    ///////////////////////////////////////////////////////////////////////////
    // copy constructor
    request::request(
        request const& other
        )
      : mc(other.mc)
      , data(new request_data(*other.data))
    {}

    // move constructor
    request::request(
        request && other
        )
      : mc(other.mc)
      , data(std::move(other.data))
    {
        other.mc = invalid_request;
    }

    // copy assignment
    request& request::operator=(
        request const& other
        )
    {
        mc = other.mc;
        data.reset(new request_data(*other.data));
        return *this;
    }

    // move assignment
    request& request::operator=(
        request&& other
        )
    {
        mc = other.mc;
        data = std::move(other.data);
        other.mc = invalid_request;
        return *this;
    }

    ///////////////////////////////////////////////////////////////////////////
    gva request::get_gva(
        error_code& ec
        ) const
    {
        HPX_ASSERT(data->which() == request_data::subtype_gid_gva_prefix);
        return data->get_data<request_data::subtype_gid_gva_prefix, 1>(ec);
    }

    std::uint64_t request::get_count(
        error_code& ec
        ) const
    { // {{{
        switch (data->which())
        {
            case request_data::subtype_gid_count:
                return data->get_data<request_data::subtype_gid_count, 1>(ec);

            case request_data::subtype_locality_count:
                return data->get_data<request_data::subtype_locality_count, 1>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "request::get_count",
                    "invalid operation for request type");
                return 0;
            }
        }
    } // }}}

    std::int64_t request::get_credit(
        error_code& ec
        ) const
    {
        HPX_ASSERT(data->which() == request_data::subtype_gid_gid_credit);
        return data->get_data<request_data::subtype_gid_gid_credit, 2>(ec);
    }

    components::component_type request::get_component_type(
        error_code& ec
        ) const
    {
        HPX_ASSERT(data->which() == request_data::subtype_ctype);
        return data->get_data<request_data::subtype_ctype, 0>(ec);
    }

    std::uint32_t request::get_locality_id(
        error_code& ec
        ) const
    {
        switch (data->which())
        {
            case request_data::subtype_gid_gva_prefix:
                return naming::get_locality_id_from_gid(
                    data->get_data<request_data::subtype_gid_gva_prefix, 2>(ec));

            case request_data::subtype_name_prefix:
                return data->get_data<request_data::subtype_name_prefix, 1>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "request::get_locality_id",
                    "invalid operation for request type");
                return naming::invalid_locality_id;
            }
        }
    }

    naming::gid_type request::get_locality(
        error_code& ec
        ) const
    {
        HPX_ASSERT(data->which() == request_data::subtype_gid_gva_prefix);
        return data->get_data<request_data::subtype_gid_gva_prefix, 2>(ec);
    }

    request::iterate_names_function_type request::get_iterate_names_function(
        error_code& ec
        ) const
    {
        HPX_ASSERT(data->which() == request_data::subtype_iterate_names_function);
        return data->get_data<request_data::subtype_iterate_names_function, 0>(ec);
    }

    request::iterate_types_function_type request::get_iterate_types_function(
        error_code& ec
        ) const
    {
        HPX_ASSERT(data->which() == request_data::subtype_iterate_types_function);
        return data->get_data<request_data::subtype_iterate_types_function, 0>(ec);
    }

    parcelset::endpoints_type request::get_endpoints(
        error_code& ec
        ) const
    { // {{{
        switch (data->which())
        {
            case request_data::subtype_locality_count:
                return data->get_data<request_data::subtype_locality_count, 0>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "request::get_endpoints",
                    "invalid operation for request type");
                return parcelset::endpoints_type();
            }
        }
    } // }}}

    naming::gid_type request::get_gid(
        error_code& ec
        ) const
    { // {{{
        switch (data->which())
        {
            case request_data::subtype_gid:
                return data->get_data<request_data::subtype_gid, 0>(ec);

            case request_data::subtype_gid_gva_prefix:
                return data->get_data<request_data::subtype_gid_gva_prefix, 0>(ec);

            case request_data::subtype_gid_count:
                return data->get_data<request_data::subtype_gid_count, 0>(ec);

            case request_data::subtype_name_gid:
                return data->get_data<request_data::subtype_name_gid, 1>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "request::get_gid",
                    "invalid operation for request type");
                return naming::invalid_gid;
            }
        }
    } // }}}

    naming::gid_type request::get_lower_bound(
        error_code& ec
        ) const
    {
        HPX_ASSERT(data->which() == request_data::subtype_gid_gid_credit);
        return data->get_data<request_data::subtype_gid_gid_credit, 0>(ec);
    }

    naming::gid_type request::get_upper_bound(
        error_code& ec
        ) const
    {
        HPX_ASSERT(data->which() == request_data::subtype_gid_gid_credit);
        return data->get_data<request_data::subtype_gid_gid_credit, 1>(ec);
    }

    std::string request::get_name(
        error_code& ec
        ) const
    { // {{{
        switch (data->which())
        {
            case request_data::subtype_name:
                return data->get_data<request_data::subtype_name, 0>(ec);

            case request_data::subtype_name_prefix:
                return data->get_data<request_data::subtype_name_prefix, 0>(ec);

            case request_data::subtype_name_gid:
                return data->get_data<request_data::subtype_name_gid, 0>(ec);

            case request_data::subtype_name_evt_id:
                return data->get_data<request_data::subtype_name_evt_id, 0>(ec);

            default: {
                HPX_THROWS_IF(ec, bad_parameter,
                    "request::get_name",
                    "invalid operation for request type");
                return "";
            }
        };
    } // }}}

    std::string request::get_statistics_counter_name(
        error_code& ec
        ) const
    {
        HPX_ASSERT(data->which() == request_data::subtype_name);
        return data->get_data<request_data::subtype_name, 0>(ec);
    }

    std::uint32_t request::get_num_threads(
        error_code& ec
        ) const
    {
        HPX_ASSERT(data->which() == request_data::subtype_locality_count);
        return data->get_data<request_data::subtype_locality_count, 2>(ec);
    }

    naming::gid_type request::get_suggested_prefix(
        error_code& ec
        ) const
    {
        HPX_ASSERT(data->which() == request_data::subtype_locality_count);
        return data->get_data<request_data::subtype_locality_count, 3>(ec);
    }

    namespace_action_code request::get_on_event_event(
        error_code& ec
        ) const
    {
        HPX_ASSERT(data->which() == request_data::subtype_name_evt_id);
        return data->get_data<request_data::subtype_name_evt_id, 1>(ec);
    }

    bool request::get_on_event_call_for_past_event(
        error_code& ec
        ) const
    {
        HPX_ASSERT(data->which() == request_data::subtype_name_evt_id);
        return data->get_data<request_data::subtype_name_evt_id, 2>(ec);
    }

    hpx::id_type request::get_on_event_result_lco(
        error_code& ec
        ) const
    {
        HPX_ASSERT(data->which() == request_data::subtype_name_evt_id);
        return data->get_data<request_data::subtype_name_evt_id, 3>(ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    struct save_visitor : boost::static_visitor<void>
    {
      private:
        hpx::serialization::output_archive& ar;

      public:
        save_visitor(hpx::serialization::output_archive& ar_)
          : ar(ar_)
        {}

        template <
            typename Sequence
        >
        void operator()(Sequence const& seq) const
        {
            // TODO: verification?
            ar << seq;
        }
    };

    void request::save(serialization::output_archive& ar, const unsigned int) const
    { // {{{
        // TODO: versioning?
        int which = data->which();

        ar & which;
        ar & mc;
        boost::apply_visitor(save_visitor(ar), data->data);
    } // }}}

#define HPX_LOAD_SEQUENCE(z, n, _)                                            \
    case n:                                                                   \
        {                                                                     \
            boost::mpl::at_c<                                                 \
                request_data::data_type::types, n                             \
            >::type d;                                                        \
            ar >> d;                                                          \
            data->data = d;                                                   \
            return;                                                           \
        }                                                                     \
    /**/

    void request::load(serialization::input_archive& ar, const unsigned int)
    { // {{{
        // TODO: versioning
        int which = -1;

        ar & which;
        ar & mc;

        // Build the jump table.
        switch (which)
        {
            BOOST_PP_REPEAT(HPX_AGAS_REQUEST_SUBTYPES, HPX_LOAD_SEQUENCE, _)

            default: {
                HPX_THROW_EXCEPTION(invalid_data,
                    "request::load",
                    "unknown or invalid data loaded");
                return;
            }
        };
    } // }}}

#undef HPX_LOAD_SEQUENCE
}}

HPX_UTIL_REGISTER_FUNCTION(
    void(std::string const&, hpx::naming::gid_type const&)
  , hpx::util::function<void(std::string const&, hpx::naming::gid_type const&)>
  , request_iterate_names_function_type
)

HPX_UTIL_REGISTER_FUNCTION(
    void(std::string const&, hpx::components::component_type)
  , hpx::util::function<void(std::string const&, hpx::components::component_type)>
  , request_iterate_types_function_type
)

