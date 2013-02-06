//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2012-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_HWLOC)

#include <hpx/exception.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/policies/hwloc_topology.hpp>

#include <hwloc.h>

#include <boost/format.hpp>

// #define BOOST_SPIRIT_DEBUG
#define BOOST_SPIRIT_USE_PHOENIX_V3
#include <boost/spirit/include/qi_char.hpp>
#include <boost/spirit/include/qi_nonterminal.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/qi_operator.hpp>
#include <boost/spirit/include/qi_parse.hpp>
#include <boost/spirit/include/qi_string.hpp>
#include <boost/spirit/include/qi_auxiliary.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>

#include <hpx/runtime/threads/detail/partlit.hpp>

///////////////////////////////////////////////////////////////////////////////
BOOST_FUSION_ADAPT_STRUCT(
    hpx::threads::detail::spec_type,
    (hpx::threads::mask_type, index_min_)
    (hpx::threads::mask_type, index_max_)
    (hpx::threads::detail::spec_type::type, type_)
)

namespace hpx { namespace threads { namespace detail
{
    static char const* const type_names[] =
    {
        "unknown", "thread", "socket", "numanode", "core", "pu"
    };

    char const* const spec_type::type_name(spec_type::type t)
    {
        if (t < spec_type::unknown || t > spec_type::pu)
            return type_names[0];
        return type_names[t];
    }

    ///////////////////////////////////////////////////////////////////////////
    //
    //    mappings:
    //        mapping(;mapping)*
    //
    //    mapping:
    //        thread-spec=pu-specs
    //
    //    thread-spec:
    //        t:int
    //        t:int-int
    //        t:all
    //
    //    pu-specs:
    //        pu-spec(.pu-spec)*
    //
    //    pu-spec:
    //        type:int
    //        type:int-int
    //        type:all
    //        ~pu-spec
    //
    //    type:
    //        socket | numanode
    //        core
    //        pu
    //
    namespace qi = boost::spirit::qi;

    // parser for affinity options
    template <typename Iterator>
    struct mappings_parser : qi::grammar<Iterator, mappings_type()>
    {
        mappings_parser()
          : mappings_parser::base_type(start)
        {
            using detail::partlit;
            using detail::spec_type;

            start = mapping % ';';

            mapping =  thread_spec >> '=' >> pu_spec;

            thread_spec =
                    partlit("thread") >> ':'
                    >>  (   qi::uint_
                            >>  ('-' >> qi::uint_ | qi::attr(0ul))
                            >>  qi::attr(spec_type::thread)
                        |   partlit("all") >> qi::attr(~0x0ul) >> qi::attr(0ul)
                            >>  qi::attr(spec_type::thread)
                        )
                ;

            pu_spec =
                    socket_spec >> core_spec >> processing_unit_spec
//                 |   '~' >> pu_spec
                ;

            socket_spec =
                    partlit("socket") >> ':'
                    >>  (   qi::uint_
                            >>  ('-' >> qi::uint_ | qi::attr(0ul))
                            >>  qi::attr(spec_type::socket)
                        |   partlit("all") >> qi::attr(~0x0ul) >> qi::attr(0ul)
                            >>  qi::attr(spec_type::socket)
                        )
                |   partlit("numanode") >> ':'
                    >>  (   qi::uint_
                            >>  ('-' >> qi::uint_ | qi::attr(0ul))
                            >>  qi::attr(spec_type::numanode)
                        |   partlit("all") >> qi::attr(~0x0ul) >> qi::attr(0ul)
                            >>  qi::attr(spec_type::numanode)
                        )
                |   qi::attr(0ul) >> qi::attr(0ul) >> qi::attr(spec_type::unknown)
                ;

            core_spec =
                    -qi::lit('.') >> partlit("core") >> ':'
                    >>  (   qi::uint_
                            >>  ('-' >> qi::uint_ | qi::attr(0ul))
                            >>  qi::attr(spec_type::core)
                        |   partlit("all") >> qi::attr(~0x0ul) >> qi::attr(0ul)
                            >>  qi::attr(spec_type::core)
                        )
                |   qi::attr(0ul) >> qi::attr(0ul) >> qi::attr(spec_type::unknown)
                ;

            processing_unit_spec =
                    -qi::lit('.') >> partlit("pu") >> ':'
                    >>  (   qi::uint_
                            >>  ('-' >> qi::uint_ | qi::attr(0ul))
                            >>  qi::attr(spec_type::pu)
                        |   partlit("all") >> qi::attr(~0x0ul) >> qi::attr(0ul)
                            >>  qi::attr(spec_type::pu)
                        )
                |   qi::attr(0ul) >> qi::attr(0ul) >> qi::attr(spec_type::unknown)
                ;

            BOOST_SPIRIT_DEBUG_NODES(
                (start)(mapping)(thread_spec)(pu_spec)
                (socket_spec)(core_spec)(processing_unit_spec)
            );
        }

        qi::rule<Iterator, mappings_type()> start;
        qi::rule<Iterator, full_mapping_type()> mapping;
        qi::rule<Iterator, spec_type()> thread_spec;
        qi::rule<Iterator, mapping_type()> pu_spec;
        qi::rule<Iterator, spec_type()> socket_spec;
        qi::rule<Iterator, spec_type()> core_spec;
        qi::rule<Iterator, spec_type()> processing_unit_spec;
    };

    template <typename Iterator>
    inline bool parse(Iterator& begin, Iterator end, mappings_type& m)
    {
        mappings_parser<Iterator> p;
        return qi::parse(begin, end, p, m);
    }

    ///////////////////////////////////////////////////////////////////////////
    void parse_mappings(std::string const& spec, mappings_type& mappings,
        error_code& ec)
    {
        std::string::const_iterator begin = spec.begin();
        if (!detail::parse(begin, spec.end(), mappings) || begin != spec.end())
        {
            HPX_THROWS_IF(ec, bad_parameter, "parse_affinity_options",
                "failed to parse affinity specification: " + spec);
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();
    }

    ///////////////////////////////////////////////////////////////////////////
    std::pair<std::size_t, std::size_t> extract_bounds(spec_type const& m,
        std::size_t default_last, error_code& ec)
    {
        std::size_t first = m.index_min_;
        std::size_t last = m.index_max_;

        if (first == ~0x0ul) {
            if (last == ~0x0ul) {
                HPX_THROWS_IF(ec, bad_parameter, "decode_mappings",
                    boost::str(boost::format("invalid thread specification, "
                        "min: %x, max %x") % first % last));
                return std::make_pair(first, last);
            }

            // bind all entities
            first = 0;
            last = default_last-1;
        }
        if (0 == last)
            last = first;

        if (&ec != &throws)
            ec = make_success_code();

        return std::make_pair(first, last);
    }

    ///////////////////////////////////////////////////////////////////////////
    mask_type decode_mapping2_unknown(hwloc_topology const& t,
        mapping_type const& m, mask_type size, std::size_t mask,
        error_code& ec)
    {
        if (&ec != &throws)
            ec = make_success_code();
        return mask;
    }

    mask_type decode_mapping_pu(hwloc_topology const& t,
        mapping_type const& m, std::size_t size, mask_type mask,
        std::size_t pu_base_index, std::size_t thread_index, error_code& ec)
    {
        std::pair<std::size_t, std::size_t> b = extract_bounds(m[2], size, ec);
        if (ec) return 0;

        std::size_t index = std::size_t(-1);
        if (b.first != b.second)
            index = thread_index;

        mask_type pu_mask = 0;
        std::size_t pu_index = 0;
        for (std::size_t i = b.first; i <= b.second; ++i, ++pu_index)
        {
            if (index == std::size_t(-1) || pu_index == index)
                pu_mask |= t.init_thread_affinity_mask(i+pu_base_index);
        }

        return mask & pu_mask;
    }

    ///////////////////////////////////////////////////////////////////////////
    mask_type decode_mapping1_unknown(hwloc_topology const& t,
        mapping_type const& m, std::size_t size, mask_type mask,
        std::size_t pu_base_index, std::size_t thread_index, error_code& ec)
    {
        switch (m[2].type_) {
        case spec_type::pu:
            mask = decode_mapping_pu(t, m, size, mask, pu_base_index, thread_index, ec);
            break;

        case spec_type::unknown:
            mask = decode_mapping2_unknown(t, m, size, mask, ec);
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter, "decode_mapping1_unknown",
                boost::str(boost::format("unexpected specification type at "
                    "index two: %x (%s)") % m[1].type_ %
                        spec_type::type_name(m[1].type_)));
            break;
        }
        return mask;
    }

    mask_type decode_mapping_core(hwloc_topology const& t,
        mapping_type const& m, std::size_t size, mask_type mask,
        std::size_t core_base_index, std::size_t thread_index, error_code& ec)
    {
        std::pair<std::size_t, std::size_t> b = extract_bounds(m[1], size, ec);
        if (ec) return 0;

        // We have to account for the thread index at this level if there are
        // no specifications related to processing units.
        std::size_t index = std::size_t(-1);
        if (m[2].type_ == spec_type::unknown && b.first != b.second)
            index = thread_index;

        mask_type core_mask = 0;
        std::size_t core_index = 0;
        for (std::size_t i = b.first; i <= b.second; ++i, ++core_index)
        {
            if (index == std::size_t(-1) || core_index == index)
                core_mask |= t.init_core_affinity_mask_from_core(i+core_base_index, 0);
        }

        core_base_index += b.first;
        if (thread_index != std::size_t(-1) && b.first != b.second)
            core_base_index += thread_index;

        std::size_t base_index = 0;
        for (std::size_t i = 0; i != core_base_index; ++i)
            base_index += t.get_number_of_core_pus(i);

        return decode_mapping1_unknown(t, m, size, mask & core_mask,
            base_index, thread_index, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    mask_type decode_mapping0_unknown(hwloc_topology const& t,
        mapping_type const& m, std::size_t size, mask_type mask,
        std::size_t core_base_index, std::size_t thread_index, error_code& ec)
    {
        switch (m[1].type_) {
        case spec_type::core:
            mask = decode_mapping_core(t, m, size, mask, core_base_index,
                thread_index, ec);
            break;

        case spec_type::unknown:
            {
                std::size_t base_index = 0;
                for (std::size_t i = 0; i != core_base_index; ++i)
                    base_index += t.get_number_of_core_pus(i);

                mask = decode_mapping1_unknown(t, m, size, mask, base_index,
                    thread_index, ec);
            }
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter, "decode_mapping0_unknown",
                boost::str(boost::format("unexpected specification type at "
                    "index one: %x (%s)") % m[1].type_ %
                        spec_type::type_name(m[1].type_)));
            break;
        }
        return mask;
    }

    mask_type decode_mapping_socket(hwloc_topology const& t,
        mapping_type const& m, std::size_t size, std::size_t thread_index,
        error_code& ec)
    {
        std::pair<std::size_t, std::size_t> b = extract_bounds(m[0], size, ec);
        if (ec) return 0;

        std::size_t index = std::size_t(-1);
        if (m[1].type_ == spec_type::unknown &&
            m[2].type_ == spec_type::unknown &&
            b.first != b.second)
        {
            index = thread_index;
        }

        mask_type mask = 0;
        std::size_t socket_index = 0;
        for (std::size_t i = b.first; i <= b.second; ++i, ++socket_index)
        {
            if (index == std::size_t(-1) || socket_index == index)
                mask |= t.init_socket_affinity_mask_from_socket(i);
        }

        std::size_t socket_base_index = b.first;
        if (thread_index != std::size_t(-1) && b.first != b.second)
            socket_base_index += thread_index;

        std::size_t base_index = 0;
        for (std::size_t i = 0; i != socket_base_index; ++i)
            base_index += t.get_number_of_socket_cores(i);

        return decode_mapping0_unknown(t, m, size, mask, base_index,
            thread_index, ec);
    }

    mask_type decode_mapping_numanode(hwloc_topology const& t,
        mapping_type const& m, std::size_t size, std::size_t thread_index,
        error_code& ec)
    {
        std::pair<std::size_t, std::size_t> b = extract_bounds(m[0], size, ec);
        if (ec) return 0;

        std::size_t index = std::size_t(-1);
        if (m[1].type_ == spec_type::unknown &&
            m[2].type_ == spec_type::unknown &&
            b.first != b.second)
        {
            index = thread_index;
        }

        mask_type mask = 0;
        std::size_t node_index = 0;
        for (std::size_t i = b.first; i <= b.second; ++i, ++node_index)
        {
            if (index == std::size_t(-1) || node_index == index)
                mask |= t.init_numa_node_affinity_mask_from_numa_node(i);
        }

        std::size_t node_base_index = b.first;
        if (thread_index != std::size_t(-1) && b.first != b.second)
            node_base_index += thread_index;

        std::size_t base_index = 0;
        for (std::size_t i = 0; i != node_base_index; ++i)
            base_index += t.get_number_of_numa_node_cores(i);

        return decode_mapping0_unknown(t, m, size, mask, base_index,
            thread_index, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    mask_type decode_mapping(hwloc_topology const& t,
        mapping_type const& m, std::vector<mask_type>& affinities,
        std::size_t thread_index, error_code& ec)
    {
        std::size_t size = affinities.size();
        mask_type mask = 0;
        switch (m[0].type_) {
        case spec_type::socket:
            // requested top level is a socket
            mask = decode_mapping_socket(t, m, size, thread_index, ec);
            break;

        case spec_type::numanode:
            // requested top level is a NUMA node
            mask = decode_mapping_numanode(t, m, size, thread_index, ec);
            break;

        case spec_type::unknown:
            // no top level is requested
            mask = decode_mapping0_unknown(t, m, size,
                t.get_machine_affinity_mask(), 0, thread_index, ec);
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter, "decode_mapping",
                boost::str(boost::format("unexpected specification type at "
                    "index zero: %x (%s)") % m[0].type_ %
                        spec_type::type_name(m[0].type_)));
            return 0;
        }
        return mask;
    }

    ///////////////////////////////////////////////////////////////////////////
    // sanity checks
    void mappings_sanity_checks(mapping_type const& m, std::size_t size, 
        std::pair<std::size_t, std::size_t> const& b, error_code& ec)
    {
        if (m.size() != 3) {
            HPX_THROWS_IF(ec, bad_parameter, "decode_mapping",
                "bad size of mappings specification array");
            return;
        }

        std::size_t count_ranges = 0;
        for (std::size_t i = 0; i != 3; ++i)
        {
            std::pair<std::size_t, std::size_t> bounds =
                extract_bounds(m[i], size, ec);
            if (ec) return;

            if (bounds.first != bounds.second) {
                ++count_ranges;
// FIXME: replace this with proper counting of processing units specified by 
//        the affinity desc
//                 if (b.first != b.second) {
//                     // threads have bounds ranges as well
//                     if (b.second - b.first > bounds.second - bounds.first) {
//                         HPX_THROWS_IF(ec, bad_parameter, "decode_mapping",
//                             boost::str(boost::format("the thread index range "
//                                 "is larger than the index range specified for "
//                                 "the %s node") % spec_type::type_name(
//                                     m[i].type_)));
//                         return;
//                     }
//                 }
            }
        }
        if (count_ranges > 1) {
            HPX_THROWS_IF(ec, bad_parameter, "decode_mapping",
                "index ranges can be specified only for one node type "
                "(socket/numanode, core, or pu)");
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();
    }

    void decode_mappings(full_mapping_type const& m,
        std::vector<mask_type>& affinities, error_code& ec)
    {
        // We need to instantiate a new topology object as the runtime has not
        // been initialized yet
        hwloc_topology t;

        // repeat for each of the threads in the affinity specification
        std::size_t size = affinities.size();
        std::pair<std::size_t, std::size_t> b = extract_bounds(m.first, size, ec);
        if (ec) return;

        mappings_sanity_checks(m.second, size, b, ec);
        if (ec) return;

        // we need to keep track of the thread index if the bounds are different
        // (i.e. we need to bind more than one thread)
        std::size_t index = (b.first != b.second) ? 0 : std::size_t(-1);
        for (std::size_t i = b.first; i <= b.second; ++i)
        {
            mask_type mask = decode_mapping(t, m.second, affinities, index, ec);
            if (ec) return;

            // set each thread affinity only once
            if (0 != affinities[i])
            {
                HPX_THROWS_IF(ec, bad_parameter, "decode_mapping",
                    boost::str(boost::format("affinity mask for thread %1% has "
                        "already been set") % i));
                return;
            }

            // set result
            affinities[i] = mask;

            if (index != std::size_t(-1))
                ++index;
        }
    }
}}}

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities, error_code& ec)
    {
        detail::mappings_type mappings;
        detail::parse_mappings(spec, mappings, ec);
        if (!ec) {
            BOOST_FOREACH(detail::full_mapping_type const& m, mappings)
            {
                detail::decode_mappings(m, affinities, ec);
                if (ec) return;
            }
        }
    }
}}

#endif
