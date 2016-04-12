//  Copyright (c) 2007-2013 Hartmut Kaiser
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
#include <hpx/runtime.hpp>

#include <sstream>
#include <string>

///////////////////////////////////////////////////////////////////////////////
BOOST_FUSION_ADAPT_STRUCT(
    hpx::threads::detail::spec_type,
    (hpx::threads::detail::bounds_type, index_bounds_)
    (hpx::threads::detail::spec_type::type, type_)
)

namespace hpx { namespace threads { namespace detail
{
    static char const* const type_names[] =
    {
        "unknown", "thread", "socket", "numanode", "core", "pu"
    };

    char const* spec_type::type_name(spec_type::type t)
    {
        if (t < spec_type::unknown || t > spec_type::pu)
            return type_names[0];
        return type_names[t];
    }

    ///////////////////////////////////////////////////////////////////////////
    //
    //    mappings:
    //        distribution
    //        mapping(;mapping)*
    //
    //    distribution:
    //        compact
    //        scatter
    //        balanced
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
    //        type:range-specs
    //        ~pu-spec
    //
    //    range-specs:
    //        range-spec(,range-spec)*
    //
    //    range-spec:
    //        int
    //        int-int
    //        all
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

            start = distribution | (mapping % ';');

            mapping =  thread_spec >> '=' >> pu_spec;

            distribution =
                    partlit("compact") >> qi::attr(compact)
                |   partlit("scatter") >> qi::attr(scatter)
                |   partlit("balanced") >> qi::attr(balanced)
                ;

            thread_spec =
                    partlit("thread") >> ':'
                    >>  specs >> qi::attr(spec_type::thread)
                ;

            pu_spec =
                    socket_spec >> core_spec >> processing_unit_spec
//                 |   '~' >> pu_spec
                ;

            socket_spec =
                    partlit("socket") >> ':'
                    >> specs >> qi::attr(spec_type::socket)
                |   partlit("numanode") >> ':'
                    >> specs >> qi::attr(spec_type::numanode)
                |   qi::attr(spec_type::unknown)
                ;

            core_spec =
                   -qi::lit('.') >> partlit("core") >> ':'
                    >> specs >> qi::attr(spec_type::core)
                |   qi::attr(spec_type::unknown)
                ;

            processing_unit_spec =
                   -qi::lit('.') >> partlit("pu") >> ':'
                    >> specs >> qi::attr(spec_type::pu)
                |   qi::attr(spec_type::unknown)
                ;

            specs = spec % ',';

            spec =
                    qi::uint_ >> -(qi::int_)
                |   partlit("all") >> qi::attr(spec_type::all_entities())
                ;

            BOOST_SPIRIT_DEBUG_NODES(
                (start)(mapping)(distribution)(thread_spec)(pu_spec)(specs)
                (spec)(socket_spec)(core_spec)(processing_unit_spec)
            );
        }

        qi::rule<Iterator, mappings_type()> start;
        qi::rule<Iterator, distribution_type()> distribution;
        qi::rule<Iterator, full_mapping_type()> mapping;
        qi::rule<Iterator, spec_type()> thread_spec;
        qi::rule<Iterator, mapping_type()> pu_spec;
        qi::rule<Iterator, spec_type()> socket_spec;
        qi::rule<Iterator, spec_type()> core_spec;
        qi::rule<Iterator, bounds_type()> specs;
        qi::rule<Iterator, bounds_type()> spec;
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
    bounds_type
    extract_bounds(spec_type& m, std::size_t default_last, error_code& ec)
    {
        bounds_type result;

        if (m.index_bounds_.empty())
            return result;

        bounds_type::const_iterator first = m.index_bounds_.begin();
        bounds_type::const_iterator last = m.index_bounds_.end();

        while (first != last) {
            if (*first == spec_type::all_entities()) {
                // bind all entities
                for (std::size_t i = 0; i < default_last; ++i)
                    result.push_back(i);
                break;    // we will not get more than 'all'
            }

            bounds_type::const_iterator second = first;
            if (++second != last) {
                if (*second == 0 || *second == spec_type::all_entities()) {
                    // one element only
                    result.push_back(*first);
                }
                else if (*second < 0) {
                    // all elements between min and -max
                    for (boost::int64_t i = *first; i <= -*second; ++i)
                        result.push_back(i);
                }
                else {
                    // just min and max
                    result.push_back(*first);
                    result.push_back(*second);
                }
                first = second;
            }
            else {
                // one element only
                result.push_back(*first);
            }
            ++first;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    mask_type decode_mapping2_unknown(hwloc_topology const& t,
        mapping_type const& m, std::size_t size, mask_type mask,
        error_code& ec)
    {
        if (&ec != &throws)
            ec = make_success_code();
        return mask;
    }

    mask_type decode_mapping_pu(hwloc_topology const& t,
        mapping_type& m, std::size_t size, mask_type mask,
        std::size_t pu_base_index, std::size_t thread_index, error_code& ec)
    {
        bounds_type b = extract_bounds(m[2], size, ec);
        if (ec) return mask_type();

        std::size_t index = std::size_t(-1);
        if (b.size() > 1)
            index = thread_index;

        mask_type pu_mask = mask_type();
        resize(pu_mask, size);

        std::size_t pu_index = 0;
        for (bounds_type::const_iterator it = b.begin(); it != b.end(); ++it, ++pu_index)
        {
            if (index == std::size_t(-1) || pu_index == index)
                pu_mask |= t.init_thread_affinity_mask(std::size_t(*it+pu_base_index));
        }

        return mask & pu_mask;
    }

    ///////////////////////////////////////////////////////////////////////////
    mask_type decode_mapping1_unknown(hwloc_topology const& t,
        mapping_type& m, std::size_t size, mask_type mask,
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
                    "index two: %x (%s)") %
                        static_cast<unsigned>(m[1].type_) %
                        spec_type::type_name(m[1].type_)));
            break;
        }
        return mask;
    }

    mask_type decode_mapping_core(hwloc_topology const& t,
        mapping_type& m, std::size_t size, mask_type mask,
        std::size_t core_base_index, std::size_t thread_index, error_code& ec)
    {
        bounds_type b = extract_bounds(m[1], size, ec);
        if (ec) return mask_type();

        // We have to account for the thread index at this level if there are
        // no specifications related to processing units.
        std::size_t index = std::size_t(-1);
        if (m[2].type_ == spec_type::unknown && b.size() > 1)
            index = thread_index;

        mask_type core_mask = mask_type();
        resize(core_mask, size);

        std::size_t core_index = 0;
        for (bounds_type::const_iterator it = b.begin(); it != b.end();
            ++it, ++core_index)
        {
            if (index == std::size_t(-1) || core_index == index)
            {
                core_mask |= t.init_core_affinity_mask_from_core(
                    std::size_t(*it+core_base_index), mask_type());
            }
        }

        core_base_index += std::size_t(*b.begin());
        if (thread_index != std::size_t(-1) && b.size() > 1)
            core_base_index += thread_index;

        std::size_t base_index = 0;
        for (std::size_t i = 0; i != core_base_index; ++i)
            base_index += t.get_number_of_core_pus(i);

        return decode_mapping1_unknown(t, m, size, mask & core_mask,
            base_index, thread_index, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    mask_type decode_mapping0_unknown(hwloc_topology const& t,
        mapping_type& m, std::size_t size, mask_type mask,
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
                    "index one: %x (%s)") %
                        static_cast<unsigned>(m[1].type_) %
                        spec_type::type_name(m[1].type_)));
            break;
        }
        return mask;
    }

    mask_type decode_mapping_socket(hwloc_topology const& t,
        mapping_type& m, std::size_t size, std::size_t thread_index,
        error_code& ec)
    {
        bounds_type b = extract_bounds(m[0], size, ec);
        if (ec) return mask_type();

        std::size_t index = std::size_t(-1);
        if (m[1].type_ == spec_type::unknown &&
            m[2].type_ == spec_type::unknown &&
            b.size() > 1)
        {
            index = thread_index;
        }

        mask_type mask = mask_type();
        resize(mask, size);

        std::size_t socket_index = 0;
        for (bounds_type::const_iterator it = b.begin(); it != b.end();
            ++it, ++socket_index)
        {
            if (index == std::size_t(-1) || socket_index == index)
                mask |= t.init_socket_affinity_mask_from_socket(std::size_t(*it));
        }

        std::size_t socket_base_index = std::size_t(*b.begin());
        if (thread_index != std::size_t(-1) && b.size() > 1)
            socket_base_index += thread_index;

        std::size_t base_index = 0;
        for (std::size_t i = 0; i != socket_base_index; ++i)
            base_index += t.get_number_of_socket_cores(i);

        return decode_mapping0_unknown(t, m, size, mask, base_index,
            thread_index, ec);
    }

    mask_type decode_mapping_numanode(hwloc_topology const& t,
        mapping_type& m, std::size_t size, std::size_t thread_index,
        error_code& ec)
    {
        bounds_type b = extract_bounds(m[0], size, ec);
        if (ec) return mask_type();

        std::size_t index = std::size_t(-1);
        if (m[1].type_ == spec_type::unknown &&
            m[2].type_ == spec_type::unknown &&
            b.size() > 1)
        {
            index = thread_index;
        }

        mask_type mask = mask_type();
        resize(mask, size);

        std::size_t node_index = 0;
        for (bounds_type::const_iterator it = b.begin(); it != b.end();
            ++it, ++node_index)
        {
            if (index == std::size_t(-1) || node_index == index)
                mask |= t.init_numa_node_affinity_mask_from_numa_node(std::size_t(*it));
        }

        std::size_t node_base_index = std::size_t(*b.begin());
        if (thread_index != std::size_t(-1) && b.size() > 1)
            node_base_index += thread_index;

        std::size_t base_index = 0;
        for (std::size_t i = 0; i != node_base_index; ++i)
            base_index += t.get_number_of_numa_node_cores(i);

        return decode_mapping0_unknown(t, m, size, mask, base_index,
            thread_index, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    mask_type decode_mapping(hwloc_topology const& t,
        mapping_type& m, std::vector<mask_type>& affinities,
        std::size_t thread_index, error_code& ec)
    {
        std::size_t size = affinities.size();
        mask_type mask;
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
                    "index zero: %x (%s)") %
                        static_cast<unsigned>(m[0].type_) %
                        spec_type::type_name(m[0].type_)));
            return mask_type();
        }
        return mask;
    }

    ///////////////////////////////////////////////////////////////////////////
    // sanity checks
    void mappings_sanity_checks(mapping_type& m, std::size_t size,
        bounds_type const& b, error_code& ec)
    {
        if (m.size() != 3) {
            HPX_THROWS_IF(ec, bad_parameter, "decode_mapping",
                "bad size of mappings specification array");
            return;
        }

        std::size_t count_ranges = 0;
        for (std::size_t i = 0; i != 3; ++i)
        {
            bounds_type bounds = extract_bounds(m[i], size, ec);
            if (ec) return;

            if (bounds.size() > 1) {
                ++count_ranges;
// FIXME: replace this with proper counting of processing units specified by
//        the affinity desc
//                 if (b.begin() != b.end()) {
//                     // threads have bounds ranges as well
//                     if (b.end() - b.begin() > bounds.second - bounds.first) {
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

    void decode_mappings(hwloc_topology const& t, full_mapping_type& m,
        std::vector<mask_type>& affinities, error_code& ec)
    {
        // repeat for each of the threads in the affinity specification
        std::size_t size = affinities.size();
        bounds_type b = extract_bounds(m.first, size, ec);
        if (ec) return;

        mappings_sanity_checks(m.second, size, b, ec);
        if (ec) return;

        // we need to keep track of the thread index if the bounds are different
        // (i.e. we need to bind more than one thread)
        std::size_t index = (b.begin() != b.end()) ? 0 : std::size_t(-1);
        for (bounds_type::const_iterator it = b.begin(); it != b.end(); ++it)
        {
            mask_type mask = decode_mapping(t, m.second, affinities, index, ec);
            if (ec) return;

            // set each thread affinity only once
            HPX_ASSERT(std::size_t(*it) < affinities.size());
            if (any(affinities[std::size_t(*it)]))
            {
                HPX_THROWS_IF(ec, bad_parameter, "decode_mapping",
                    boost::str(boost::format("affinity mask for thread %1% has "
                        "already been set") % *it));
                return;
            }

            // set result
            affinities[std::size_t(*it)] = mask;

            if (index != std::size_t(-1))
                ++index;
        }
    }

    void decode_compact_distribution(hwloc_topology& t,
        std::vector<mask_type>& affinities,
        std::size_t used_cores, std::size_t max_cores,
        std::vector<std::size_t>& num_pus, error_code& ec)
    {
        std::size_t num_threads = affinities.size();
        std::size_t num_cores = (std::min)(max_cores, t.get_number_of_cores());
        num_pus.resize(num_threads);

        for (std::size_t num_thread = 0; num_thread != num_threads; /**/)
        {
            for(std::size_t num_core = 0; num_core != num_cores; ++num_core)
            {
                std::size_t num_core_pus
                    = t.get_number_of_core_pus(num_core + used_cores);
                for(std::size_t num_pu = 0; num_pu != num_core_pus; ++num_pu)
                {
                    if (any(affinities[num_thread]))
                    {
                        HPX_THROWS_IF(ec, bad_parameter, "decode_compact_distribution",
                            boost::str(boost::format("affinity mask for thread %1% has "
                                "already been set") % num_thread));
                        return;
                    }
                    affinities[num_thread]
                        = t.init_thread_affinity_mask(num_core + used_cores, num_pu);
                    num_pus[num_thread] = num_thread;

                    if(++num_thread == num_threads)
                        return;
                }
            }
        }
    }

    void decode_scatter_distribution(hwloc_topology& t,
        std::vector<mask_type>& affinities,
        std::size_t used_cores, std::size_t max_cores,
        std::vector<std::size_t>& num_pus, error_code& ec)
    {
        std::size_t num_threads = affinities.size();
        std::size_t num_cores = (std::min)(max_cores, t.get_number_of_cores());

        std::vector<std::size_t> num_pus_cores(num_cores, 0);
        num_pus.resize(num_threads);

        for (std::size_t num_thread = 0; num_thread != num_threads; /**/)
        {
            for(std::size_t num_core = 0; num_core != num_cores; ++num_core)
            {
                if (any(affinities[num_thread]))
                {
                    HPX_THROWS_IF(ec, bad_parameter, "decode_scatter_distribution",
                        boost::str(boost::format("affinity mask for thread %1% has "
                            "already been set") % num_thread));
                    return;
                }

                num_pus[num_thread] = t.get_pu_number(num_core + used_cores,
                    num_pus_cores[num_core]);
                affinities[num_thread] = t.init_thread_affinity_mask(
                    num_core + used_cores, num_pus_cores[num_core]++);

                if(++num_thread == num_threads)
                    return;
            }
        }
    }

    void decode_balanced_distribution(hwloc_topology& t,
        std::vector<mask_type>& affinities,
        std::size_t used_cores, std::size_t max_cores,
        std::vector<std::size_t>& num_pus, error_code& ec)
    {
        std::size_t num_threads = affinities.size();
        std::size_t num_cores = (std::min)(max_cores, t.get_number_of_cores());

        std::vector<std::size_t> num_pus_cores(num_cores, 0);
        num_pus.resize(num_threads);
        // At first, calculate the number of used pus per core.
        // This needs to be done to make sure that we occupy all the available cores
        for (std::size_t num_thread = 0; num_thread != num_threads; /**/)
        {
            for(std::size_t num_core = 0; num_core != num_cores; ++num_core)
            {
                num_pus_cores[num_core]++;
                if(++num_thread == num_threads)
                    break;
            }
        }

        // Iterate over the cores and assigned pus per core. this additional loop
        // is needed so that we have consecutive worker thread numbers
        std::size_t num_thread = 0;
        for(std::size_t num_core = 0; num_core != num_cores; ++num_core)
        {
            for(std::size_t num_pu = 0; num_pu != num_pus_cores[num_core]; ++num_pu)
            {
                if (any(affinities[num_thread]))
                {
                    HPX_THROWS_IF(ec, bad_parameter, "decode_balanced_distribution",
                        boost::str(boost::format("affinity mask for thread %1% has "
                            "already been set") % num_thread));
                    return;
                }
                num_pus[num_thread] = t.get_pu_number(num_core + used_cores, num_pu);
                affinities[num_thread] = t.init_thread_affinity_mask(
                    num_core + used_cores, num_pu);
                ++num_thread;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void decode_distribution(distribution_type d, hwloc_topology& t,
        std::vector<mask_type>& affinities,
        std::size_t used_cores, std::size_t max_cores,
        std::vector<std::size_t>& num_pus, error_code& ec)
    {
        switch (d) {
        case compact:
            decode_compact_distribution(t, affinities, used_cores, max_cores,
                num_pus, ec);
            break;

        case scatter:
            decode_scatter_distribution(t, affinities, used_cores, max_cores,
                num_pus, ec);
            break;

        case balanced:
            decode_balanced_distribution(t, affinities, used_cores, max_cores,
                num_pus, ec);
            break;

        default:
            HPX_ASSERT(false);
        }
    }
}}}

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities,
        std::size_t used_cores, std::size_t max_cores,
        std::vector<std::size_t>& num_pus, error_code& ec)
    {
        detail::mappings_type mappings;
        detail::parse_mappings(spec, mappings, ec);
        if (ec) return;

        // We need to instantiate a new topology object as the runtime has not
        // been initialized yet
        threads::hwloc_topology& t = threads::create_topology();

        switch (mappings.which())
        {
        case 0:
            {
                detail::decode_distribution(
                    boost::get<detail::distribution_type>(mappings),
                    t, affinities, used_cores, max_cores, num_pus, ec);
                if (ec) return;
            }
            break;
        case 1:
            {
//                    std::cout << "got mappings " << spec << " ...\n";
                detail::mappings_spec_type mappings_specs(
                    boost::get<detail::mappings_spec_type>(mappings));

                for (detail::full_mapping_type& m : mappings_specs)
                {
                    if (m.first.type_ != detail::spec_type::thread)
                    {
                        HPX_THROWS_IF(ec, bad_parameter, "parse_affinity_options",
                            boost::str(boost::format("bind specification (%1%) is "
                                "ill formatted") % spec));
                        return;
                    }

                    if (m.second.size() != 3)
                    {
                        HPX_THROWS_IF(ec, bad_parameter, "parse_affinity_options",
                            boost::str(boost::format("bind specification (%1%) is "
                                "ill formatted") % spec));
                        return;
                    }

                    if (m.second[0].type_ == detail::spec_type::unknown &&
                        m.second[1].type_ == detail::spec_type::unknown &&
                        m.second[2].type_ == detail::spec_type::unknown)
                    {
                        HPX_THROWS_IF(ec, bad_parameter, "parse_affinity_options",
                            boost::str(boost::format("bind specification (%1%) is "
                                "ill formatted") % spec));
                        return;
                    }

                    detail::decode_mappings(t, m, affinities, ec);
                    if (ec) return;
                }

                if(num_pus.empty())
                {
                    num_pus.resize(affinities.size());
                    for (std::size_t i = 0; i != affinities.size(); ++i)
                    {
                        num_pus[i] = threads::find_first(affinities[i]);
                    }
                }
            }
            break;
        }
    }
}}

#endif
