//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/performance_counters/registry.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/base_lco_factory.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/util/function.hpp>

#include <hpx/util/bind.hpp>
#include <hpx/util/safe_lexical_cast.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#define BOOST_SPIRIT_USE_PHOENIX_V3
#include <boost/spirit/include/qi_char.hpp>
#include <boost/spirit/include/qi_nonterminal.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/qi_operator.hpp>
#include <boost/spirit/include/qi_parse.hpp>
#include <boost/spirit/include/qi_string.hpp>
#include <boost/spirit/include/qi_auxiliary.hpp>
#include <boost/spirit/include/qi_directive.hpp>
#include <boost/fusion/include/adapt_struct.hpp>

///////////////////////////////////////////////////////////////////////////////
// Initialization support for the performance counter actions
HPX_REGISTER_ACTION_ID(
    hpx::performance_counters::server
    ::base_performance_counter::get_counter_info_action,
    performance_counter_get_counter_info_action,
    hpx::actions::performance_counter_get_counter_info_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::performance_counters::server
    ::base_performance_counter::get_counter_value_action,
    performance_counter_get_counter_value_action,
    hpx::actions::performance_counter_get_counter_value_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::performance_counters::server
    ::base_performance_counter::set_counter_value_action,
    performance_counter_set_counter_value_action,
    hpx::actions::performance_counter_set_counter_value_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::performance_counters::server
    ::base_performance_counter::reset_counter_value_action,
    performance_counter_reset_counter_value_action,
    hpx::actions::performance_counter_reset_counter_value_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::performance_counters::server::base_performance_counter::start_action,
    performance_counter_start_action,
    hpx::actions::performance_counter_start_action_id)
HPX_REGISTER_ACTION_ID(
    hpx::performance_counters::server::base_performance_counter::stop_action,
    performance_counter_stop_action,
    hpx::actions::performance_counter_stop_action_id)

HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    hpx::performance_counters::counter_info, hpx_counter_info,
    hpx::actions::base_lco_with_value_hpx_counter_info_get,
    hpx::actions::base_lco_with_value_hpx_counter_info_set)
HPX_REGISTER_BASE_LCO_WITH_VALUE_ID(
    hpx::performance_counters::counter_value, hpx_counter_value,
    hpx::actions::base_lco_with_value_hpx_counter_value_get,
    hpx::actions::base_lco_with_value_hpx_counter_value_set)

HPX_DEFINE_GET_COMPONENT_TYPE(
    hpx::performance_counters::server::base_performance_counter)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    /// \brief Create a full name of a counter from the contents of the given
    ///        \a counter_path_elements instance.
    counter_status get_counter_name(counter_path_elements const& path,
        std::string& result, error_code& ec)
    {
        if (path.objectname_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_name",
                "empty counter object name");
            return status_invalid_data;
        }

        result = "/";
        result += path.objectname_;

        if (!path.parentinstancename_.empty() || !path.instancename_.empty())
        {
            result += "{";
            if (!path.parentinstancename_.empty())
            {
                result += path.parentinstancename_;
                if (-1 != path.parentinstanceindex_)
                {
                    result += "#";
                    result += boost::lexical_cast<std::string>
                                (path.parentinstanceindex_);
                }
                if (!path.instancename_.empty())
                    result += "/";
            }
            if (!path.instancename_.empty()) {
                result += path.instancename_;
                if (-1 != path.instanceindex_)
                {
                    result += "#";
                    result += boost::lexical_cast<std::string>(path.instanceindex_);
                }
            }
            result += "}";
        }
        if (!path.countername_.empty()) {
            result += "/";
            result += path.countername_;
        }

        if (!path.parameters_.empty()) {
            result += "@";
            result += path.parameters_;
        }

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    /// \brief Create a full name of a counter from the contents of the given
    ///        \a counter_path_elements instance.
    counter_status get_counter_type_name(counter_type_path_elements const& path,
        std::string& result, error_code& ec)
    {
        if (path.objectname_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_type_name",
                "empty counter object name");
            return status_invalid_data;
        }

        result = "/";
        result += path.objectname_;

        if (!path.countername_.empty()) {
            result += "/";
            result += path.countername_;
        }

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    /// \brief Create a full name of a counter from the contents of the given
    ///        \a counter_path_elements instance.
    counter_status get_full_counter_type_name(
        counter_type_path_elements const& path, std::string& result,
        error_code& ec)
    {
        if (path.objectname_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter, "get_full_counter_type_name",
                "empty counter object name");
            return status_invalid_data;
        }

        result = "/";
        result += path.objectname_;

        if (!path.countername_.empty()) {
            result += "/";
            result += path.countername_;
        }

        if (!path.parameters_.empty()) {
            result += "@";
            result += path.parameters_;
        }

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    /// \brief Create a name of a counter instance from the contents of the
    ///        given \a counter_path_elements instance.
    counter_status get_counter_instance_name(
        counter_path_elements const& path, std::string& result,
        error_code& ec)
    {
        if (path.parentinstancename_.empty()) {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_instance_name",
                "empty counter instance name");
            return status_invalid_data;
        }

        if (path.parentinstance_is_basename_) {
            result = path.parentinstancename_;
        }
        else {
            result = "/";
            result += path.parentinstancename_;

            if (!path.instancename_.empty()) {
                result += "/";
                result += path.instancename_;
            }
        }

        if (&ec != &throws)
            ec = make_success_code();
        return status_valid_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    struct instance_name
    {
        instance_name() : basename_(false) {}

        std::string name_;
        std::string index_;
        bool basename_;
    };

    struct instance_elements
    {
        instance_name parent_;
        instance_name child_;
    };

    struct path_elements
    {
        std::string object_;
        instance_elements instance_;
        std::string counter_;
        std::string parameters_;
    };
}}

BOOST_FUSION_ADAPT_STRUCT(
    hpx::performance_counters::instance_name,
    (std::string, name_)
    (std::string, index_)
    (bool, basename_)
)

BOOST_FUSION_ADAPT_STRUCT(
    hpx::performance_counters::instance_elements,
    (hpx::performance_counters::instance_name, parent_)
    (hpx::performance_counters::instance_name, child_)
)

BOOST_FUSION_ADAPT_STRUCT(
    hpx::performance_counters::path_elements,
    (std::string, object_)
    (hpx::performance_counters::instance_elements, instance_)
    (std::string, counter_)
    (std::string, parameters_)
)

namespace hpx { namespace performance_counters
{
    ///
    ///    /objectname{parentinstancename#parentindex/instancename#instanceindex}
    ///       /countername#parameters
    ///    /objectname{parentinstancename#*/instancename#*}/countername#parameters
    ///    /objectname{/basecounter}/countername,parameters
    ///
    namespace qi = boost::spirit::qi;

    template <typename Iterator>
    struct path_parser : qi::grammar<Iterator, path_elements()>
    {
        path_parser()
          : path_parser::base_type(start)
        {
            start = -qi::lit(counter_prefix)
                >> '/' >> +~qi::char_("/{#@") >> -instance
                >> -('/' >>  +~qi::char_("#}@")) >> -('@' >> +qi::char_);
            instance = '{' >> parent >> -('/' >> child) >> '}';
            parent =
                    &qi::lit('/') >> qi::raw[start] >> qi::attr(-1) >> qi::attr(true)
                    // base counter
                |  +~qi::char_("#/}")
                    >>  (   '#' >> raw_uint     // counter parentinstance name
                        |  -qi::string("#*")    // counter parentinstance skeleton name
                        )
                    >> qi::attr(false)
                ;
            child =
                   +~qi::char_("#}")
                    >>  (   '#' >> raw_uint     // counter instance name
                        |  -qi::string("#*")    // counter instance skeleton name
                        )
                    >> qi::attr(false)
                ;
            raw_uint = qi::raw[qi::uint_];
        }

        qi::rule<Iterator, path_elements()> start;
        qi::rule<Iterator, instance_elements()> instance;
        qi::rule<Iterator, instance_name()> parent;
        qi::rule<Iterator, instance_name()> child;
        qi::rule<Iterator, std::string()> raw_uint;
    };

    /// \brief Fill the given \a counter_path_elements instance from the given
    ///        full name of a counter
    ///
    ///    /objectname{parentinstancename#parentindex
    ///       /instancename#instanceindex}/countername
    ///
    counter_status get_counter_path_elements(std::string const& name,
        counter_path_elements& path, error_code& ec)
    {
        path_elements elements;
        path_parser<std::string::const_iterator> p;

        std::string::const_iterator begin = name.begin();
        if (!qi::parse(begin, name.end(), p, elements) || begin != name.end())
        {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_path_elements",
                boost::str(boost::format("invalid counter name format: %s") % name));
            return status_invalid_data;
        }

        path.objectname_ = elements.object_;
        path.countername_ = elements.counter_;

        path.parentinstancename_ = elements.instance_.parent_.name_;
        path.parentinstance_is_basename_ = elements.instance_.parent_.basename_;
        path.parentinstanceindex_ = -1;

        path.instancename_ = elements.instance_.child_.name_;
        path.instanceindex_ = -1;

        path.parameters_ = elements.parameters_;

        if (!path.parentinstance_is_basename_) {
            if (elements.instance_.parent_.index_ == "#*") {
                path.parentinstancename_ += "#*";
            }
            else if (!elements.instance_.parent_.index_.empty()) {
                path.parentinstanceindex_ =
                    hpx::util::safe_lexical_cast<boost::uint64_t>
                    (elements.instance_.parent_.index_);
            }

            if (elements.instance_.child_.index_ == "#*") {
                path.instancename_ += "#*";
            }
            else if (!elements.instance_.child_.index_.empty()) {
                path.instanceindex_ =
                    hpx::util::safe_lexical_cast<boost::uint64_t>
                    (elements.instance_.child_.index_);
            }
        }

        if (&ec != &throws)
            ec = make_success_code();

        return status_valid_data;
    }

    /// \brief Fill the given \a counter_type_path_elements instance from the
    ///        given full name of a counter
    ///
    ///    /objectname{...}/countername
    ///    /objectname
    ///
    counter_status get_counter_type_path_elements(std::string const& name,
        counter_type_path_elements& path, error_code& ec)
    {
        path_elements elements;
        path_parser<std::string::const_iterator> p;

        // parse the full name
        std::string::const_iterator begin = name.begin();
        if (!qi::parse(begin, name.end(), p, elements) || begin != name.end())
        {
            HPX_THROWS_IF(ec, bad_parameter, "get_counter_type_path_elements",
                boost::str(boost::format("invalid counter name format: %s") % name));
            return status_invalid_data;
        }

        // but extract only counter type elements
        path.objectname_ = elements.object_;
        path.countername_ = elements.counter_;
        path.parameters_ = elements.parameters_;

        if (&ec != &throws)
            ec = make_success_code();

        return status_valid_data;
    }

    /// \brief Return the counter type name from a given full instance name
    counter_status get_counter_type_name(std::string const& name,
        std::string& type_name, error_code& ec)
    {
        counter_type_path_elements p;

        counter_status status = get_counter_type_path_elements(name, p, ec);
        if (!status_is_valid(status)) return status;

        return get_counter_type_name(p, type_name, ec);
    }

    /// \brief Return the counter type name from a given full instance name
    counter_status get_counter_name(std::string const& name,
        std::string& countername, error_code& ec)
    {
        counter_path_elements p;

        counter_status status = get_counter_path_elements(name, p, ec);
        if (!status_is_valid(status)) return status;

        return get_counter_name(p, countername, ec);
    }

    /// \brief Complement the counter info if parent instance name is missing
    counter_status complement_counter_info(counter_info& info,
        counter_info const& type_info, error_code& ec)
    {
        info.type_ = type_info.type_;
        if (info.helptext_.empty())
            info.helptext_ = type_info.helptext_;
        return complement_counter_info(info, ec);
    }

    counter_status complement_counter_info(counter_info& info, error_code& ec)
    {
        counter_path_elements p;

        counter_status status = get_counter_path_elements(info.fullname_, p, ec);
        if (!status_is_valid(status)) return status;

        if (p.parentinstancename_.empty()) {
            p.parentinstancename_ = "locality";
            p.parentinstanceindex_ = static_cast<boost::int64_t>(get_locality_id());
            if (p.instancename_.empty())
            {
                p.instancename_ = "total";
                p.instanceindex_ = -1;
            }
        }

        // fill with complete counter type info
        std::string type_name;
        get_counter_type_name(p, type_name, ec);
        if (!status_is_valid(status)) return status;

        get_counter_type(type_name, info, ec);
        if (!status_is_valid(status)) return status;

        // last, set full counter name
        return get_counter_name(p, info.fullname_, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Get the name for a given counter type
    namespace strings
    {
        char const* const counter_type_names[] =
        {
            "counter_text",
            "counter_raw",
            "counter_average_base",
            "counter_average_count",
            "counter_aggregating",
            "counter_average_timer",
            "counter_elapsed_time"
        };
    }

    char const* get_counter_type_name(counter_type type)
    {
        if (type < counter_text || type > counter_elapsed_time)
            return "unknown";
        return strings::counter_type_names[type];
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status add_counter_type(counter_info const& info,
        create_counter_func const& create_counter,
        discover_counters_func const& discover_counters,
        error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        return rt ? rt->get_counter_registry().add_counter_type(
            info, create_counter, discover_counters, ec) : status_generic_error;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Call the supplied function for each registered counter type
    counter_status discover_counter_types(
        discover_counter_func const& discover_counter,
        discover_counters_mode mode, error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        return rt ? rt->get_counter_registry().discover_counter_types(
            discover_counter, mode, ec) : status_generic_error;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Call the supplied function for the given registered counter type.
    counter_status discover_counter_type(
        counter_info const& info,
        discover_counter_func const& discover_counter,
        discover_counters_mode mode, error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        return rt ? rt->get_counter_registry().discover_counter_type(
            info, discover_counter, mode, ec) : status_generic_error;
    }

    counter_status discover_counter_type(
        std::string const& name,
        discover_counter_func const& discover_counter,
        discover_counters_mode mode, error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        return rt ? rt->get_counter_registry().discover_counter_type(
            name, discover_counter, mode, ec) : status_generic_error;
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        bool discover_counters(counter_info const& info,
            std::vector<counter_info>& counters, error_code& ec)
        {
            counters.push_back(info);
            return true;
        }
    }

    counter_status discover_counter_types(std::vector<counter_info>& counters,
        discover_counters_mode mode, error_code& ec)
    {
        using hpx::util::placeholders::_1;

        discover_counter_func func(
            hpx::util::bind(&detail::discover_counters, _1, boost::ref(counters),
                boost::ref(ec)));

        return discover_counter_types(std::move(func), mode, ec);
    }

    counter_status discover_counter_type(
        std::string const& name, std::vector<counter_info>& counters,
        discover_counters_mode mode, error_code& ec)
    {
        using hpx::util::placeholders::_1;

        discover_counter_func func(
            hpx::util::bind(&detail::discover_counters, _1, boost::ref(counters),
                boost::ref(ec)));

        return discover_counter_type(name, std::move(func), mode, ec);
    }

    counter_status discover_counter_type(
        counter_info const& info, std::vector<counter_info>& counters,
        discover_counters_mode mode, error_code& ec)
    {
        using hpx::util::placeholders::_1;

        discover_counter_func func(
            hpx::util::bind(&detail::discover_counters, _1, boost::ref(counters),
                boost::ref(ec)));

        return discover_counter_type(info, std::move(func), mode, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status remove_counter_type(counter_info const& info, error_code& ec)
    {
        // the runtime might not be available any more
        runtime* rt = get_runtime_ptr();
        return rt ? rt->get_counter_registry().remove_counter_type(info, ec) :
            status_generic_error;
    }

    /// \brief Retrieve the counter type for the given counter name from the
    ///        (local) registry
    counter_status get_counter_type(std::string const& name,
        counter_info& info, error_code& ec)
    {
        // the runtime might not be available any more
        runtime* rt = get_runtime_ptr();
        return rt ?
            rt->get_counter_registry().get_counter_type(name, info, ec) :
            status_generic_error;
    }

    namespace detail
    {
        naming::gid_type create_raw_counter_value(
            counter_info const& info, boost::int64_t* countervalue,
            error_code& ec)
        {
            naming::gid_type gid;
            get_runtime().get_counter_registry().create_raw_counter_value(
                info, countervalue, gid, ec);
            return gid;
        }

        naming::gid_type create_raw_counter(counter_info const& info,
            hpx::util::function_nonser<boost::int64_t()> const& f, error_code& ec)
        {
            naming::gid_type gid;
            get_runtime().get_counter_registry().create_raw_counter(
                info, f, gid, ec);
            return gid;
        }

        naming::gid_type create_raw_counter(counter_info const& info,
            hpx::util::function_nonser<boost::int64_t(bool)> const& f, error_code& ec)
        {
            naming::gid_type gid;
            get_runtime().get_counter_registry().create_raw_counter(
                info, f, gid, ec);
            return gid;
        }

        // \brief Create a new performance counter instance based on given
        //        counter info
        naming::gid_type create_counter(counter_info const& info, error_code& ec)
        {
            naming::gid_type gid;
            get_runtime().get_counter_registry().create_counter(info, gid, ec);
            return gid;
        }

        // \brief Create a new aggregating performance counter instance based
        //        on given base counter name and given base time interval
        //        (milliseconds).
        naming::gid_type create_statistics_counter(
            counter_info const& info, std::string const& base_counter_name,
            std::vector<boost::int64_t> const& parameters, error_code& ec)
        {
            naming::gid_type gid;
            get_runtime().get_counter_registry().
                create_statistics_counter(info, base_counter_name,
                    parameters, gid, ec);
            return gid;
        }

        // \brief Create a new aggregating performance counter instance based
        //        on given base counter name and given base time interval
        //        (milliseconds).
        naming::gid_type create_arithmetics_counter(counter_info const& info,
            std::vector<std::string> const& base_counter_names, error_code& ec)
        {
            naming::gid_type gid;
            get_runtime().get_counter_registry().
                create_arithmetics_counter(info, base_counter_names, gid, ec);
            return gid;
        }

        ///////////////////////////////////////////////////////////////////////
        counter_status add_counter(naming::id_type const& id,
            counter_info const& info, error_code& ec)
        {
            return get_runtime().get_counter_registry().add_counter(id, info, ec);
        }

        counter_status remove_counter(counter_info const& info,
            naming::id_type const& id, error_code& ec)
        {
            return get_runtime().get_counter_registry().remove_counter(
                info, id, ec);
        }

        // create an arbitrary counter on this locality
        naming::gid_type create_counter_local(counter_info const& info)
        {
            // find create function for given counter
            error_code ec;

            create_counter_func f;
            get_runtime().get_counter_registry().get_counter_create_function(
                info, f, ec);
            if (ec) {
                HPX_THROW_EXCEPTION(bad_parameter, "create_counter_local",
                    "no create function for performance counter found: " +
                    remove_counter_prefix(info.fullname_) +
                        " (" + ec.get_message() + ")");
                return naming::invalid_gid;
            }

            counter_path_elements paths;
            get_counter_path_elements(info.fullname_, paths, ec);
            if (ec) return hpx::naming::invalid_gid;

            if (paths.parentinstancename_ == "locality" &&
                paths.parentinstanceindex_ !=
                    static_cast<boost::int64_t>(hpx::get_locality_id()))
            {
                HPX_THROW_EXCEPTION(bad_parameter, "create_counter_local",
                    "attempt to create counter on wrong locality ("
                     + ec.get_message() + ")");
                return hpx::naming::invalid_gid;
            }

            // attempt to create the new counter instance
            naming::gid_type gid = f(info, ec);
            if (ec) {
                HPX_THROW_EXCEPTION(bad_parameter, "create_counter_local",
                    "couldn't create performance counter: " +
                    remove_counter_prefix(info.fullname_) +
                        " (" + ec.get_message() + ")");
                return naming::invalid_gid;
            }

            return gid;
        }

        ///////////////////////////////////////////////////////////////////////
        inline bool is_thread_kind(std::string const& pattern)
        {
            std::string::size_type p = pattern.find("-thread#*");
            return p != std::string::npos && p == pattern.size() - 9;
        }

        inline std::string get_thread_kind(std::string const& pattern)
        {
            HPX_ASSERT(is_thread_kind(pattern));
            return pattern.substr(0, pattern.find_last_of('-'));
        }

        ///////////////////////////////////////////////////////////////////////
        inline bool is_node_kind(std::string const& pattern)
        {
            std::string::size_type p = pattern.find("-node#*");
            return p != std::string::npos && p == pattern.size() - 7;
        }

        inline std::string get_node_kind(std::string const& pattern)
        {
            HPX_ASSERT(is_node_kind(pattern));
            return pattern.substr(0, pattern.find_last_of('-'));
        }

        ///////////////////////////////////////////////////////////////////////
        /// Expand all wild-cards in a counter base name (for aggregate counters)
        bool expand_basecounter(
            counter_info const& info, counter_path_elements& p,
            discover_counter_func const& f, error_code& ec)
        {
            // discover all base names
            std::vector<counter_info> counter_infos;
            counter_status status = discover_counter_type(p.parentinstancename_,
                counter_infos, discover_counters_full, ec);
            if (!status_is_valid(status) || ec)
                return false;

            counter_info i = info;
            for (counter_info& basei : counter_infos)
            {
                p.parentinstancename_ = basei.fullname_;
                counter_status status = get_counter_name(p, i.fullname_, ec);
                if (!status_is_valid(status) || !f(i, ec) || ec)
                    return false;
            }
            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        // expand main counter name
        bool expand_counter_info_threads(
            counter_info& i, counter_path_elements& p,
            discover_counter_func const& f, error_code& ec)
        {
            std::size_t num_threads = get_os_thread_count();
            for (std::size_t l = 0; l != num_threads; ++l)
            {
                p.instanceindex_ = static_cast<boost::int64_t>(l);
                counter_status status = get_counter_name(p, i.fullname_, ec);
                if (!status_is_valid(status) || !f(i, ec) || ec)
                    return false;
            }
            return true;
        }

        bool expand_counter_info_nodes(
            counter_info& i, counter_path_elements& p,
            discover_counter_func const& f, error_code& ec)
        {
            std::size_t num_nodes
                = hpx::threads::get_topology().get_number_of_numa_nodes();
            for (std::size_t l = 0; l != num_nodes; ++l)
            {
                p.instanceindex_ = static_cast<boost::int64_t>(l);
                counter_status status = get_counter_name(p, i.fullname_, ec);
                if (!status_is_valid(status) || !f(i, ec) || ec)
                    return false;
            }
            return true;
        }

        bool expand_counter_info_localities(
            counter_info& i, counter_path_elements& p,
            discover_counter_func const& f, error_code& ec)
        {
            bool expand_threads = false;
            bool expand_nodes = false;
            if (is_thread_kind(p.instancename_))
            {
                p.instancename_ = get_thread_kind(p.instancename_) + "-thread";
                expand_threads = true;
            }
            else if (is_node_kind(p.instancename_))
            {
                p.instancename_ = get_node_kind(p.instancename_) + "-node";
                expand_nodes = true;
            }

            boost::uint32_t last_locality = get_num_localities_sync();
            for (boost::uint32_t l = 0; l != last_locality; ++l)
            {
                p.parentinstanceindex_ = static_cast<boost::int32_t>(l);
                if (expand_threads) {
                    if (!detail::expand_counter_info_threads(i, p, f, ec))
                        return false;
                }
                else if(expand_nodes) {
                    if (!detail::expand_counter_info_nodes(i, p, f, ec))
                        return false;
                }
                else {
                    counter_status status = get_counter_name(p, i.fullname_, ec);
                    if (!status_is_valid(status) || !f(i, ec) || ec)
                        return false;
                }
            }
            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        bool expand_counter_info(
            counter_info const& info, counter_path_elements& p,
            discover_counter_func const& f, error_code& ec)
        {
            // A '*' wild-card as the instance name is equivalent to no instance
            // name at all.
            if (p.parentinstancename_ == "*")
            {
                HPX_ASSERT(p.parentinstanceindex_ == -1);
                p.parentinstancename_.clear();
            }

            // first expand "locality*"
            if (p.parentinstancename_ == "locality#*")
            {
                counter_info i = info;
                p.parentinstancename_ = "locality";
                return detail::expand_counter_info_localities(i, p, f, ec);
            }

            // now expand "<...>-thread#*"
            if (detail::is_thread_kind(p.instancename_))
            {
                counter_info i = info;
                p.instancename_ = detail::get_thread_kind(p.instancename_) + "-thread";
                return detail::expand_counter_info_threads(i, p, f, ec);
            }

            // now expand "<...>-node#*"
            if (detail::is_node_kind(p.instancename_))
            {
                counter_info i = info;
                p.instancename_ = detail::get_node_kind(p.instancename_) + "-node";
                return detail::expand_counter_info_nodes(i, p, f, ec);
            }

            // handle wild-cards in aggregate counters
            if (p.parentinstance_is_basename_)
            {
                return detail::expand_basecounter(info, p, f, ec);
            }

            // everything else is handled directly
            return f(info, ec);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief call the supplied function will all expanded versions of the
    /// supplied counter info.
    bool expand_counter_info(counter_info const& info,
        discover_counter_func const& f, error_code& ec)
    {
        counter_path_elements p;
        counter_status status = get_counter_path_elements(info.fullname_, p, ec);
        if (!status_is_valid(status)) return false;

        return detail::expand_counter_info(info, p, f, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    naming::id_type register_with_agas(lcos::future<naming::id_type> f,
        std::string const& fullname)
    {
        // register the canonical name with AGAS
        naming::id_type id = f.get();
        agas::register_name_sync(fullname, id);
        return id;
    }

    ///////////////////////////////////////////////////////////////////////////
    lcos::future<naming::id_type> get_counter_async(
        counter_info const& info, error_code& ec)
    {
        typedef lcos::future<naming::id_type> result_type;

        // complement counter info data
        counter_info complemented_info = info;
        complement_counter_info(complemented_info, ec);
        if (ec) result_type();

        ensure_counter_prefix(complemented_info.fullname_);
        // pre-pend prefix, if necessary

        // ask AGAS for the id of the given counter
        naming::id_type id;
        bool result = agas::resolve_name_sync(complemented_info.fullname_, id, ec);
        if (!result) {
            try {
                // figure out target locality
                counter_path_elements p;
                get_counter_path_elements(complemented_info.fullname_, p, ec);
                if (ec) return result_type();

                // Take target locality from base counter if if this is an
                // aggregating counter (the instance name is a base counter).
                if (p.parentinstance_is_basename_)
                {
                    get_counter_path_elements(p.parentinstancename_, p, ec);
                    if (ec) return result_type();
                }

                if (p.parentinstancename_ == "locality" &&
                        (   p.parentinstanceindex_ < 0 ||
                            p.parentinstanceindex_ >= static_cast<boost::int32_t>
                            (get_num_localities_sync())
                        )
                    )
                {
                    HPX_THROWS_IF(ec, bad_parameter, "get_counter",
                        "attempt to create counter on non-existing locality");
                    return result_type();
                }

                // use the runtime_support component of the target locality to
                // create the new performance counter
                using namespace components::stubs;
                lcos::future<naming::id_type> f;
                if (p.parentinstanceindex_ >= 0) {
                    f = runtime_support::create_performance_counter_async(
                        naming::get_id_from_locality_id(
                            static_cast<boost::uint32_t>(p.parentinstanceindex_)),
                        complemented_info);
                }
                else {
                    f = runtime_support::create_performance_counter_async(
                        find_here(), complemented_info);
                }

                // attach the function which registers the id_type with AGAS
                return f.then(hpx::util::bind(&register_with_agas,
                    hpx::util::placeholders::_1,
                    complemented_info.fullname_));
            }
            catch (hpx::exception const& e) {
                if (&ec == &throws)
                    throw;
                ec = make_error_code(e.get_error(), e.what());
                LPCS_(warning) << (boost::format("failed to create counter %s (%s)")
                    % remove_counter_prefix(complemented_info.fullname_) % e.what());
                return lcos::future<naming::id_type>();
            }
        }
        if (ec) return result_type();

        return lcos::make_ready_future(id);
    }

    lcos::future<naming::id_type> get_counter_async(
        std::string const& name, error_code& ec)
    {
        ensure_counter_prefix(name);      // pre-pend prefix, if necessary

        counter_info info(name);          // set full counter name
        return get_counter_async(info, ec);
    }
}}

