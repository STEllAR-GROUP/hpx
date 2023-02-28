//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/performance_counters/counters_fwd.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters {
    ///////////////////////////////////////////////////////////////////////////
    constexpr char const counter_prefix[] = "/counters";
    constexpr std::size_t counter_prefix_len =
        (sizeof(counter_prefix) / sizeof(counter_prefix[0])) - 1;

    ///////////////////////////////////////////////////////////////////////////
    inline std::string& ensure_counter_prefix(std::string& name)
    {
        if (name.compare(0, counter_prefix_len, counter_prefix) != 0)
            name = counter_prefix + name;
        return name;
    }

    inline std::string ensure_counter_prefix(    //-V659
        std::string const& counter)
    {
        std::string name(counter);
        return ensure_counter_prefix(name);
    }

    inline std::string& remove_counter_prefix(std::string& name)
    {
        if (name.compare(0, counter_prefix_len, counter_prefix) == 0)
            name = name.substr(counter_prefix_len);
        return name;
    }

    inline std::string remove_counter_prefix(    //-V659
        std::string const& counter)
    {
        std::string name(counter);
        return remove_counter_prefix(name);
    }

#if defined(DOXYGEN)
    ///////////////////////////////////////////////////////////////////////////
    enum class counter_type
    {
        /// \a text shows a variable-length text string. It does not
        /// deliver calculated values.
        ///
        /// Formula:  None
        /// Average:  None
        /// Type:     Text
        text,

        /// \a raw shows the last observed value only. It does
        /// not deliver an average.
        ///
        /// Formula:  None. Shows raw data as collected.
        /// Average:  None
        /// Type:     Instantaneous
        raw,

        /// \a monotonically_increasing shows the cumulatively accumulated
        /// observed value. It does not deliver an average.
        ///
        /// Formula:  None. Shows cumulatively accumulated data as collected.
        /// Average:  None
        /// Type:     Instantaneous
        monotonically_increasing,

        /// \a average_base is used as the base data (denominator) in the
        /// computation of time or count averages for the
        /// \a counter_type::average_count and \a counter_type::average_timer
        /// counter types.
        /// This counter type collects the last observed value only.
        ///
        /// Formula:  None. This counter uses raw data in factional calculations
        /// without delivering an output.
        /// Average:  SUM (N) / x Type:
        /// Instantaneous
        average_base,

        /// \a average_count shows how many items are processed, on
        /// average, during an operation. Counters of this type display a ratio
        /// of the items processed (such as bytes sent) to the number of
        /// operations completed. The ratio is calculated by comparing the
        /// number of items processed during the last interval to the number of
        /// operations completed during the last interval.
        ///
        /// Formula:  (N1 - N0) / (D1 - D0), where the numerator (N) represents
        ///           the number of items processed during the last sample
        ///           interval, and the denominator (D) represents the number
        ///           of operations completed during the last two sample
        ///           intervals.
        /// Average:  (Nx - N0) / (Dx - D0)
        /// Type:     Average
        average_count,

        /// \a aggregating applies a function to an embedded counter
        /// instance. The embedded counter is usually evaluated repeatedly
        /// after a fixed (but configurable) time interval.
        ///
        /// Formula:  F(Nx)
        aggregating,

        /// \a average_timer measures the average time it takes to
        /// complete a process or operation. Counters of this type display a
        /// ratio of the total elapsed time of the sample interval to the
        /// number of processes or operations completed during that time. This
        /// counter type measures time in ticks of the system clock. The
        /// variable F represents the number of ticks per second. The value of
        /// F is factored into the equation so that the result is displayed in
        /// seconds.
        ///
        /// Formula:  ((N1 - N0) / F) / (D1 - D0), where the numerator (N)
        ///           represents the number of ticks counted during the last
        ///           sample interval, the variable F represents the frequency
        ///           of the ticks, and the denominator (D) represents the
        ///           number of operations completed during the last sample
        ///           interval.
        /// Average:  ((Nx - N0) / F) / (Dx - D0)
        /// Type:     Average
        average_timer,

        /// \a elapsed_time shows the total time between when the
        /// component or process started and the time when this value is
        /// calculated. The variable F represents the number of time units that
        /// elapse in one second. The value of F is factored into the equation
        /// so that the result is displayed in seconds.
        ///
        /// Formula:  (D0 - N0) / F, where the nominator (D) represents the
        ///           current time, the numerator (N) represents the time the
        ///           object was started, and the variable F represents the
        ///           number of time units that elapse in one second.
        /// Average:  (Dx - N0) / F
        /// Type:     Difference
        elapsed_time,

        /// \a histogram exposes a histogram of the measured values
        /// instead of a single value as many of the other counter types.
        /// Counters of this type expose a \a counter_value_array instead of a
        /// \a counter_value. Those will also not implement the
        /// \a get_counter_value() functionality. The results are exposed
        /// through a separate \a get_counter_values_array() function.
        ///
        /// The first three values in the returned array represent the lower
        /// and upper boundaries, and the size of the histogram buckets. All
        /// remaining values in the returned array represent the number of
        /// measurements for each of the buckets in the histogram.
        histogram,

        /// \a raw_values exposes an array of measured values
        /// instead of a single value as many of the other counter types.
        /// Counters of this type expose a \a counter_value_array instead of a
        /// \a counter_value. Those will also not implement the
        /// \a get_counter_value() functionality. The results are exposed
        /// through a separate \a get_counter_values_array() function.
        raw_values
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the readable name of a given counter type
    HPX_EXPORT char const* get_counter_type_name(counter_type state);

#if defined(DOXYGEN)
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Status and error codes used by the functions related to
    ///        performance counters.
    enum class counter_status
    {
        valid_data,         ///< No error occurred, data is valid
        new_data,           ///< Data is valid and different from last call
        invalid_data,       ///< Some error occurred, data is not value
        already_defined,    ///< The type or instance already has been defined
        counter_unknown,    ///< The counter instance is unknown
        counter_type_unknown,    ///< The counter type is unknown
        generic_error            ///< A unknown error occurred
    };
#endif

    inline bool status_is_valid(counter_status s)
    {
        return s == counter_status::valid_data || s == counter_status::new_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// A counter_type_path_elements holds the elements of a full name for a
    /// counter type. Generally, a full name of a counter type has the
    /// structure:
    ///
    ///    /objectname/countername
    ///
    /// i.e.
    ///    /queue/length
    ///
    struct counter_type_path_elements
    {
        counter_type_path_elements() = default;

        counter_type_path_elements(std::string const& objectname,
            std::string const& countername, std::string const& parameters)
          : objectname_(objectname)
          , countername_(countername)
          , parameters_(parameters)
        {
        }

        std::string objectname_;     ///< the name of the performance object
        std::string countername_;    ///< contains the counter name
        std::string parameters_;     ///< optional parameters for the
                                     ///< counter instance

    protected:
        // serialization support
        friend class hpx::serialization::access;

        HPX_EXPORT void serialize(
            serialization::output_archive& ar, const unsigned int);
        HPX_EXPORT void serialize(
            serialization::input_archive& ar, const unsigned int);
    };

    ///////////////////////////////////////////////////////////////////////////
    /// A counter_path_elements holds the elements of a full name for a counter
    /// instance. Generally, a full name of a counter instance has the
    /// structure:
    ///
    ///    /objectname{parentinstancename#parentindex/instancename#instanceindex}
    ///      /countername#parameters
    ///
    /// i.e.
    ///    /queue{localityprefix/thread#2}/length
    ///
    struct counter_path_elements : counter_type_path_elements
    {
        using base_type = counter_type_path_elements;

        counter_path_elements()
          : parentinstanceindex_(-1)
          , instanceindex_(-1)
          , subinstanceindex_(-1)
          , parentinstance_is_basename_(false)
        {
        }

        counter_path_elements(std::string const& objectname,
            std::string const& countername, std::string const& parameters,
            std::string const& parentname, std::string const& instancename,
            std::int64_t parentindex = -1, std::int64_t instanceindex = -1,
            bool parentinstance_is_basename = false)
          : counter_type_path_elements(objectname, countername, parameters)
          , parentinstancename_(parentname)
          , instancename_(instancename)
          , subinstancename_()
          , parentinstanceindex_(parentindex)
          , instanceindex_(instanceindex)
          , subinstanceindex_(-1)
          , parentinstance_is_basename_(parentinstance_is_basename)
        {
        }

        counter_path_elements(std::string const& objectname,
            std::string const& countername, std::string const& parameters,
            std::string const& parentname, std::string const& instancename,
            std::string const& subinstancename, std::int64_t parentindex = -1,
            std::int64_t instanceindex = -1, std::int64_t subinstanceindex = -1,
            bool parentinstance_is_basename = false)
          : counter_type_path_elements(objectname, countername, parameters)
          , parentinstancename_(parentname)
          , instancename_(instancename)
          , subinstancename_(subinstancename)
          , parentinstanceindex_(parentindex)
          , instanceindex_(instanceindex)
          , subinstanceindex_(subinstanceindex)
          , parentinstance_is_basename_(parentinstance_is_basename)
        {
        }

        std::string parentinstancename_;    ///< the name of the parent instance
        std::string instancename_;          ///< the name of the object instance
        std::string
            subinstancename_;    ///< the name of the object sub-instance
        std::int64_t parentinstanceindex_;    ///< the parent instance index
        std::int64_t instanceindex_;          ///< the instance index
        std::int64_t subinstanceindex_;       ///< the sub-instance index
        bool parentinstance_is_basename_;     ///< the parentinstancename_
            ///member holds a base counter name

    private:
        // serialization support
        friend class hpx::serialization::access;

        HPX_EXPORT void serialize(
            serialization::output_archive& ar, const unsigned int);
        HPX_EXPORT void serialize(
            serialization::input_archive& ar, const unsigned int);
    };

    ///////////////////////////////////////////////////////////////////////////
    struct counter_info
    {
        counter_info(counter_type type = counter_type::raw)
          : type_(type)
          , version_(HPX_PERFORMANCE_COUNTER_V1)
          , status_(counter_status::invalid_data)
        {
        }

        counter_info(std::string const& name)
          : type_(counter_type::raw)
          , version_(HPX_PERFORMANCE_COUNTER_V1)
          , status_(counter_status::invalid_data)
          , fullname_(name)
        {
        }

        counter_info(counter_type type, std::string const& name,
            std::string const& helptext = "",
            std::uint32_t version = HPX_PERFORMANCE_COUNTER_V1,
            std::string const& uom = "")
          : type_(type)
          , version_(version)
          , status_(counter_status::invalid_data)
          , fullname_(name)
          , helptext_(helptext)
          , unit_of_measure_(uom)
        {
        }

        counter_type type_;        ///< The type of the described counter
        std::uint32_t version_;    ///< The version of the described counter
                                   ///< using the 0xMMmmSSSS scheme
        counter_status status_;    ///< The status of the counter object
        std::string fullname_;     ///< The full name of this counter
        std::string
            helptext_;    ///< The full descriptive text for this counter
        std::string
            unit_of_measure_;    ///< The unit of measure for this counter

    private:
        // serialization support
        friend class hpx::serialization::access;

        HPX_EXPORT void serialize(
            serialization::output_archive& ar, const unsigned int);
        HPX_EXPORT void serialize(
            serialization::input_archive& ar, const unsigned int);
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief This declares the type of a function, which will be
    ///        called by HPX whenever a new performance counter instance of a
    ///        particular type needs to be created.
    using create_counter_func =
        hpx::function<naming::gid_type(counter_info const&, error_code&)>;

    ///////////////////////////////////////////////////////////////////////////
    /// \brief This declares a type of a function, which will be passed to
    ///        a \a discover_counters_func in order to be called for each
    ///        discovered performance counter instance.
    using discover_counter_func =
        hpx::function<bool(counter_info const&, error_code&)>;

    /// \brief This declares the type of a function, which will be called by
    ///        HPX whenever it needs to discover all performance counter
    ///        instances of a particular type.
    using discover_counters_func = hpx::function<bool(counter_info const&,
        discover_counter_func const&, discover_counters_mode, error_code&)>;

    ///////////////////////////////////////////////////////////////////////
    inline counter_status add_counter_type(
        counter_info const& info, error_code& ec)
    {
        return add_counter_type(
            info, create_counter_func(), discover_counters_func(), ec);
    }

    inline hpx::id_type get_counter(std::string const& name, error_code& ec)
    {
        hpx::future<hpx::id_type> f = get_counter_async(name, ec);
        if (ec)
            return hpx::invalid_id;

        return f.get(ec);
    }

    inline hpx::id_type get_counter(counter_info const& info, error_code& ec)
    {
        hpx::future<hpx::id_type> f = get_counter_async(info, ec);
        if (ec)
            return hpx::invalid_id;

        return f.get(ec);
    }
}}    // namespace hpx::performance_counters
