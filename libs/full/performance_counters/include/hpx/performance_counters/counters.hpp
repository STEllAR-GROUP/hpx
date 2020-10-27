//  Copyright (c) 2007-2016 Hartmut Kaiser
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

    inline std::string ensure_counter_prefix(
        std::string const& counter)    //-V659
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

    inline std::string remove_counter_prefix(
        std::string const& counter)    //-V659
    {
        std::string name(counter);
        return remove_counter_prefix(name);
    }

#if defined(DOXYGEN)
    ///////////////////////////////////////////////////////////////////////////
    enum counter_type
    {
        /// \a counter_text shows a variable-length text string. It does not
        /// deliver calculated values.
        ///
        /// Formula:  None
        /// Average:  None
        /// Type:     Text
        counter_text,

        /// \a counter_raw shows the last observed value only. It does
        /// not deliver an average.
        ///
        /// Formula:  None. Shows raw data as collected.
        /// Average:  None
        /// Type:     Instantaneous
        counter_raw,

        // \a counter_raw shows the cumulatively accumulated observed value.
        // It does not deliver an average.
        //
        // Formula:  None. Shows cumulatively accumulated data as collected.
        // Average:  None
        // Type:     Instantaneous
        counter_monotonically_increasing,

        /// \a counter_average_base is used as the base data (denominator) in the
        /// computation of time or count averages for the \a counter_average_count
        /// and \a counter_average_timer counter types. This counter type
        /// collects the last observed value only.
        ///
        /// Formula:  None. This counter uses raw data in factional calculations
        ///           without delivering an output.
        /// Average:  SUM (N) / x
        /// Type:     Instantaneous
        counter_average_base,

        /// \a counter_average_count shows how many items are processed, on
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
        counter_average_count,

        /// \a counter_aggregating applies a function to an embedded counter
        /// instance. The embedded counter is usually evaluated repeatedly
        /// after a fixed (but configurable) time interval.
        ///
        /// Formula:  F(Nx)
        counter_aggregating,

        /// \a counter_average_timer measures the average time it takes to
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
        counter_average_timer,

        /// \a counter_elapsed_time shows the total time between when the
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
        counter_elapsed_time,

        /// \a counter_histogram exposes a histogram of the measured values
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
        counter_histogram,

        /// \a counter_raw_values exposes an array of measured values
        /// instead of a single value as many of the other counter types.
        /// Counters of this type expose a \a counter_value_array instead of a
        /// \a counter_value. Those will also not implement the
        /// \a get_counter_value() functionality. The results are exposed
        /// through a separate \a get_counter_values_array() function.
        counter_raw_values
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the readable name of a given counter type
    HPX_EXPORT char const* get_counter_type_name(counter_type state);

#if defined(DOXYGEN)
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Status and error codes used by the functions related to
    ///        performance counters.
    enum counter_status
    {
        status_valid_data,      ///< No error occurred, data is valid
        status_new_data,        ///< Data is valid and different from last call
        status_invalid_data,    ///< Some error occurred, data is not value
        status_already_defined,    ///< The type or instance already has been defined
        status_counter_unknown,         ///< The counter instance is unknown
        status_counter_type_unknown,    ///< The counter type is unknown
        status_generic_error            ///< A unknown error occurred
    };
#endif

    inline bool status_is_valid(counter_status s)
    {
        return s == status_valid_data || s == status_new_data;
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
        counter_type_path_elements() {}

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
        typedef counter_type_path_elements base_type;

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
        counter_info(counter_type type = counter_raw)
          : type_(type)
          , version_(HPX_PERFORMANCE_COUNTER_V1)
          , status_(status_invalid_data)
        {
        }

        counter_info(std::string const& name)
          : type_(counter_raw)
          , version_(HPX_PERFORMANCE_COUNTER_V1)
          , status_(status_invalid_data)
          , fullname_(name)
        {
        }

        counter_info(counter_type type, std::string const& name,
            std::string const& helptext = "",
            std::uint32_t version = HPX_PERFORMANCE_COUNTER_V1,
            std::string const& uom = "")
          : type_(type)
          , version_(version)
          , status_(status_invalid_data)
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
    typedef hpx::util::function_nonser<naming::gid_type(
        counter_info const&, error_code&)>
        create_counter_func;

    ///////////////////////////////////////////////////////////////////////////
    /// \brief This declares a type of a function, which will be passed to
    ///        a \a discover_counters_func in order to be called for each
    ///        discovered performance counter instance.
    typedef hpx::util::function_nonser<bool(counter_info const&, error_code&)>
        discover_counter_func;

    /// \brief This declares the type of a function, which will be called by
    ///        HPX whenever it needs to discover all performance counter
    ///        instances of a particular type.
    typedef hpx::util::function_nonser<bool(counter_info const&,
        discover_counter_func const&, discover_counters_mode, error_code&)>
        discover_counters_func;

    ///////////////////////////////////////////////////////////////////////
    inline counter_status add_counter_type(
        counter_info const& info, error_code& ec)
    {
        return add_counter_type(
            info, create_counter_func(), discover_counters_func(), ec);
    }

    inline naming::id_type get_counter(std::string const& name, error_code& ec)
    {
        lcos::future<naming::id_type> f = get_counter_async(name, ec);
        if (ec)
            return naming::invalid_id;

        return f.get(ec);
    }

    inline naming::id_type get_counter(counter_info const& info, error_code& ec)
    {
        lcos::future<naming::id_type> f = get_counter_async(info, ec);
        if (ec)
            return naming::invalid_id;

        return f.get(ec);
    }
}}    // namespace hpx::performance_counters
