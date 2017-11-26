//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_FWD_NOV_26_2017_0107PM)
#define HPX_PERFORMANCE_COUNTERS_FWD_NOV_26_2017_0107PM

#include <hpx/config.hpp>
#include <hpx/lcos_fwd.hpp>
#include <hpx/error_code.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/util/function.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    ///////////////////////////////////////////////////////////////////////////
    inline std::string& ensure_counter_prefix(std::string& name);
    inline std::string ensure_counter_prefix(std::string const& counter);
    inline std::string& remove_counter_prefix(std::string& name);
    inline std::string remove_counter_prefix(std::string const& counter);

    ///////////////////////////////////////////////////////////////////////////
    enum counter_type
    {
        // \a counter_text shows a variable-length text string. It does not
        // deliver calculated values.
        //
        // Formula:  None
        // Average:  None
        // Type:     Text
        counter_text,

        // \a counter_raw shows the last observed value only. It does
        // not deliver an average.
        //
        // Formula:  None. Shows raw data as collected.
        // Average:  None
        // Type:     Instantaneous
        counter_raw,

        // \a counter_average_base is used as the base data (denominator) in the
        // computation of time or count averages for the \a counter_average_count
        // and \a counter_average_timer counter types. This counter type
        // collects the last observed value only.
        //
        // Formula:  None. This counter uses raw data in factional calculations
        //           without delivering an output.
        // Average:  SUM (N) / x
        // Type:     Instantaneous
        counter_average_base,

        // \a counter_average_count shows how many items are processed, on
        // average, during an operation. Counters of this type display a ratio
        // of the items processed (such as bytes sent) to the number of
        // operations completed. The ratio is calculated by comparing the
        // number of items processed during the last interval to the number of
        // operations completed during the last interval.
        //
        // Formula:  (N1 - N0) / (D1 - D0), where the numerator (N) represents
        //           the number of items processed during the last sample
        //           interval, and the denominator (D) represents the number
        //           of operations completed during the last two sample
        //           intervals.
        // Average:  (Nx - N0) / (Dx - D0)
        // Type:     Average
        counter_average_count,

        // \a counter_aggregating applies a function to an embedded counter
        // instance. The embedded counter is usually evaluated repeatedly
        // after a fixed (but configurable) time interval.
        //
        // Formula:  F(Nx)
        counter_aggregating,

        // \a counter_average_timer measures the average time it takes to
        // complete a process or operation. Counters of this type display a
        // ratio of the total elapsed time of the sample interval to the
        // number of processes or operations completed during that time. This
        // counter type measures time in ticks of the system clock. The
        // variable F represents the number of ticks per second. The value of
        // F is factored into the equation so that the result is displayed in
        // seconds.
        //
        // Formula:  ((N1 - N0) / F) / (D1 - D0), where the numerator (N)
        //           represents the number of ticks counted during the last
        //           sample interval, the variable F represents the frequency
        //           of the ticks, and the denominator (D) represents the
        //           number of operations completed during the last sample
        //           interval.
        // Average:  ((Nx - N0) / F) / (Dx - D0)
        // Type:     Average
        counter_average_timer,

        // \a counter_elapsed_time shows the total time between when the
        // component or process started and the time when this value is
        // calculated. The variable F represents the number of time units that
        // elapse in one second. The value of F is factored into the equation
        // so that the result is displayed in seconds.
        //
        // Formula:  (D0 - N0) / F, where the nominator (D) represents the
        //           current time, the numerator (N) represents the time the
        //           object was started, and the variable F represents the
        //           number of time units that elapse in one second.
        // Average:  (Dx - N0) / F
        // Type:     Difference
        counter_elapsed_time,

        // \a counter_histogram exposes a histogram of the measured values
        // instead of a single value as many of the other counter types.
        // Counters of this type expose a \a counter_value_array instead of a
        // \a counter_value. Those will also not implement the
        // \a get_counter_value() functionality. The results are exposed
        // through a separate \a get_counter_values_array() function.
        //
        // The first three values in the returned array represent the lower
        // and upper boundaries, and the size of the histogram buckets. All
        // remaining values in the returned array represent the number of
        // measurements for each of the buckets in the histogram.
        counter_histogram
    };

    ///////////////////////////////////////////////////////////////////////////
    // Return the readable name of a given counter type
    HPX_API_EXPORT char const* get_counter_type_name(counter_type state);

    ///////////////////////////////////////////////////////////////////////////
    // Status and error codes used by the functions related to
    // performance counters.
    enum counter_status
    {
        status_valid_data,      // No error occurred, data is valid
        status_new_data,        // Data is valid and different from last call
        status_invalid_data,    // Some error occurred, data is not value
        status_already_defined, // The type or instance already has been defined
        status_counter_unknown, // The counter instance is unknown
        status_counter_type_unknown,  // The counter type is unknown
        status_generic_error    // A unknown error occurred
    };

    inline bool status_is_valid(counter_status s);

    ///////////////////////////////////////////////////////////////////////////
    // A counter_type_path_elements holds the elements of a full name for a
    // counter type.
    struct counter_type_path_elements;

    ///////////////////////////////////////////////////////////////////////////
    // A counter_path_elements holds the elements of a full name for a counter
    // instance.
    struct counter_path_elements;

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a full name of a counter type from the contents of the
    ///        given \a counter_type_path_elements instance.The generated
    ///        counter type name will not contain any parameters.
    HPX_API_EXPORT counter_status get_counter_type_name(
        counter_type_path_elements const& path, std::string& result,
        error_code& ec = throws);

    /// \brief Create a full name of a counter type from the contents of the
    ///        given \a counter_type_path_elements instance. The generated
    ///        counter type name will contain all parameters.
    HPX_API_EXPORT counter_status get_full_counter_type_name(
        counter_type_path_elements const& path, std::string& result,
        error_code& ec = throws);

    /// \brief Create a full name of a counter from the contents of the given
    ///        \a counter_path_elements instance.
    HPX_API_EXPORT counter_status get_counter_name(
        counter_path_elements const& path, std::string& result,
        error_code& ec = throws);

    /// \brief Create a name of a counter instance from the contents of the
    ///        given \a counter_path_elements instance.
    HPX_API_EXPORT counter_status get_counter_instance_name(
        counter_path_elements const& path, std::string& result,
        error_code& ec = throws);

    /// \brief Fill the given \a counter_type_path_elements instance from the
    ///        given full name of a counter type
    HPX_API_EXPORT counter_status get_counter_type_path_elements(
        std::string const& name, counter_type_path_elements& path,
        error_code& ec = throws);

    /// \brief Fill the given \a counter_path_elements instance from the given
    ///        full name of a counter
    HPX_API_EXPORT counter_status get_counter_path_elements(
        std::string const& name, counter_path_elements& path,
        error_code& ec = throws);

    /// \brief Return the canonical counter instance name from a given full
    ///        instance name
    HPX_API_EXPORT counter_status get_counter_name(
        std::string const& name, std::string& countername,
        error_code& ec = throws);

    /// \brief Return the canonical counter type name from a given (full)
    ///        instance name
    HPX_API_EXPORT counter_status get_counter_type_name(
        std::string const& name, std::string& type_name,
        error_code& ec = throws);

    // default version of performance counter structures
    #define HPX_PERFORMANCE_COUNTER_V1 0x01000000

    ///////////////////////////////////////////////////////////////////////////
    struct counter_info;

    ///////////////////////////////////////////////////////////////////////////
    // This declares the type of a function, which will be
    // called by HPX whenever a new performance counter instance of a
    // particular type needs to be created.
    typedef hpx::util::function_nonser<
        naming::gid_type(counter_info const&, error_code&)>
        create_counter_func;

    ///////////////////////////////////////////////////////////////////////////
    // This declares a type of a function, which will be passed to
    // a \a discover_counters_func in order to be called for each
    // discovered performance counter instance.
    typedef hpx::util::function_nonser<
        bool(counter_info const&, error_code&)>
        discover_counter_func;

    enum discover_counters_mode
    {
        discover_counters_minimal,
        discover_counters_full      // fully expand all wild cards
    };

    // This declares the type of a function, which will be called by
    // HPX whenever it needs to discover all performance counter
    // instances of a particular type.
    typedef hpx::util::function_nonser<
        bool(counter_info const&, discover_counter_func const&,
            discover_counters_mode, error_code&)>
        discover_counters_func;

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Complement the counter info if parent instance name is missing
    HPX_API_EXPORT counter_status complement_counter_info(counter_info& info,
        counter_info const& type_info, error_code& ec = throws);

    HPX_API_EXPORT counter_status complement_counter_info(counter_info& info,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    struct counter_value;

    ///////////////////////////////////////////////////////////////////////////
    struct counter_values_array;

    ///////////////////////////////////////////////////////////////////////
    // Add a new performance counter type to the (local) registry
    HPX_API_EXPORT counter_status add_counter_type(counter_info const& info,
        create_counter_func const& create_counter,
        discover_counters_func const& discover_counters,
        error_code& ec = throws);

    inline counter_status add_counter_type(counter_info const& info,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Call the supplied function for each registered counter type
    HPX_API_EXPORT counter_status discover_counter_types(
        discover_counter_func const& discover_counter,
        discover_counters_mode mode = discover_counters_minimal,
        error_code& ec = throws);

    /// \brief Return a list of all available counter descriptions.
    HPX_API_EXPORT counter_status discover_counter_types(
        std::vector<counter_info>& counters,
        discover_counters_mode mode = discover_counters_minimal,
        error_code& ec = throws);

    /// \brief Call the supplied function for the given registered counter type.
    HPX_API_EXPORT counter_status discover_counter_type(
        std::string const& name,
        discover_counter_func const& discover_counter,
        discover_counters_mode mode = discover_counters_minimal,
        error_code& ec = throws);

    HPX_API_EXPORT counter_status discover_counter_type(
        counter_info const& info,
        discover_counter_func const& discover_counter,
        discover_counters_mode mode = discover_counters_minimal,
        error_code& ec = throws);

    /// \brief Return a list of matching counter descriptions for the given
    ///        registered counter type.
    HPX_API_EXPORT counter_status discover_counter_type(
        std::string const& name, std::vector<counter_info>& counters,
        discover_counters_mode mode = discover_counters_minimal,
        error_code& ec = throws);

    HPX_API_EXPORT counter_status discover_counter_type(
        counter_info const& info, std::vector<counter_info>& counters,
        discover_counters_mode mode = discover_counters_minimal,
        error_code& ec = throws);

    /// \brief call the supplied function will all expanded versions of the
    /// supplied counter info.
    ///
    /// This function expands all locality#* and worker-thread#* wild
    /// cards only.
    HPX_API_EXPORT bool expand_counter_info(counter_info const&,
        discover_counter_func const&, error_code&);

    /// \brief Remove an existing counter type from the (local) registry
    ///
    /// \note This doesn't remove existing counters of this type, it just
    ///       inhibits defining new counters using this type.
    HPX_API_EXPORT counter_status remove_counter_type(
        counter_info const& info, error_code& ec = throws);

    /// \brief Retrieve the counter type for the given counter name from the
    ///        (local) registry
    HPX_API_EXPORT counter_status get_counter_type(std::string const& name,
        counter_info& info, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Get the global id of an existing performance counter, if the
    ///        counter does not exist yet, the function attempts to create the
    ///        counter based on the given counter name.
    HPX_API_EXPORT lcos::future<naming::id_type>
        get_counter_async(std::string name, error_code& ec = throws);

    inline naming::id_type get_counter(std::string const& name,
        error_code& ec = throws);

    /// \brief Get the global id of an existing performance counter, if the
    ///        counter does not exist yet, the function attempts to create the
    ///        counter based on the given counter info.
    HPX_API_EXPORT lcos::future<naming::id_type>
        get_counter_async(counter_info const& info, error_code& ec = throws);

    inline naming::id_type get_counter(counter_info const& info,
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Retrieve the meta data specific for the given counter instance
    HPX_API_EXPORT void get_counter_infos(counter_info const& info,
        counter_type& type, std::string& helptext, std::uint32_t& version,
        error_code& ec = throws);

    /// \brief Retrieve the meta data specific for the given counter instance
    HPX_API_EXPORT void get_counter_infos(std::string name, counter_type& type,
        std::string& helptext, std::uint32_t& version, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \brief Add an existing performance counter instance to the registry
        HPX_API_EXPORT counter_status add_counter(naming::id_type const& id,
            counter_info const& info, error_code& ec = throws);

        /// \brief Remove an existing performance counter instance with the
        ///        given id (as returned from \a create_counter)
        HPX_API_EXPORT counter_status remove_counter(
            counter_info const& info, naming::id_type const& id,
            error_code& ec = throws);

        ///////////////////////////////////////////////////////////////////////
        // Helper function for creating counters encapsulating a function
        // returning the counter value.
        HPX_EXPORT naming::gid_type create_raw_counter(counter_info const&,
            hpx::util::function_nonser<std::int64_t()> const&, error_code&);

        // Helper function for creating counters encapsulating a function
        // returning the counter value.
        HPX_EXPORT naming::gid_type create_raw_counter(counter_info const&,
            hpx::util::function_nonser<std::int64_t(bool)> const&, error_code&);

        // Helper function for creating counters encapsulating a function
        // returning the counter values array.
        HPX_EXPORT naming::gid_type create_raw_counter(counter_info const&,
            hpx::util::function_nonser<std::vector<std::int64_t>()> const&,
            error_code&);

        // Helper function for creating counters encapsulating a function
        // returning the counter values array.
        HPX_EXPORT naming::gid_type create_raw_counter(counter_info const&,
            hpx::util::function_nonser<std::vector<std::int64_t>(bool)> const&,
            error_code&);

        // Helper function for creating a new performance counter instance
        // based on a given counter value.
        HPX_EXPORT naming::gid_type create_raw_counter_value(
            counter_info const&, std::int64_t*, error_code&);

        // Creation function for aggregating performance counters; to be
        // registered with the counter types.
        HPX_EXPORT naming::gid_type statistics_counter_creator(
            counter_info const&, error_code&);

        // Creation function for aggregating performance counters; to be
        // registered with the counter types.
        HPX_EXPORT naming::gid_type arithmetics_counter_creator(
            counter_info const&, error_code&);

        // Creation function for extended aggregating performance counters; to
        // be registered with the counter types.
        HPX_EXPORT naming::gid_type arithmetics_counter_extended_creator(
            counter_info const&, error_code&);

        // Creation function for uptime counters.
        HPX_EXPORT naming::gid_type uptime_counter_creator(
            counter_info const&, error_code&);

        // Creation function for instance counters.
        HPX_EXPORT naming::gid_type component_instance_counter_creator(
            counter_info const&, error_code&);

        // \brief Create a new statistics performance counter instance based on
        //        the given base counter name and given base time interval
        //        (milliseconds).
        HPX_EXPORT naming::gid_type create_statistics_counter(
            counter_info const& info, std::string const& base_counter_name,
            std::vector<std::size_t> const& parameters,
            error_code& ec = throws);

        // \brief Create a new arithmetics performance counter instance based on
        //        the given base counter names
        HPX_EXPORT naming::gid_type create_arithmetics_counter(
            counter_info const& info,
            std::vector<std::string> const& base_counter_names,
            error_code& ec = throws);

        // \brief Create a new extended arithmetics performance counter instance
        //        based on the given base counter names
        HPX_EXPORT naming::gid_type create_arithmetics_counter_extended(
            counter_info const& info,
            std::vector<std::string> const& base_counter_names,
            error_code& ec = throws);

        // \brief Create a new performance counter instance based on given
        //        counter info
        HPX_EXPORT naming::gid_type create_counter(counter_info const& info,
            error_code& ec = throws);

        // \brief Create an arbitrary counter on this locality
        HPX_EXPORT naming::gid_type create_counter_local(
            counter_info const& info);
    }
}}

#endif

