//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_MAR_01_2009_0134PM)
#define HPX_PERFORMANCE_COUNTERS_MAR_01_2009_0134PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/serialization/base_object.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/function.hpp>

#include <boost/cstdint.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    ///////////////////////////////////////////////////////////////////////////
    char const counter_prefix[] = "/counters";

    ///////////////////////////////////////////////////////////////////////////
    inline std::string& ensure_counter_prefix(std::string& name)
    {
        if (name.find(counter_prefix) != 0)
            name = counter_prefix + name;
        return name;
    }

    inline std::string ensure_counter_prefix(std::string const& counter) //-V659
    {
        std::string name(counter);
        return ensure_counter_prefix(name);
    }

    inline std::string& remove_counter_prefix(std::string& name)
    {
        if (name.find(counter_prefix) == 0)
            name = name.substr(sizeof(counter_prefix)-1);
        return name;
    }

    inline std::string remove_counter_prefix(std::string const& counter) //-V659
    {
        std::string name(counter);
        return remove_counter_prefix(name);
    }

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
        counter_histogram
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the readable name of a given counter type
    HPX_API_EXPORT char const* get_counter_type_name(counter_type state);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Status and error codes used by the functions related to
    ///        performance counters.
    enum counter_status
    {
        status_valid_data,      ///< No error occurred, data is valid
        status_new_data,        ///< Data is valid and different from last call
        status_invalid_data,    ///< Some error occurred, data is not value
        status_already_defined, ///< The type or instance already has been defined
        status_counter_unknown, ///< The counter instance is unknown
        status_counter_type_unknown,  ///< The counter type is unknown
        status_generic_error    ///< A unknown error occurred
    };

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
        counter_type_path_elements()
        {}

        counter_type_path_elements(std::string const& objectname,
                std::string const& countername, std::string const& parameters)
          : objectname_(objectname),
            countername_(countername),
            parameters_(parameters)
        {}

        std::string objectname_;          ///< the name of the performance object
        std::string countername_;         ///< contains the counter name
        std::string parameters_;          ///< optional parameters for the
                                          ///< counter instance

    protected:
        // serialization support
        friend class hpx::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & objectname_ & countername_ & parameters_;
        }
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
          : parentinstanceindex_(-1), instanceindex_(-1),
            parentinstance_is_basename_(false)
        {}

        counter_path_elements(std::string const& objectname,
                std::string const& countername, std::string const& parameters,
                std::string const& parentname, std::string const& instancename,
                boost::int64_t parentindex = -1, boost::int64_t instanceindex = -1,
                bool parentinstance_is_basename = false)
          : counter_type_path_elements(objectname, countername, parameters),
            parentinstancename_(parentname), instancename_(instancename),
            parentinstanceindex_(parentindex), instanceindex_(instanceindex),
            parentinstance_is_basename_(parentinstance_is_basename)
        {}

        std::string parentinstancename_;  ///< the name of the parent instance
        std::string instancename_;        ///< the name of the object instance
        boost::int64_t parentinstanceindex_;    ///< the parent instance index
        boost::int64_t instanceindex_;    ///< the instance index
        bool parentinstance_is_basename_; ///< the parentinstancename_
                                          ///member holds a base counter name

    private:
        // serialization support
        friend class hpx::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            typedef counter_type_path_elements base_type;
            hpx::serialization::base_object_type<
              counter_path_elements, base_type> base =
                hpx::serialization::base_object<base_type>(*this);
            ar & base & parentinstancename_ & instancename_ &
                 parentinstanceindex_ & instanceindex_ &
                 parentinstance_is_basename_;
        }
    };

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
    struct counter_info
    {
        counter_info(counter_type type = counter_raw)
          : type_(type), version_(HPX_PERFORMANCE_COUNTER_V1),
            status_(status_invalid_data)
        {}

        counter_info(std::string const& name)
          : type_(counter_raw), version_(HPX_PERFORMANCE_COUNTER_V1),
            status_(status_invalid_data), fullname_(name)
        {}

        counter_info(counter_type type, std::string const& name,
                std::string const& helptext = "",
                boost::uint32_t version = HPX_PERFORMANCE_COUNTER_V1,
                std::string const& uom = "")
          : type_(type), version_(version), status_(status_invalid_data),
            fullname_(name), helptext_(helptext), unit_of_measure_(uom)
        {}

        counter_type type_;         ///< The type of the described counter
        boost::uint32_t version_;   ///< The version of the described counter
                                    ///< using the 0xMMmmSSSS scheme
        counter_status status_;     ///< The status of the counter object
        std::string fullname_;      ///< The full name of this counter
        std::string helptext_;      ///< The full descriptive text for this counter
        std::string unit_of_measure_; ///< The unit of measure for this counter

    private:
        // serialization support
        friend class hpx::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & type_ & version_ & status_
               & fullname_ & helptext_
               & unit_of_measure_;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief This declares the type of a function, which will be
    ///        called by HPX whenever a new performance counter instance of a
    ///        particular type needs to be created.
    typedef hpx::util::function_nonser<
        naming::gid_type(counter_info const&, error_code&)>
        create_counter_func;

    ///////////////////////////////////////////////////////////////////////////
    /// \brief This declares a type of a function, which will be passed to
    ///        a \a discover_counters_func in order to be called for each
    ///        discovered performance counter instance.
    typedef hpx::util::function_nonser<
        bool(counter_info const&, error_code&)>
        discover_counter_func;

    enum discover_counters_mode
    {
        discover_counters_minimal,
        discover_counters_full      ///< fully expand all wild cards
    };

    /// \brief This declares the type of a function, which will be called by
    ///        HPX whenever it needs to discover all performance counter
    ///        instances of a particular type.
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
    struct counter_value
    {
        counter_value(boost::int64_t value = 0, boost::int64_t scaling = 1,
                bool scale_inverse = false)
          : time_(), count_(0), value_(value), scaling_(scaling),
            status_(status_new_data),
            scale_inverse_(scale_inverse)
        {}

        boost::uint64_t time_;      ///< The local time when data was collected
        boost::uint64_t count_;     ///< The invocation counter for the data
        boost::int64_t value_;      ///< The current counter value
        boost::int64_t scaling_;    ///< The scaling of the current counter value
        counter_status status_;     ///< The status of the counter value
        bool scale_inverse_;        ///< If true, value_ needs to be divided by
                                    ///< scaling_, otherwise it has to be
                                    ///< multiplied.

        /// \brief Retrieve the 'real' value of the counter_value, converted to
        ///        the requested type \a T
        template <typename T>
        T get_value(error_code& ec = throws) const
        {
            if (!status_is_valid(status_)) {
                HPX_THROWS_IF(ec, invalid_status,
                    "counter_value::get_value<T>",
                    "counter value is in invalid status");
                return T();
            }

            T val = static_cast<T>(value_);

            if (scaling_ != 1) {
                if (scaling_ == 0) {
                    HPX_THROWS_IF(ec, uninitialized_value,
                        "counter_value::get_value<T>",
                        "scaling should not be zero");
                    return T();
                }

                // calculate and return the real counter value
                if (scale_inverse_)
                    return val / static_cast<T>(scaling_);

                return val * static_cast<T>(scaling_);
            }
            return val;
        }

    private:
        // serialization support
        friend class hpx::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & status_ & time_ & count_ & value_ & scaling_ & scale_inverse_;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    struct counter_values_array
    {
        counter_values_array(boost::int64_t scaling = 1,
                bool scale_inverse = false)
          : time_(), count_(0), values_(), scaling_(scaling),
            status_(status_new_data),
            scale_inverse_(scale_inverse)
        {}

        counter_values_array(std::vector<boost::int64_t> && values,
                boost::int64_t scaling = 1, bool scale_inverse = false)
          : time_(), count_(0), values_(std::move(values)), scaling_(scaling),
            status_(status_new_data),
            scale_inverse_(scale_inverse)
        {}

        counter_values_array(std::vector<boost::int64_t> const& values,
                boost::int64_t scaling = 1, bool scale_inverse = false)
          : time_(), count_(0), values_(values), scaling_(scaling),
            status_(status_new_data),
            scale_inverse_(scale_inverse)
        {}

        boost::uint64_t time_;      ///< The local time when data was collected
        boost::uint64_t count_;     ///< The invocation counter for the data
        std::vector<boost::int64_t> values_;  ///< The current counter values
        boost::int64_t scaling_;    ///< The scaling of the current counter values
        counter_status status_;     ///< The status of the counter value
        bool scale_inverse_;        ///< If true, value_ needs to be divided by
                                    ///< scaling_, otherwise it has to be
                                    ///< multiplied.

        /// \brief Retrieve the 'real' value of the counter_value, converted to
        ///        the requested type \a T
        template <typename T>
        T get_value(std::size_t index, error_code& ec = throws) const
        {
            if (!status_is_valid(status_)) {
                HPX_THROWS_IF(ec, invalid_status,
                    "counter_values_array::get_value<T>",
                    "counter value is in invalid status");
                return T();
            }
            if (index >= values_.size()) {
                HPX_THROWS_IF(ec, bad_parameter,
                    "counter_values_array::get_value<T>",
                    "index out of bounds");
                return T();
            }

            T val = static_cast<T>(values_[index]);

            if (scaling_ != 1) {
                if (scaling_ == 0) {
                    HPX_THROWS_IF(ec, uninitialized_value,
                        "counter_values_array::get_value<T>",
                        "scaling should not be zero");
                    return T();
                }

                // calculate and return the real counter value
                if (scale_inverse_)
                    return val / static_cast<T>(scaling_);

                return val * static_cast<T>(scaling_);
            }
            return val;
        }

    private:
        // serialization support
        friend class hpx::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & status_ & time_ & count_ & values_ & scaling_ & scale_inverse_;
        }
    };

    ///////////////////////////////////////////////////////////////////////
    /// \brief Add a new performance counter type to the (local) registry
    HPX_API_EXPORT counter_status add_counter_type(counter_info const& info,
        create_counter_func const& create_counter,
        discover_counters_func const& discover_counters,
        error_code& ec = throws);

    inline counter_status add_counter_type(counter_info const& info,
        error_code& ec = throws)
    {
        return add_counter_type(info, create_counter_func(),
            discover_counters_func(), ec);
    }

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
        get_counter_async(std::string const& name, error_code& ec = throws);

    inline naming::id_type get_counter(std::string const& name,
        error_code& ec = throws)
    {
        lcos::future<naming::id_type> f = get_counter_async(name, ec);
        if (ec) return naming::invalid_id;

        return f.get(ec);
    }

    /// \brief Get the global id of an existing performance counter, if the
    ///        counter does not exist yet, the function attempts to create the
    ///        counter based on the given counter info.
    HPX_API_EXPORT lcos::future<naming::id_type>
        get_counter_async(counter_info const& info, error_code& ec = throws);

    inline naming::id_type get_counter(counter_info const& info,
        error_code& ec = throws)
    {
        lcos::future<naming::id_type> f = get_counter_async(info, ec);
        if (ec) return naming::invalid_id;

        return f.get(ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Retrieve the meta data specific for the given counter instance
    HPX_API_EXPORT void get_counter_infos(counter_info const& info,
        counter_type& type, std::string& helptext, boost::uint32_t& version,
        error_code& ec = throws);

    /// \brief Retrieve the meta data specific for the given counter instance
    HPX_API_EXPORT void get_counter_infos(std::string name, counter_type& type,
        std::string& helptext, boost::uint32_t& version, error_code& ec = throws);

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
            hpx::util::function_nonser<boost::int64_t()> const&, error_code&);

        // Helper function for creating counters encapsulating a function
        // returning the counter value.
        HPX_EXPORT naming::gid_type create_raw_counter(counter_info const&,
            hpx::util::function_nonser<boost::int64_t(bool)> const&, error_code&);

        // Helper function for creating counters encapsulating a function
        // returning the counter values array.
        HPX_EXPORT naming::gid_type create_raw_counter(counter_info const&,
            hpx::util::function_nonser<std::vector<boost::int64_t>()> const&,
            error_code&);

        // Helper function for creating counters encapsulating a function
        // returning the counter values array.
        HPX_EXPORT naming::gid_type create_raw_counter(counter_info const&,
            hpx::util::function_nonser<std::vector<boost::int64_t>(bool)> const&,
            error_code&);

        // Helper function for creating a new performance counter instance
        // based on a given counter value.
        HPX_EXPORT naming::gid_type create_raw_counter_value(
            counter_info const&, boost::int64_t*, error_code&);

        // Creation function for aggregating performance counters; to be
        // registered with the counter types.
        HPX_EXPORT naming::gid_type statistics_counter_creator(
            counter_info const&, error_code&);

        // Creation function for aggregating performance counters; to be
        // registered with the counter types.
        HPX_EXPORT naming::gid_type arithmetics_counter_creator(
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
            std::vector<boost::int64_t> const& parameters,
            error_code& ec = throws);

        // \brief Create a new arithmetics performance counter instance based on
        //        the given base counter names
        HPX_EXPORT naming::gid_type create_arithmetics_counter(
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

