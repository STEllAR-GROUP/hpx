//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_MAR_01_2009_0134PM)
#define HPX_PERFORMANCE_COUNTERS_MAR_01_2009_0134PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>

#include <boost/cstdint.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>

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

    inline std::string ensure_counter_prefix(std::string const& counter)
    {
        std::string name(counter);
        return ensure_counter_prefix(name);
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
        counter_elapsed_time
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
        status_generic_error,   ///< A unknown error occurred
    };

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

        template <typename S1, typename S2>
        counter_type_path_elements(S1 const& obj, S2 const& counter)
          : objectname_(obj), countername_(counter)
        {}

        std::string objectname_;          ///< the name of the performance object 
        std::string countername_;         ///< contains the counter name

    protected:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & objectname_ & countername_;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    /// A counter_path_elements holds the elements of a full name for a counter 
    /// instance. Generally, a full name of a counter instance has the 
    /// structure:
    ///
    ///    /objectname(parentinstancename#parentindex/instancename#instanceindex)/countername
    ///
    /// i.e.
    ///    /queue(localityprefix/thread#2)/length
    ///
    struct counter_path_elements : counter_type_path_elements
    {
        typedef counter_type_path_elements base_type;

        counter_path_elements() 
          : parentinstanceindex_(-1), instanceindex_(-1)
        {}

        template <typename S1, typename S2, typename S3, typename S4>
        counter_path_elements(S1 const& obj, S2 const& counter, 
                S3 const& parent, S4 const& instance, 
                boost::int32_t parentindex = -1, boost::int32_t index = -1) 
          : base_type(obj, counter), 
            parentinstancename_(parent), instancename_(instance),
            parentinstanceindex_(parentindex), instanceindex_(index)
        {}

        std::string parentinstancename_;  ///< the name of the parent instance 
        std::string instancename_;        ///< the name of the object instance 
        boost::int32_t parentinstanceindex_;    ///< the parent instance index 
        boost::int32_t instanceindex_;    ///< the instance index 

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            typedef counter_type_path_elements base_type;
            ar & boost::serialization::base_object<base_type>(*this);
            ar & parentinstancename_ & instancename_ & parentinstanceindex_ & instanceindex_;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Create a full name of a counter type from the contents of the 
    ///        given \a counter_type_path_elements instance.
    HPX_API_EXPORT counter_status get_counter_name(
        counter_type_path_elements const& path, std::string& result, 
        error_code& ec = throws);

    /// \brief Create a full name of a counter from the contents of the given 
    ///        \a counter_path_elements instance.
    HPX_API_EXPORT counter_status get_counter_name(
        counter_path_elements const& path, std::string& result, 
        error_code& ec = throws);

    /// \brief Fill the given \a counter_type_path_elements instance from the 
    ///        given full name of a counter type
    HPX_API_EXPORT counter_status get_counter_path_elements(
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
            status_(status_valid_data)
        {}

        counter_info(counter_type type, std::string const& name, 
                std::string const& helptext = "", 
                boost::uint32_t version = HPX_PERFORMANCE_COUNTER_V1)
          : type_(type), version_(version), status_(status_valid_data),
            fullname_(name), helptext_(helptext)
        {}

        counter_type type_;         ///< The type of the described counter
        boost::uint32_t version_;   ///< The version of the described counter
                                    ///< using the 0xMMmmSSSS scheme
        counter_status status_;     ///< The status of the counter object
        std::string fullname_;      ///< The full name of this counter 
        std::string helptext_;      ///< The full descriptive text for this counter

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & type_ & version_ & status_ & fullname_ & helptext_;
        }
    };

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
          : status_(status_new_data), time_(), 
            value_(value), scaling_(scaling), scale_inverse_(scale_inverse)
        {}

        counter_status status_;     ///< The status of the counter value
        boost::int64_t time_;       ///< The local time when data was collected
        boost::int64_t value_;      ///< The current counter value
        boost::int64_t scaling_;    ///< The scaling of the current counter value
        bool scale_inverse_;        ///< If true, value_ needs to be deleted by 
                                    ///< scaling_, otherwise it has to be 
                                    ///< multiplied.

        /// \brief Retrieve the 'real' value of the counter_value, converted to 
        ///        the requested type \a T
        template <typename T>
        T get_value(error_code& ec = throws)
        {
            if (status_valid_data != status_) {
                HPX_THROWS_IF(ec, invalid_status,
                    "counter_value::get_value<T>", 
                    "counter value is in invalid status");
                return T();
            }

            if (scaling_ != 1) {
                if (scaling_ == 0) {
                    HPX_THROWS_IF(ec, uninitialized_value,
                        "counter_value::get_value<T>", 
                        "scaling should not be zero");
                    return T();
                }

                // calculate and return the real counter value
                if (scale_inverse_) 
                    return T(value_) / scaling_;

                return T(value_) * scaling_;
            }
            return value_;
        }

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & status_ & time_ & value_ & scaling_ & scale_inverse_;
        }
    };

    ///////////////////////////////////////////////////////////////////////
    /// \brief Add a new performance counter type to the (local) registry
    HPX_API_EXPORT counter_status add_counter_type(
        counter_info const& info, error_code& ec = throws);

    /// \brief Remove an existing counter type from the (local) registry
    ///
    /// \note This doesn't remove existing counters of this type, it just
    ///       inhibits defining new counters using this type.
    HPX_API_EXPORT counter_status remove_counter_type(
        counter_info const& info, error_code& ec = throws);

    /// \brief Create a new performance counter instance based on given
    ///        counter value 
    HPX_API_EXPORT counter_status create_raw_counter(
        counter_info const& info, boost::int64_t* countervalue, 
        naming::id_type& id, error_code& ec = throws);

    /// \brief Create a new performance counter instance based on given
    ///        function returning the counter value
    HPX_API_EXPORT counter_status create_raw_counter(
        counter_info const& info, boost::function<boost::int64_t()> f, 
        naming::id_type& id, error_code& ec = throws);

    /// \brief Create a new performance counter instance based on given
    ///        counter info
    HPX_API_EXPORT counter_status create_counter(
        counter_info const& info, naming::id_type& id, 
        error_code& ec = throws);

    /// \brief Create a new performance counter instance of type 
    ///        counter_average_count based on given base counter name and
    ///        given base time interval (milliseconds)
    HPX_API_EXPORT counter_status create_average_count_counter(
        performance_counters::counter_info const& info, 
        std::string const& base_counter_name, std::size_t base_time_interval,
        naming::id_type& id, error_code& ec = throws);

    /// \brief Add an existing performance counter instance to the registry
    HPX_API_EXPORT counter_status add_counter(naming::id_type const& id, 
        counter_info const& info, error_code& ec = throws);

    /// \brief Remove an existing performance counter instance with the 
    ///        given id (as returned from \a create_counter)
    HPX_API_EXPORT counter_status remove_counter(
        counter_info const& info, naming::id_type const& id, 
        error_code& ec = throws);

    /// \brief Get the global id of an existing performance counter
    HPX_API_EXPORT naming::id_type get_counter(std::string name,
        error_code& ec = throws);
}}

#endif

