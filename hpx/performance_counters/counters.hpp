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

    ///////////////////////////////////////////////////////////////////////////
    struct counter_info
    {
        counter_info(counter_type type = counter_raw)
          : type_(type), version_(0x01000000), status_(status_valid_data)
        {}

        counter_info(counter_type type, std::string const& name, 
          std::string const& helptext, boost::uint32_t version = 0x01000000)
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

    private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & status_ & time_ & value_ & scaling_ & scale_inverse_;
        }
    };
}}

#endif

