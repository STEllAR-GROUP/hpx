// HPXPy SYCL Bindings
//
// SPDX-License-Identifier: BSL-1.0
// Provides GPU acceleration through HPX's SYCL executor

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifdef HPXPY_HAVE_SYCL

#include <hpx/async_sycl/sycl_executor.hpp>
#include <hpx/async_sycl/sycl_future.hpp>
#include <hpx/async_sycl/sycl_polling_helper.hpp>
#include <hpx/future.hpp>

#include <sycl/sycl.hpp>

#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <sstream>

namespace py = pybind11;

namespace hpxpy {

// --------------------------------------------------------------------------
// SYCL Polling Manager - Required for HPX SYCL futures to resolve
// --------------------------------------------------------------------------

class SYCLPollingManager {
public:
    static SYCLPollingManager& instance() {
        static SYCLPollingManager inst;
        return inst;
    }

    void enable(std::string const& pool_name = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!polling_) {
            if (pool_name.empty()) {
                polling_ = std::make_unique<
                    hpx::sycl::experimental::enable_user_polling>();
            } else {
                polling_ = std::make_unique<
                    hpx::sycl::experimental::enable_user_polling>(pool_name);
            }
        }
    }

    void disable() {
        std::lock_guard<std::mutex> lock(mutex_);
        polling_.reset();
    }

    bool is_enabled() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return polling_ != nullptr;
    }

private:
    SYCLPollingManager() = default;
    SYCLPollingManager(const SYCLPollingManager&) = delete;
    SYCLPollingManager& operator=(const SYCLPollingManager&) = delete;

    std::unique_ptr<hpx::sycl::experimental::enable_user_polling> polling_;
    mutable std::mutex mutex_;
};

// --------------------------------------------------------------------------
// SYCL Executor Manager - Per-device executors for async operations
// --------------------------------------------------------------------------

class SYCLExecutorManager {
public:
    static SYCLExecutorManager& instance() {
        static SYCLExecutorManager inst;
        return inst;
    }

    hpx::sycl::experimental::sycl_executor& get_executor(int device_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = executors_.find(device_id);
        if (it == executors_.end()) {
            // Get SYCL devices and select by index
            auto devices = ::sycl::device::get_devices(::sycl::info::device_type::gpu);
            if (devices.empty()) {
                // Fall back to all devices if no GPU
                devices = ::sycl::device::get_devices();
            }

            if (device_id < 0 || static_cast<size_t>(device_id) >= devices.size()) {
                throw std::runtime_error("Invalid SYCL device ID: " +
                                         std::to_string(device_id));
            }

            // Create executor with the selected device
            auto [iter, inserted] = executors_.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(device_id),
                std::forward_as_tuple(devices[device_id]));
            return iter->second;
        }
        return it->second;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        executors_.clear();
    }

private:
    SYCLExecutorManager() = default;
    SYCLExecutorManager(const SYCLExecutorManager&) = delete;
    SYCLExecutorManager& operator=(const SYCLExecutorManager&) = delete;

    std::unordered_map<int, hpx::sycl::experimental::sycl_executor> executors_;
    std::mutex mutex_;
};

// --------------------------------------------------------------------------
// PyFuture - Wrapper to expose hpx::future<T> to Python
// --------------------------------------------------------------------------

template<typename T>
class PySYCLFuture {
public:
    explicit PySYCLFuture(hpx::future<T> fut)
        : future_(std::move(fut))
    {}

    PySYCLFuture(PySYCLFuture&& other) noexcept
        : future_(std::move(other.future_))
    {}

    PySYCLFuture& operator=(PySYCLFuture&& other) noexcept {
        future_ = std::move(other.future_);
        return *this;
    }

    PySYCLFuture(const PySYCLFuture&) = delete;
    PySYCLFuture& operator=(const PySYCLFuture&) = delete;

    T get() {
        py::gil_scoped_release release;
        return future_.get();
    }

    void wait() {
        py::gil_scoped_release release;
        future_.wait();
    }

    bool is_ready() const {
        return future_.is_ready();
    }

    bool valid() const {
        return future_.valid();
    }

private:
    hpx::future<T> future_;
};

// Specialization for void futures
template<>
class PySYCLFuture<void> {
public:
    explicit PySYCLFuture(hpx::future<void> fut)
        : future_(std::move(fut))
    {}

    PySYCLFuture(PySYCLFuture&& other) noexcept
        : future_(std::move(other.future_))
    {}

    PySYCLFuture& operator=(PySYCLFuture&& other) noexcept {
        future_ = std::move(other.future_);
        return *this;
    }

    PySYCLFuture(const PySYCLFuture&) = delete;
    PySYCLFuture& operator=(const PySYCLFuture&) = delete;

    void get() {
        py::gil_scoped_release release;
        future_.get();
    }

    void wait() {
        py::gil_scoped_release release;
        future_.wait();
    }

    bool is_ready() const {
        return future_.is_ready();
    }

    bool valid() const {
        return future_.valid();
    }

private:
    hpx::future<void> future_;
};

// --------------------------------------------------------------------------
// SYCL Device Information
// --------------------------------------------------------------------------

struct SYCLDevice {
    int id;
    std::string name;
    std::string vendor;
    std::string driver_version;
    std::size_t global_mem_size;
    std::size_t local_mem_size;
    std::uint32_t max_compute_units;
    std::uint32_t max_work_group_size;
    bool is_gpu;
    bool is_cpu;
    std::string backend;

    double global_mem_size_gb() const {
        return static_cast<double>(global_mem_size) / (1024.0 * 1024.0 * 1024.0);
    }
};

SYCLDevice get_device_info(::sycl::device const& dev, int id) {
    SYCLDevice info;
    info.id = id;
    info.name = dev.get_info<::sycl::info::device::name>();
    info.vendor = dev.get_info<::sycl::info::device::vendor>();
    info.driver_version = dev.get_info<::sycl::info::device::driver_version>();
    info.global_mem_size = dev.get_info<::sycl::info::device::global_mem_size>();
    info.local_mem_size = dev.get_info<::sycl::info::device::local_mem_size>();
    info.max_compute_units = dev.get_info<::sycl::info::device::max_compute_units>();
    info.max_work_group_size = dev.get_info<::sycl::info::device::max_work_group_size>();
    info.is_gpu = dev.is_gpu();
    info.is_cpu = dev.is_cpu();

    // Determine backend
    auto platform = dev.get_platform();
    auto platform_name = platform.get_info<::sycl::info::platform::name>();
    if (platform_name.find("CUDA") != std::string::npos) {
        info.backend = "CUDA";
    } else if (platform_name.find("HIP") != std::string::npos) {
        info.backend = "HIP";
    } else if (platform_name.find("Level-Zero") != std::string::npos) {
        info.backend = "Level-Zero";
    } else if (platform_name.find("OpenCL") != std::string::npos) {
        info.backend = "OpenCL";
    } else if (platform_name.find("Metal") != std::string::npos) {
        info.backend = "Metal";
    } else {
        info.backend = platform_name;
    }

    return info;
}

int get_device_count() {
    auto devices = ::sycl::device::get_devices(::sycl::info::device_type::gpu);
    return static_cast<int>(devices.size());
}

std::vector<SYCLDevice> get_all_devices() {
    std::vector<SYCLDevice> result;
    auto devices = ::sycl::device::get_devices(::sycl::info::device_type::gpu);

    for (size_t i = 0; i < devices.size(); ++i) {
        result.push_back(get_device_info(devices[i], static_cast<int>(i)));
    }

    return result;
}

// --------------------------------------------------------------------------
// SYCL Array using USM (Unified Shared Memory)
// --------------------------------------------------------------------------

template<typename T>
class SYCLArray {
public:
    SYCLArray(std::vector<py::ssize_t> shape, int device_id = 0)
        : shape_(std::move(shape))
        , device_id_(device_id)
        , data_(nullptr)
    {
        size_ = 1;
        for (auto dim : shape_) {
            size_ *= dim;
        }

        auto& exec = SYCLExecutorManager::instance().get_executor(device_id_);
        auto& queue = exec.get_queue();

        // Allocate device memory using USM
        data_ = ::sycl::malloc_device<T>(size_, queue);
        if (!data_) {
            throw std::runtime_error("Failed to allocate SYCL device memory");
        }
    }

    ~SYCLArray() {
        if (data_) {
            try {
                auto& exec = SYCLExecutorManager::instance().get_executor(device_id_);
                auto& queue = exec.get_queue();
                ::sycl::free(data_, queue);
            } catch (...) {
                // Ignore errors during cleanup
            }
        }
    }

    // Non-copyable
    SYCLArray(const SYCLArray&) = delete;
    SYCLArray& operator=(const SYCLArray&) = delete;

    // Movable
    SYCLArray(SYCLArray&& other) noexcept
        : shape_(std::move(other.shape_))
        , size_(other.size_)
        , device_id_(other.device_id_)
        , data_(other.data_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    SYCLArray& operator=(SYCLArray&& other) noexcept {
        if (this != &other) {
            if (data_) {
                auto& exec = SYCLExecutorManager::instance().get_executor(device_id_);
                ::sycl::free(data_, exec.get_queue());
            }
            shape_ = std::move(other.shape_);
            size_ = other.size_;
            device_id_ = other.device_id_;
            data_ = other.data_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void fill(T value) {
        auto& exec = SYCLExecutorManager::instance().get_executor(device_id_);
        auto& queue = exec.get_queue();

        queue.fill(data_, value, size_).wait();
    }

    void from_numpy(py::array_t<T> arr) {
        if (static_cast<std::size_t>(arr.size()) != size_) {
            throw std::runtime_error("Array size mismatch");
        }

        auto& exec = SYCLExecutorManager::instance().get_executor(device_id_);
        auto& queue = exec.get_queue();

        queue.memcpy(data_, arr.data(), size_ * sizeof(T)).wait();
    }

    py::array_t<T> to_numpy() const {
        py::array_t<T> result(shape_);

        auto& exec = SYCLExecutorManager::instance().get_executor(device_id_);
        auto& queue = exec.get_queue();

        queue.memcpy(result.mutable_data(), data_, size_ * sizeof(T)).wait();

        return result;
    }

    // Async operations using HPX SYCL executor
    PySYCLFuture<void> async_from_numpy(py::array_t<T> arr) {
        if (static_cast<std::size_t>(arr.size()) != size_) {
            throw std::runtime_error("Array size mismatch");
        }

        if (!SYCLPollingManager::instance().is_enabled()) {
            throw std::runtime_error(
                "Async operations require polling to be enabled. "
                "Call hpx.sycl.enable_async() first.");
        }

        auto& exec = SYCLExecutorManager::instance().get_executor(device_id_);
        auto& queue = exec.get_queue();

        // Use queue.memcpy and get HPX future from event
        ::sycl::event event = queue.memcpy(data_, arr.data(), size_ * sizeof(T));
        hpx::future<void> fut = hpx::sycl::experimental::detail::get_future(event);

        return PySYCLFuture<void>(std::move(fut));
    }

    PySYCLFuture<void> async_to_numpy(py::array_t<T>& result) {
        if (static_cast<std::size_t>(result.size()) != size_) {
            throw std::runtime_error("Array size mismatch");
        }

        if (!SYCLPollingManager::instance().is_enabled()) {
            throw std::runtime_error(
                "Async operations require polling to be enabled. "
                "Call hpx.sycl.enable_async() first.");
        }

        auto& exec = SYCLExecutorManager::instance().get_executor(device_id_);
        auto& queue = exec.get_queue();

        ::sycl::event event = queue.memcpy(result.mutable_data(), data_, size_ * sizeof(T));
        hpx::future<void> fut = hpx::sycl::experimental::detail::get_future(event);

        return PySYCLFuture<void>(std::move(fut));
    }

    // Accessors
    std::vector<py::ssize_t> const& shape() const { return shape_; }
    py::ssize_t size() const { return size_; }
    py::ssize_t ndim() const { return shape_.size(); }
    int device_id() const { return device_id_; }
    T* data() { return data_; }
    T const* data() const { return data_; }

private:
    std::vector<py::ssize_t> shape_;
    py::ssize_t size_;
    int device_id_;
    T* data_;
};

// --------------------------------------------------------------------------
// Array factory functions
// --------------------------------------------------------------------------

template<typename T>
SYCLArray<T> sycl_zeros(std::vector<py::ssize_t> shape, int device = 0) {
    SYCLArray<T> arr(std::move(shape), device);
    arr.fill(T{0});
    return arr;
}

template<typename T>
SYCLArray<T> sycl_ones(std::vector<py::ssize_t> shape, int device = 0) {
    SYCLArray<T> arr(std::move(shape), device);
    arr.fill(T{1});
    return arr;
}

template<typename T>
SYCLArray<T> sycl_full(std::vector<py::ssize_t> shape, T value, int device = 0) {
    SYCLArray<T> arr(std::move(shape), device);
    arr.fill(value);
    return arr;
}

template<typename T>
SYCLArray<T> sycl_from_numpy(py::array_t<T> arr, int device = 0) {
    std::vector<py::ssize_t> shape(arr.ndim());
    for (py::ssize_t i = 0; i < arr.ndim(); ++i) {
        shape[i] = arr.shape(i);
    }
    SYCLArray<T> result(shape, device);
    result.from_numpy(arr);
    return result;
}

// --------------------------------------------------------------------------
// Reduction using SYCL
// --------------------------------------------------------------------------

template<typename T>
T sycl_sum(SYCLArray<T> const& arr) {
    // For now, copy to host and sum there
    // Future: use SYCL reduction
    auto np_arr = arr.to_numpy();
    T sum = T{0};
    for (py::ssize_t i = 0; i < arr.size(); ++i) {
        sum += np_arr.data()[i];
    }
    return sum;
}

// --------------------------------------------------------------------------
// Python bindings helper
// --------------------------------------------------------------------------

template<typename T>
void bind_sycl_array(py::module_& m, const char* name) {
    using SA = SYCLArray<T>;

    py::class_<SA>(m, name, py::buffer_protocol())
        .def_property_readonly("shape", &SA::shape, "Shape of the array")
        .def_property_readonly("size", &SA::size, "Total number of elements")
        .def_property_readonly("ndim", &SA::ndim, "Number of dimensions")
        .def_property_readonly("device", &SA::device_id, "SYCL device ID")

        .def("fill", &SA::fill, py::arg("value"), "Fill array with a value")
        .def("from_numpy", &SA::from_numpy, py::arg("arr"), "Copy data from numpy array")
        .def("to_numpy", &SA::to_numpy, "Copy data to numpy array")

        // Async operations
        .def("async_from_numpy", &SA::async_from_numpy, py::arg("arr"),
             "Async copy from numpy (requires enable_async)")

        .def("__repr__", [name](SA const& arr) {
            std::ostringstream oss;
            oss << name << "(shape=[";
            auto const& shape = arr.shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << shape[i];
            }
            oss << "], device=" << arr.device_id() << ")";
            return oss.str();
        });
}

}  // namespace hpxpy

void bind_sycl(py::module_& m) {
    using namespace hpxpy;

    auto sycl = m.def_submodule("sycl", "SYCL GPU support for HPXPy (via HPX sycl_executor)");

    // --------------------------------------------------------------------------
    // Future type
    // --------------------------------------------------------------------------

    py::class_<PySYCLFuture<void>>(sycl, "Future",
        "Future for async SYCL operations")
        .def("get", &PySYCLFuture<void>::get,
             "Wait for completion (releases Python GIL)")
        .def("wait", &PySYCLFuture<void>::wait,
             "Wait for completion (releases Python GIL)")
        .def("is_ready", &PySYCLFuture<void>::is_ready,
             "Check if completed (non-blocking)")
        .def("valid", &PySYCLFuture<void>::valid,
             "Check if future has shared state");

    // --------------------------------------------------------------------------
    // Device info
    // --------------------------------------------------------------------------

    py::class_<SYCLDevice>(sycl, "Device", "SYCL device information")
        .def_readonly("id", &SYCLDevice::id, "Device ID")
        .def_readonly("name", &SYCLDevice::name, "Device name")
        .def_readonly("vendor", &SYCLDevice::vendor, "Device vendor")
        .def_readonly("driver_version", &SYCLDevice::driver_version, "Driver version")
        .def_readonly("global_mem_size", &SYCLDevice::global_mem_size, "Global memory (bytes)")
        .def_readonly("local_mem_size", &SYCLDevice::local_mem_size, "Local memory (bytes)")
        .def_readonly("max_compute_units", &SYCLDevice::max_compute_units, "Compute units")
        .def_readonly("max_work_group_size", &SYCLDevice::max_work_group_size, "Max work group size")
        .def_readonly("is_gpu", &SYCLDevice::is_gpu, "True if GPU device")
        .def_readonly("is_cpu", &SYCLDevice::is_cpu, "True if CPU device")
        .def_readonly("backend", &SYCLDevice::backend, "SYCL backend (CUDA, HIP, Level-Zero, Metal, OpenCL)")
        .def("global_mem_size_gb", &SYCLDevice::global_mem_size_gb, "Global memory in GB")
        .def("__repr__", [](SYCLDevice const& d) {
            return "SYCLDevice(" + std::to_string(d.id) + ", '" + d.name +
                   "', backend='" + d.backend + "')";
        });

    // --------------------------------------------------------------------------
    // Device queries
    // --------------------------------------------------------------------------

    sycl.def("is_available", []() {
        return get_device_count() > 0;
    }, "Check if SYCL GPU devices are available");

    sycl.def("device_count", &get_device_count,
             "Get the number of available SYCL GPU devices");

    sycl.def("get_devices", &get_all_devices,
             "Get list of all SYCL GPU devices");

    sycl.def("get_device", [](int id) {
        auto devices = ::sycl::device::get_devices(::sycl::info::device_type::gpu);
        if (id < 0 || static_cast<size_t>(id) >= devices.size()) {
            throw std::runtime_error("Invalid device ID");
        }
        return get_device_info(devices[id], id);
    }, py::arg("device") = 0, "Get info for a specific SYCL device");

    // --------------------------------------------------------------------------
    // Async operations control
    // --------------------------------------------------------------------------

    sycl.def("enable_async", [](std::string const& pool_name) {
        SYCLPollingManager::instance().enable(pool_name);
    }, py::arg("pool_name") = "",
       "Enable async SYCL operations (starts HPX polling)");

    sycl.def("disable_async", []() {
        SYCLPollingManager::instance().disable();
    }, "Disable async SYCL operations");

    sycl.def("is_async_enabled", []() {
        return SYCLPollingManager::instance().is_enabled();
    }, "Check if async SYCL operations are enabled");

    // --------------------------------------------------------------------------
    // Array types
    // --------------------------------------------------------------------------

    bind_sycl_array<double>(sycl, "ArrayF64");
    bind_sycl_array<float>(sycl, "ArrayF32");
    bind_sycl_array<std::int64_t>(sycl, "ArrayI64");
    bind_sycl_array<std::int32_t>(sycl, "ArrayI32");

    // --------------------------------------------------------------------------
    // Array creation functions
    // --------------------------------------------------------------------------

    sycl.def("zeros", &sycl_zeros<double>,
             py::arg("shape"), py::arg("device") = 0,
             "Create a zero-filled SYCL array");

    sycl.def("ones", &sycl_ones<double>,
             py::arg("shape"), py::arg("device") = 0,
             "Create a one-filled SYCL array");

    sycl.def("full", &sycl_full<double>,
             py::arg("shape"), py::arg("value"), py::arg("device") = 0,
             "Create a SYCL array filled with a value");

    sycl.def("from_numpy", [](py::array arr, int device) {
        auto np_arr = py::array_t<double>::ensure(arr);
        if (!np_arr) {
            throw std::runtime_error("Array must be convertible to float64");
        }
        return sycl_from_numpy<double>(np_arr, device);
    }, py::arg("arr"), py::arg("device") = 0,
       "Create SYCL array from numpy array");

    // --------------------------------------------------------------------------
    // Reductions
    // --------------------------------------------------------------------------

    sycl.def("sum", &sycl_sum<double>,
             py::arg("arr"), "Sum all elements of a SYCL array");
}

#else  // No SYCL support

#include <pybind11/pybind11.h>
namespace py = pybind11;

void bind_sycl(py::module_& m) {
    auto sycl = m.def_submodule("sycl", "SYCL GPU support (not available)");

    sycl.def("is_available", []() { return false; },
             "Check if SYCL is available");

    sycl.def("device_count", []() { return 0; },
             "Get number of SYCL devices");

    sycl.def("get_devices", []() { return py::list(); },
             "Get list of SYCL devices");

    sycl.def("enable_async", [](std::string const&) {},
             py::arg("pool_name") = "",
             "Enable async SYCL operations (no-op without SYCL)");

    sycl.def("disable_async", []() {},
             "Disable async SYCL operations (no-op without SYCL)");

    sycl.def("is_async_enabled", []() { return false; },
             "Check if async SYCL operations are enabled");
}

#endif  // HPXPY_HAVE_SYCL
