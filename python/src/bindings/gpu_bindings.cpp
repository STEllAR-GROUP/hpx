// HPXPy - GPU Bindings
//
// SPDX-License-Identifier: BSL-1.0
//
// Python bindings for HPX GPU/CUDA support.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifdef HPXPY_HAVE_CUDA

#include <hpx/async_cuda/cuda_executor.hpp>
#include <hpx/async_cuda/target.hpp>
#include <hpx/async_cuda/get_targets.hpp>
#include <hpx/async_cuda/cuda_exception.hpp>

#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <memory>

namespace py = pybind11;

namespace hpxpy {

// GPU device information struct
struct GPUDevice {
    int id;
    std::string name;
    std::size_t total_memory;
    std::size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int warp_size;

    static GPUDevice from_device_id(int device_id) {
        GPUDevice dev;
        dev.id = device_id;

        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("Failed to get device properties: ") +
                cudaGetErrorString(err));
        }

        dev.name = prop.name;
        dev.total_memory = prop.totalGlobalMem;
        dev.compute_capability_major = prop.major;
        dev.compute_capability_minor = prop.minor;
        dev.multiprocessor_count = prop.multiProcessorCount;
        dev.max_threads_per_block = prop.maxThreadsPerBlock;
        dev.warp_size = prop.warpSize;

        // Get free memory
        std::size_t free_mem, total_mem;
        int old_device;
        cudaGetDevice(&old_device);
        cudaSetDevice(device_id);
        cudaMemGetInfo(&free_mem, &total_mem);
        cudaSetDevice(old_device);
        dev.free_memory = free_mem;

        return dev;
    }

    std::string compute_capability() const {
        return std::to_string(compute_capability_major) + "." +
               std::to_string(compute_capability_minor);
    }

    double total_memory_gb() const {
        return static_cast<double>(total_memory) / (1024.0 * 1024.0 * 1024.0);
    }

    double free_memory_gb() const {
        return static_cast<double>(free_memory) / (1024.0 * 1024.0 * 1024.0);
    }
};

// GPU array class - holds data on GPU
template<typename T>
class GPUArray {
public:
    GPUArray(std::vector<py::ssize_t> shape, int device_id = 0)
        : shape_(std::move(shape)), device_id_(device_id), data_(nullptr)
    {
        size_ = 1;
        for (auto dim : shape_) {
            size_ *= dim;
        }

        // Allocate on GPU
        int old_device;
        cudaGetDevice(&old_device);
        cudaSetDevice(device_id_);

        cudaError_t err = cudaMalloc(&data_, size_ * sizeof(T));
        if (err != cudaSuccess) {
            cudaSetDevice(old_device);
            throw std::runtime_error(
                std::string("Failed to allocate GPU memory: ") +
                cudaGetErrorString(err));
        }

        cudaSetDevice(old_device);
    }

    ~GPUArray() {
        if (data_) {
            int old_device;
            cudaGetDevice(&old_device);
            cudaSetDevice(device_id_);
            cudaFree(data_);
            cudaSetDevice(old_device);
        }
    }

    // Copy constructor
    GPUArray(const GPUArray& other)
        : shape_(other.shape_), size_(other.size_), device_id_(other.device_id_), data_(nullptr)
    {
        int old_device;
        cudaGetDevice(&old_device);
        cudaSetDevice(device_id_);

        cudaError_t err = cudaMalloc(&data_, size_ * sizeof(T));
        if (err != cudaSuccess) {
            cudaSetDevice(old_device);
            throw std::runtime_error("Failed to allocate GPU memory for copy");
        }

        err = cudaMemcpy(data_, other.data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            cudaFree(data_);
            cudaSetDevice(old_device);
            throw std::runtime_error("Failed to copy GPU memory");
        }

        cudaSetDevice(old_device);
    }

    // Move constructor
    GPUArray(GPUArray&& other) noexcept
        : shape_(std::move(other.shape_)), size_(other.size_),
          device_id_(other.device_id_), data_(other.data_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // Prevent assignment
    GPUArray& operator=(const GPUArray&) = delete;
    GPUArray& operator=(GPUArray&&) = delete;

    // Fill with value
    void fill(T value) {
        std::vector<T> host_data(size_, value);

        int old_device;
        cudaGetDevice(&old_device);
        cudaSetDevice(device_id_);

        cudaError_t err = cudaMemcpy(data_, host_data.data(),
                                      size_ * sizeof(T), cudaMemcpyHostToDevice);
        cudaSetDevice(old_device);

        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to fill GPU array");
        }
    }

    // Copy from numpy array
    void from_numpy(py::array_t<T> arr) {
        if (static_cast<std::size_t>(arr.size()) != size_) {
            throw std::runtime_error("Array size mismatch");
        }

        int old_device;
        cudaGetDevice(&old_device);
        cudaSetDevice(device_id_);

        cudaError_t err = cudaMemcpy(data_, arr.data(),
                                      size_ * sizeof(T), cudaMemcpyHostToDevice);
        cudaSetDevice(old_device);

        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy to GPU");
        }
    }

    // Copy to numpy array
    py::array_t<T> to_numpy() const {
        py::array_t<T> result(shape_);

        int old_device;
        cudaGetDevice(&old_device);
        cudaSetDevice(device_id_);

        cudaError_t err = cudaMemcpy(result.mutable_data(), data_,
                                      size_ * sizeof(T), cudaMemcpyDeviceToHost);
        cudaSetDevice(old_device);

        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy from GPU");
        }

        return result;
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

// GPU utility functions
int get_device_count() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
}

std::vector<GPUDevice> get_all_devices() {
    std::vector<GPUDevice> devices;
    int count = get_device_count();
    for (int i = 0; i < count; ++i) {
        devices.push_back(GPUDevice::from_device_id(i));
    }
    return devices;
}

int get_current_device() {
    int device;
    cudaGetDevice(&device);
    return device;
}

void set_current_device(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to set device: ") + cudaGetErrorString(err));
    }
}

void synchronize_device(int device_id = -1) {
    int old_device;
    cudaGetDevice(&old_device);

    if (device_id >= 0) {
        cudaSetDevice(device_id);
    }

    cudaError_t err = cudaDeviceSynchronize();

    if (device_id >= 0) {
        cudaSetDevice(old_device);
    }

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Device synchronization failed: ") + cudaGetErrorString(err));
    }
}

// GPU sum reduction using thrust (if available) or simple implementation
template<typename T>
T gpu_sum(GPUArray<T>& arr) {
    // For now, copy to host and compute there
    // TODO: Use thrust or custom kernel for GPU-native reduction
    auto host = arr.to_numpy();
    T sum = 0;
    for (py::ssize_t i = 0; i < arr.size(); ++i) {
        sum += host.data()[i];
    }
    return sum;
}

// Bind GPU array type
template<typename T>
void bind_gpu_array_type(py::module_& m, const char* name) {
    using GA = GPUArray<T>;

    py::class_<GA>(m, name)
        .def(py::init<std::vector<py::ssize_t>, int>(),
             py::arg("shape"),
             py::arg("device") = 0,
             "Create a GPU array with given shape on specified device")

        .def_property_readonly("shape", &GA::shape, "Shape of the array")
        .def_property_readonly("size", &GA::size, "Total number of elements")
        .def_property_readonly("ndim", &GA::ndim, "Number of dimensions")
        .def_property_readonly("device", &GA::device_id, "GPU device ID")

        .def("fill", &GA::fill, py::arg("value"), "Fill array with a value")
        .def("from_numpy", &GA::from_numpy, py::arg("arr"), "Copy data from numpy array")
        .def("to_numpy", &GA::to_numpy, "Copy data to numpy array")

        .def("__repr__", [name](GA const& arr) {
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

void bind_gpu(py::module_& m) {
    using namespace hpxpy;

    // Create GPU submodule
    auto gpu = m.def_submodule("gpu", "GPU/CUDA support for HPXPy");

    // GPU device info class
    py::class_<GPUDevice>(gpu, "Device", "GPU device information")
        .def_readonly("id", &GPUDevice::id, "Device ID")
        .def_readonly("name", &GPUDevice::name, "Device name")
        .def_readonly("total_memory", &GPUDevice::total_memory, "Total memory (bytes)")
        .def_readonly("free_memory", &GPUDevice::free_memory, "Free memory (bytes)")
        .def_readonly("compute_capability_major", &GPUDevice::compute_capability_major)
        .def_readonly("compute_capability_minor", &GPUDevice::compute_capability_minor)
        .def_readonly("multiprocessor_count", &GPUDevice::multiprocessor_count)
        .def_readonly("max_threads_per_block", &GPUDevice::max_threads_per_block)
        .def_readonly("warp_size", &GPUDevice::warp_size)
        .def("compute_capability", &GPUDevice::compute_capability,
             "Get compute capability as string (e.g., '8.6')")
        .def("total_memory_gb", &GPUDevice::total_memory_gb,
             "Get total memory in GB")
        .def("free_memory_gb", &GPUDevice::free_memory_gb,
             "Get free memory in GB")
        .def("__repr__", [](GPUDevice const& d) {
            std::ostringstream oss;
            oss << "Device(" << d.id << ": " << d.name
                << ", " << d.total_memory_gb() << " GB, CC " << d.compute_capability() << ")";
            return oss.str();
        });

    // GPU utility functions
    gpu.def("device_count", &get_device_count,
            "Get the number of available CUDA devices");

    gpu.def("get_devices", &get_all_devices,
            "Get list of all available CUDA devices");

    gpu.def("get_device", [](int device_id) {
        return GPUDevice::from_device_id(device_id);
    }, py::arg("device") = 0,
       "Get information about a specific CUDA device");

    gpu.def("current_device", &get_current_device,
            "Get the current CUDA device ID");

    gpu.def("set_device", &set_current_device,
            py::arg("device"),
            "Set the current CUDA device");

    gpu.def("synchronize", &synchronize_device,
            py::arg("device") = -1,
            "Synchronize device (default: current device)");

    // GPU array types
    bind_gpu_array_type<double>(gpu, "ArrayF64");
    bind_gpu_array_type<float>(gpu, "ArrayF32");
    bind_gpu_array_type<std::int64_t>(gpu, "ArrayI64");
    bind_gpu_array_type<std::int32_t>(gpu, "ArrayI32");

    // Array creation functions
    gpu.def("zeros", [](std::vector<py::ssize_t> shape, int device) {
        GPUArray<double> arr(shape, device);
        arr.fill(0.0);
        return arr;
    }, py::arg("shape"), py::arg("device") = 0,
       "Create a GPU array filled with zeros");

    gpu.def("ones", [](std::vector<py::ssize_t> shape, int device) {
        GPUArray<double> arr(shape, device);
        arr.fill(1.0);
        return arr;
    }, py::arg("shape"), py::arg("device") = 0,
       "Create a GPU array filled with ones");

    gpu.def("full", [](std::vector<py::ssize_t> shape, double value, int device) {
        GPUArray<double> arr(shape, device);
        arr.fill(value);
        return arr;
    }, py::arg("shape"), py::arg("value"), py::arg("device") = 0,
       "Create a GPU array filled with a value");

    gpu.def("from_numpy", [](py::array_t<double> np_arr, int device) {
        std::vector<py::ssize_t> shape(np_arr.ndim());
        for (py::ssize_t i = 0; i < np_arr.ndim(); ++i) {
            shape[i] = np_arr.shape(i);
        }
        GPUArray<double> arr(shape, device);
        arr.from_numpy(np_arr);
        return arr;
    }, py::arg("arr"), py::arg("device") = 0,
       "Create a GPU array from a numpy array");

    // GPU reduction operations
    gpu.def("sum", [](GPUArray<double>& arr) {
        return gpu_sum(arr);
    }, py::arg("arr"), "Sum all elements of a GPU array");

    // Memory info
    gpu.def("memory_info", [](int device) {
        std::size_t free_mem, total_mem;
        int old_device;
        cudaGetDevice(&old_device);
        cudaSetDevice(device);
        cudaMemGetInfo(&free_mem, &total_mem);
        cudaSetDevice(old_device);
        return py::make_tuple(free_mem, total_mem);
    }, py::arg("device") = 0,
       "Get (free, total) memory in bytes for a device");

    // Check if GPU is available
    gpu.def("is_available", []() {
        return get_device_count() > 0;
    }, "Check if CUDA is available");
}

#else  // No CUDA support

#include <pybind11/pybind11.h>
namespace py = pybind11;

void bind_gpu(py::module_& m) {
    // Create GPU submodule with stub functions
    auto gpu = m.def_submodule("gpu", "GPU/CUDA support (not available)");

    gpu.def("is_available", []() { return false; },
            "Check if CUDA is available");

    gpu.def("device_count", []() { return 0; },
            "Get the number of available CUDA devices");

    gpu.def("get_devices", []() {
        return py::list();
    }, "Get list of all available CUDA devices");
}

#endif  // HPXPY_HAVE_CUDA
