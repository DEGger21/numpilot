#include <asnumpy/custom_ops/nextafter.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace asnumpy {
void bind_custom_ops(py::module_& module) {
    module.def("nextafter", &NumpyNextafter, py::arg("a").noconvert(), py::arg("b").noconvert(),
               "NumPy-compatible asnumpy.nextafter on the NPU.");
}
} // namespace asnumpy

void bind_custom_ops(py::module_& module) {
    asnumpy::bind_custom_ops(module);
}
