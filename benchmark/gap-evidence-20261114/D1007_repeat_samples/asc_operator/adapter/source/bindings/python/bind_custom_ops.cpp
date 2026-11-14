#include <asnumpy/custom_ops/repeat.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace asnumpy {
void bind_custom_ops(py::module_& module) {
    module.def("repeat", &NumpyRepeat, py::arg("a").noconvert(), py::arg("repeats"), py::arg("axis") = 1,
               "NumPy-compatible asnumpy.repeat on the NPU.");
}
} // namespace asnumpy

void bind_custom_ops(py::module_& module) {
    asnumpy::bind_custom_ops(module);
}
