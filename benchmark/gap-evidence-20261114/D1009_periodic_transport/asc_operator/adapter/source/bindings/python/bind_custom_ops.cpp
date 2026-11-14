#include <asnumpy/custom_ops/roll.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace asnumpy {
void bind_custom_ops(py::module_& module) {
    module.def("roll", &NumpyRoll, py::arg("a").noconvert(), py::arg("shift"), py::arg("axis") = 1,
               "NumPy-compatible asnumpy.roll on the NPU.");
}
} // namespace asnumpy

void bind_custom_ops(py::module_& module) {
    asnumpy::bind_custom_ops(module);
}
