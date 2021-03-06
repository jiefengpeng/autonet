#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;


REGISTER_OP("BinaryOut")
      .Input("to_binary: float")
      .Output("binaryed: float")
      .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
				          c->set_output(0, c->input(0));
					  return Status::OK();
					});



class BinaryOutOp : public OpKernel {
 public:
  explicit BinaryOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // 获取输入 tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // 创建输出 tensor, context->allocate_output 用来分配输出内存？
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    // 执行计算操作。
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      if (input(i) >= 0.) {output_flat(i) = 1.;}
      else {output_flat(i) = 0.;}
    }

    // Preserve the first input value if possible.
    // if (N > 0) output_flat(0) = input(0);
  }
};
REGISTER_KERNEL_BUILDER(Name("BinaryOut").Device(DEVICE_CPU), BinaryOutOp);



