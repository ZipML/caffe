#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/util/quantize.cuh"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* oriweight = this->blobs_[0]->gpu_data();

  this->usage_counter ++;
  if(this->qblob == NULL){
    cudaMalloc(&this->qblob, this->blobs_[0]->count() * sizeof(Dtype));
    cudaMalloc(&this->scratch, 128 * sizeof(Dtype));
    cudaMalloc(&this->ql1, 1 * sizeof(Dtype));
    cudaMalloc(&this->qmax, 1 * sizeof(Dtype));
    cudaMalloc(&this->qmin, 1 * sizeof(Dtype));
    cudaMalloc(&this->qzipml_k, 1 * sizeof(Dtype));
  }
  
  const Dtype * weight;
  if(this->is_quantize_model){
    quantize<Dtype>(oriweight, this->qblob, this->blobs_[0]->count(), this->nlevels, this->is_optimal, this->usage_counter, this->scratch, this->ql1, this->qmax, this->qmin, this->qzipml_k);
    weight = this->qblob; 
  }else{
    weight = oriweight;
  }

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  //const Dtype* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* oriweight = this->blobs_[0]->gpu_data();  // TODO: Backward quantization is not useful at all, reuse forward

  if(this->qblob == NULL){
    cudaMalloc(&this->qblob, this->blobs_[0]->count() * sizeof(Dtype));
    cudaMalloc(&this->scratch, 128 * sizeof(Dtype));
    cudaMalloc(&this->ql1, 1 * sizeof(Dtype));
    cudaMalloc(&this->qmax, 1 * sizeof(Dtype));
    cudaMalloc(&this->qmin, 1 * sizeof(Dtype));
    cudaMalloc(&this->qzipml_k, 1 * sizeof(Dtype));
  }
  
  const Dtype * weight;
  if(this->is_quantize_model){
    quantize<Dtype>(oriweight, this->qblob, this->blobs_[0]->count(), this->nlevels, this->is_optimal, this->usage_counter, this->scratch, this->ql1, this->qmax, this->qmin, this->qzipml_k);
    weight = this->qblob; 
  }else{
    weight = oriweight;
  }

  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
      postprocess_gradient(weight, this->blobs_[0]->mutable_gpu_diff(), this->blobs_[0]->count(), this->nlevels, this->is_optimal, this->usage_counter);
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
      postprocess_gradient(weight, this->blobs_[0]->mutable_gpu_diff(), this->blobs_[0]->count(), this->nlevels, this->is_optimal, this->usage_counter);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, weight,
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, weight,
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
