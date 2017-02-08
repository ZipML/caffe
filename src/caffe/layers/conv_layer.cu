#include <vector>

#include "caffe/util/quantize.cuh"
#include "caffe/layers/conv_layer.hpp"
namespace caffe {

template <typename Dtype>
__global__ void print(const Dtype * a, int n){
  printf("----------------\n");
  for(int i=0;i<n;i++){
    printf("%d: %f\n", i, a[i]);
  }
  printf("----------------\n");
}


template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* oriweight = this->blobs_[0]->gpu_data(); // TODO: Backward quantization is not useful at all, reuse forward

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
  //const Dtype* weight = this->blobs_[0]->gpu_data();
  
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
          postprocess_gradient(weight, weight_diff, this->blobs_[0]->count(), this->nlevels, this->is_optimal, this->usage_counter);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
