#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <vector>

#include "caffe/layers/memory_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void MemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  this->batch_size_ = this->layer_param_.memory_data_param().batch_size();
  this->channels_ = this->layer_param_.memory_data_param().channels();
  this->height_ = this->layer_param_.memory_data_param().height();
  this->width_ = this->layer_param_.memory_data_param().width();
  // multi-label
  this->dim_label_ = this->layer_param_.memory_data_param().dim_label();
  this->size_ = this->channels_ * this->height_ * this->width_;
  CHECK_GT(this->batch_size_ * this->size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  // multi-label
  // vector<int> label_shape(1, this->batch_size_);
  vector<int> label_shape(this->dim_label_, this->batch_size_);
  top[0]->Reshape(batch_size_, channels_, height_, width_);
  top[1]->Reshape(label_shape);
  this->added_data_.Reshape(batch_size_, channels_, height_, width_);
  this->added_label_.Reshape(label_shape);
  this->data_ = NULL;
  this->labels_ = NULL;
  this->added_data_.cpu_data();
  this->added_label_.cpu_data();
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::AddDatumVector(const vector<Datum>& datum_vector) {
  CHECK(!has_new_data_) <<
      "Can't add data until current data has been consumed.";
  size_t num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to add.";
  CHECK_EQ(num % this->batch_size_, 0) <<
      "The added data must be a multiple of the batch size.";
  added_data_.Reshape(num, this->channels_, this->height_, this->width_);
  // multi-label
  // added_label_.Reshape(num, 1, 1, 1);
  added_label_.Reshape(num, this->dim_label_, 1, 1);
  // Apply data transformations (mirror, scale, crop...)
  this->data_transformer_->Transform(datum_vector, &added_data_);
  // Copy Labels
  Dtype* top_label = this->added_label_.mutable_cpu_data();
  // for (int item_id = 0; item_id < num; ++item_id) {
  //   top_label[item_id] = datum_vector[item_id].label();
  // }

  // multi-label
  for (int item_id = 0; item_id < num; ++item_id) {
    for(int label_id = 0; label_id < this->dim_label_; ++label_id) {
      top_label[item_id * this->dim_label_ + label_id] = datum_vector[item_id].label(label_id);
    }
  }
  // num_images == batch_size_
  Dtype* top_data = this->added_data_.mutable_cpu_data();
  Reset(top_data, top_label, num);
  this->has_new_data_ = true;
}

#ifdef USE_OPENCV
template <typename Dtype>
void MemoryDataLayer<Dtype>::AddMatVector(const vector<cv::Mat>& mat_vector,
    const vector<vector<int> >& labels) {
  size_t num = mat_vector.size();
  CHECK(!has_new_data_) <<
      "Can't add mat until current data has been consumed.";
  CHECK_GT(num, 0) << "There is no mat to add";
  CHECK_EQ(num % this->batch_size_, 0) <<
      "The added data must be a multiple of the batch size.";
  added_data_.Reshape(num, this->channels_, this->height_, this->width_);
  // added_label_.Reshape(num, 1, 1, 1);
  added_label_.Reshape(num, this->dim_label_, 1, 1);
  // Apply data transformations (mirror, scale, crop...)
  this->data_transformer_->Transform(mat_vector, &added_data_);
  // Copy Labels
  Dtype* top_label = this->added_label_.mutable_cpu_data();
  // for (int item_id = 0; item_id < num; ++item_id) {
  //   top_label[item_id] = labels[item_id];
  // }

  // multi-label
  for (int item_id = 0; item_id < num; ++item_id) {
    for(int label_id = 0; label_id < this->dim_label_; ++label_id) {
      top_label[item_id * this->dim_label_ + label_id] = labels[item_id][label_id];
    }
  }

  // num_images == batch_size_
  Dtype* top_data = this->added_data_.mutable_cpu_data();
  Reset(top_data, top_label, num);
  this->has_new_data_ = true;
}
#endif  // USE_OPENCV

template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
  CHECK(data);
  CHECK(labels);
  CHECK_EQ(n % this->batch_size_, 0) << "n must be a multiple of batch size";
  // Warn with transformation parameters since a memory array is meant to
  // be generic and no transformations are done with Reset().
  if (this->layer_param_.has_transform_param()) {
    LOG(WARNING) << this->type() << " does not transform array data on Reset()";
  }
  this->data_ = data;
  this->labels_ = labels;
  this->n_ = n;
  this->pos_ = 0;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::set_batch_size(int new_size) {
  CHECK(!has_new_data_) <<
      "Can't change batch_size until current data has been consumed.";
  this->batch_size_ = new_size;
  this->added_data_.Reshape(this->batch_size_, this->channels_, this->height_, this->width_);
  //added_label_.Reshape(batch_size_, 1, 1, 1);
  // multi-label
  this->added_label_.Reshape(this->batch_size_, this->dim_label_, 1, 1);
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->data_) << "MemoryDataLayer needs to be initialized by calling Reset";
  top[0]->Reshape(this->batch_size_, this->channels_, this->height_, this->width_);
  //top[1]->Reshape(batch_size_, 1, 1, 1);
  // multi-label
  top[1]->Reshape(this->batch_size_, this->dim_label_, 1, 1);
  top[0]->set_cpu_data(this->data_ + this->pos_ * this->size_);
  //top[1]->set_cpu_data(labels_ + pos_);
  // muti-label
  top[1]->set_cpu_data(this->labels_ + this->pos_ * this->dim_label_);
  this->pos_ = (this->pos_ + this->batch_size_) % this->n_;
  if (this->pos_ == 0)
    this->has_new_data_ = false;
}

INSTANTIATE_CLASS(MemoryDataLayer);
REGISTER_LAYER_CLASS(MemoryData);

}  // namespace caffe
