#ifndef AUTOENCODER_NETWORK_H
#define AUTOENCODER_NETWORK_H

#include "network.h"

//namespace tiny_cnn {

class autoencoder_network : public tiny_cnn::network<tiny_cnn::mse, tiny_cnn::gradient_descent> {
    public:
        typedef tiny_cnn::network<tiny_cnn::mse, tiny_cnn::gradient_descent> CNN;
        typedef tiny_cnn::AE_layer<CNN, tiny_cnn::activation::tan_h> AEL;

        autoencoder_network() {}

        virtual ~autoencoder_network() {}

        /// First training
        template <typename OnBatchEnumerate, typename OnEpochEnumerate>
        void train(const std::vector<tiny_cnn::vec_t>& in, size_t batch_size, int epoch, OnBatchEnumerate on_batch_enumerate, OnEpochEnumerate on_epoch_enumerate) {
            init_weight();
            layers_.set_parallelize(batch_size < CNN_TASK_SIZE);

            for (int iter = 0; iter < epoch; iter++) {
                if (optimizer_.requires_hessian())
                    calc_hessian(in);
                for (size_t i = 0; i < in.size(); i+=batch_size) {
                    train_once(&in[i], &in[i], std::min(batch_size, in.size() - i));
                    on_batch_enumerate();
                }
                on_epoch_enumerate();
            }
        }

        /// Continued training
        template <typename OnBatchEnumerate, typename OnEpochEnumerate>
        void train_next(const std::vector<tiny_cnn::vec_t>& in, size_t batch_size, int epoch, OnBatchEnumerate on_batch_enumerate, OnEpochEnumerate on_epoch_enumerate) {
            layers_.set_parallelize(batch_size < CNN_TASK_SIZE);

            for (int iter = 0; iter < epoch; iter++) {
                if (optimizer_.requires_hessian())
                    calc_hessian(in);
                for (size_t i = 0; i < in.size(); i+=batch_size) {
                    train_once(&in[i], &in[i], std::min(batch_size, in.size() - i));
                    on_batch_enumerate();
                }
                on_epoch_enumerate();
            }
        }

        tiny_cnn::vec_t encode(const tiny_cnn::vec_t& in){
            forward_propagation(in);
            return get_last_encoded();
        }

        tiny_cnn::vec_t get_last_encoded(){
            return last_encoder_->encoder_output(0); // worker zero, output is duplicated for each parallel propagation of the batch
        }
    protected:
        AEL* last_encoder_;
    private:
};

//} // namespace tiny_cnn

#endif // AUTOENCODER_NETWORK_H
