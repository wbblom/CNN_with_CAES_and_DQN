#ifndef CAES_HPP
#define CAES_HPP

#include "tiny_cnn.h"
#include "util.h"
#include "autoencoder_network.h"
#include "convolutional_AE_layer.h"
#include <stdio.h>

class caes : public autoencoder_network {
    public:
        caes(int in_width, int in_height, int in_channels, int levels, int* window_sizes, int* num_filters){
            printf("New %d-layer CAES\n", levels);
            levels_ = levels;
            init_network(in_width, in_height, in_channels, levels, window_sizes, num_filters);
        }

        int num_levels() {return levels_;}

    protected:
        int levels_;
    private:
        typedef tiny_cnn::network<tiny_cnn::mse, tiny_cnn::gradient_descent> CNN;
        typedef tiny_cnn::convolutional_AE_layer<CNN, tiny_cnn::activation::tan_h> CAE;

        void init_network(int in_width, int in_height, int in_channels, int levels, int* window_sizes, int* num_filters){
            int width = in_width;
            int height = in_height;
            int channels = in_channels;
            for (int level = 0; level < levels; level++){
                printf("in_size: %d x %d x %d\n", width, height, channels);
                add(new CAE(width, height, window_sizes[level], channels, num_filters[level]));
                width = width - window_sizes[level] + 1;
                height = height - window_sizes[level] + 1;
                channels = num_filters[level];
                }
            last_encoder_ = static_cast<AEL*>(layers_.tail()); // static_cast is valid because we use the tan_h activation in the layer
            assert(this->in_dim() == in_width * in_height * in_channels);
            assert(this->out_dim() == width * height * channels);
        }

};

#endif // CAES_HPP
