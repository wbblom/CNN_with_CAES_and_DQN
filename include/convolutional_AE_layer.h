/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include "util.h"
#include "partial_connected_AE_layer.h"
#include "image.h"
#include <assert.h>
#include <stdio.h>

namespace tiny_cnn {

template<typename N, typename Activation>
class convolutional_AE_layer : public partial_connected_AE_layer<N, Activation> {
public:
    typedef partial_connected_AE_layer<N, Activation> Base;
    typedef typename Base::Optimizer Optimizer;

    convolutional_AE_layer(int in_width, int in_height,
                           int window_size,
                           int in_channels, int out_channels)
        : partial_connected_AE_layer<N, Activation>(in_width * in_height * in_channels, (in_width - window_size + 1) * (in_height - window_size + 1) * out_channels,
        window_size * window_size * in_channels * out_channels, out_channels, in_channels),
        encoder_in_(in_width, in_height, in_channels),
        decoder_out_(in_width, in_height, in_channels),
        encoder_out_((in_width - window_size + 1), (in_height - window_size + 1), out_channels),
        decoder_in_((in_width - window_size + 1), (in_height - window_size + 1), out_channels),
        weight_(window_size, window_size, in_channels*out_channels),
        window_size_(window_size)
    {
        printf("New Convolutional Auto-Eencoder Layer\nin: %d x %d x %d\nwindow: %d x %d\nout: %d x %d x %d\n", in_width, in_height, in_channels, window_size, window_size, (in_width - window_size + 1), (in_height - window_size + 1), out_channels);
        printf("Set up neural connections...\n");
        init_connection(connection_table(), connection_table());
        printf("Set up neural connections DONE\n");
    }

    convolutional_AE_layer(int in_width, int in_height,
                           int window_size,
                           int in_channels, int out_channels,
                           const connection_table& encoder_connection_table,
                           const connection_table& decoder_connection_table)
        : partial_connected_AE_layer<N, Activation>(in_width * in_height * in_channels, (in_width - window_size + 1) * (in_height - window_size + 1) * out_channels,
        window_size * window_size * in_channels * out_channels, out_channels, in_channels),
        encoder_in_(in_width, in_height, in_channels),
        decoder_out_(in_width, in_height, in_channels),
        encoder_out_((in_width - window_size + 1), (in_height - window_size + 1), out_channels),
        decoder_in_((in_width - window_size + 1), (in_height - window_size + 1), out_channels),
        weight_(window_size, window_size, in_channels*out_channels),
        window_size_(window_size)
    {
        init_connection(encoder_connection_table, encoder_connection_table);
        this->remap();
    }

    void weight_to_image(image& img) {
        const int border_width = 1;
        const int pitch = window_size_ + border_width;
        const int width = encoder_out_.depth_ * pitch + border_width;
        const int height = encoder_in_.depth_ * pitch + border_width;
        const image::intensity_t bg_color = 255;

        img.resize(width, height);
        img.fill(bg_color);

        auto minmax = std::minmax_element(this->W_.begin(), this->W_.end());

        for (int r = 0; r < encoder_in_.depth_; r++) {
            for (int c = 0; c < encoder_out_.depth_; c++) {
                if (!encoder_connection_.is_connected(c, r)) continue;

                const int top = r * pitch + border_width;
                const int left = c * pitch + border_width;

                for (int y = 0; y < window_size_; y++) {
                    for (int x = 0; x < window_size_; x++) {
                        const float_t w = this->W_[weight_.get_index(x, y, c * encoder_in_.depth_ + r)];

                        img.at(left + x, top + y)
                            = (image::intensity_t)rescale<float_t, int>(w, *minmax.first, *minmax.second, 0, 255);
                    }
                }
            }
        }
    }

private:
    void init_connection(const connection_table& encoder_table, const connection_table& decoder_table) {
        printf("Encoder weights...\n");
        for (int inc = 0; inc < encoder_in_.depth_; inc++) {
            for (int outc = 0; outc < encoder_out_.depth_; outc++) {
                if (!encoder_table.is_connected(outc, inc)) {
                    continue;
                }

                for (int y = 0; y < encoder_out_.height_; y++)
                    for (int x = 0; x < encoder_out_.width_; x++)
                        connect_encoder_kernel(inc, outc, x, y);
            }
        }
        printf("Encoder biases...\n");
        for (int outc = 0; outc < encoder_out_.depth_; outc++)
            for (int y = 0; y < encoder_out_.height_; y++)
                for (int x = 0; x < encoder_out_.width_; x++)
                    this->connect_encoder_bias(outc, encoder_out_.get_index(x, y, outc));
        printf("Decoder weights...\n");
        for (int inc = 0; inc < decoder_in_.depth_; inc++) {
            for (int outc = 0; outc < decoder_out_.depth_; outc++) {
                if (!decoder_table.is_connected(outc, inc)) {
                    continue;
                }

                for (int y = 0; y < decoder_out_.height_; y++)
                    for (int x = 0; x < decoder_out_.width_; x++)
                        connect_decoder_kernel(inc, outc, x, y);
            }
        }
        printf("Decoder biases...\n");
        for (int outc = 0; outc < decoder_out_.depth_; outc++)
            for (int y = 0; y < decoder_out_.height_; y++)
                for (int x = 0; x < decoder_out_.width_; x++)
                    this->connect_decoder_bias(outc, decoder_out_.get_index(x, y, outc));
    }

    /// Valid convolution connections
    void connect_encoder_kernel(int inc, int outc, int x, int y) {
        for (int dy = 0; dy < window_size_; dy++)
            for (int dx = 0; dx < window_size_; dx++)
                this->connect_encoder_weight(
                    encoder_in_.get_index(x + dx, y + dy, inc),
                    encoder_out_.get_index(x, y, outc),
                    weight_.get_index(dx, dy, outc * encoder_in_.depth_ + inc));
    }

    /// Full convolution connections
    void connect_decoder_kernel(int inc, int outc, int x, int y) {
        int dy0 = std::max(window_size_ - decoder_out_.height_ + y, 0);
        int dy1 = std::min(window_size_, y + 1);
        int dx0 = std::max(window_size_ - decoder_out_.width_ + x, 0);
        int dx1 = std::min(window_size_, x + 1);
        for (int dy = dy0; dy < dy1; dy++){
            for (int dx = dx0; dx < dx1; dx++){
                this->connect_decoder_weight(
                decoder_in_.get_index(x - dx, y - dy, inc),
                decoder_out_.get_index(x, y, outc),
                weight_.get_index(dx, dy, outc * decoder_in_.depth_ + inc)); // check this later
            }
        }
    }

    tensor3d encoder_in_;
    tensor3d decoder_out_;
    tensor3d encoder_out_;
    tensor3d decoder_in_;
    tensor3d weight_;
    connection_table encoder_connection_;
    connection_table decoder_connection_;
    int window_size_;
};

} // namespace tiny_cnn
