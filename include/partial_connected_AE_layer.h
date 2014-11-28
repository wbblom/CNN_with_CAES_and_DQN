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
#include "layer.h"

namespace tiny_cnn {

template<typename N, typename Activation>
class partial_connected_AE_layer : public AE_layer<N, Activation> {
public:
    typedef std::vector<std::pair<unsigned short, unsigned short> > io_connections;
    typedef std::vector<std::pair<unsigned short, unsigned short> > wi_connections;
    typedef std::vector<std::pair<unsigned short, unsigned short> > wo_connections;
    typedef AE_layer<N, Activation> Base;
    typedef typename Base::Optimizer Optimizer;

    partial_connected_AE_layer(int in_dim, int out_dim, int weight_dim, int encoder_bias_dim, int decoder_bias_dim, float_t scale_factor = 1.0)
        : AE_layer<N, Activation> (in_dim, out_dim, weight_dim, encoder_bias_dim, decoder_bias_dim),
        encoder_weight2io_(weight_dim), encoder_out2wi_(out_dim), encoder_in2wo_(in_dim), encoder_bias2out_(encoder_bias_dim), encoder_out2bias_(encoder_bias_dim),
        decoder_weight2io_(weight_dim), decoder_out2wi_(out_dim), decoder_in2wo_(in_dim), decoder_bias2out_(decoder_bias_dim), decoder_out2bias_(decoder_bias_dim),
        scale_factor_(scale_factor) {
        if (in_dim <= 0 || weight_dim <= 0 || weight_dim <= 0 || encoder_bias_dim <= 0 || decoder_bias_dim)
            throw nn_error("invalid layer size");
    }

    int param_size() const {
        int total_param = 0;
        for (auto w : encoder_weight2io_)
            if (w.size() > 0) total_param++;
        for (auto b : encoder_bias2out_)
            if (b.size() > 0) total_param++;
        for (auto b : decoder_bias2out_)
            if (b.size() > 0) total_param++;
        return total_param;
    }

    int connection_size() const {
        int total_size = 0;
        for (auto io : encoder_weight2io_)
            total_size += io.size();
        for (auto b : encoder_bias2out_)
            total_size += b.size();
        for (auto io : decoder_weight2io_)
            total_size += io.size();
        for (auto b : decoder_bias2out_)
            total_size += b.size();
        return total_size;
    }

    int fan_in_size() const {
        return encoder_out2wi_[0].size();
    }

    void connect_encoder_weight(int input_index, int output_index, int weight_index) {
        encoder_weight2io_[weight_index].push_back(std::make_pair(input_index, output_index));
        encoder_out2wi_[output_index].push_back(std::make_pair(weight_index, input_index));
        encoder_in2wo_[input_index].push_back(std::make_pair(weight_index, output_index));
    }

    void connect_decoder_weight(int input_index, int output_index, int weight_index) {
        decoder_weight2io_[weight_index].push_back(std::make_pair(input_index, output_index));
        decoder_out2wi_[output_index].push_back(std::make_pair(weight_index, input_index));
        decoder_in2wo_[input_index].push_back(std::make_pair(weight_index, output_index));
    }

    void connect_encoder_bias(int bias_index, int output_index) {
        encoder_out2bias_[output_index] = bias_index;
        encoder_bias2out_[bias_index].push_back(output_index);
    }

    void connect_decoder_bias(int bias_index, int output_index) {
        decoder_out2bias_[output_index] = bias_index;
        decoder_bias2out_[bias_index].push_back(output_index);
    }

    virtual const vec_t& forward_propagation(const vec_t& in, int index) {
        forward_propagation(in, this->encoder_output_, index, this->encoder_b_, encoder_out2wi_, encoder_out2bias_);

        if (this->next_AE_){ // go to next encoder
            return this->next_AE_->forward_propagation(this->encoder_output_[index], index);
        }
        else if (!this->next_){ // this the tail, start decoding
            return this->decoder_forward_propagation(this->encoder_output_[index], index);
        }
        else { // split propagation, first do full decoding, then continue propagating other-type layers
            vec_t justforAEpropagation = this->decoder_forward_propagation(this->encoder_output_[index], index);
            return this->next_->forward_propagation(this->encoder_output_[index], index);
        }
    }

    virtual const vec_t& decoder_forward_propagation(const vec_t& in, int index) {
        forward_propagation(in, this->decoder_output_, index, this->decoder_b_, decoder_out2wi_, decoder_out2bias_);
        // is there an autoencoder before this one
        return this->prev_AE_ ? this->prev_AE_->decoder_forward_propagation(this->decoder_output_[index], index) : this->decoder_output_[index]; // 15.6%
    }

    void forward_propagation(const vec_t& in, vec_t output[CNN_TASK_SIZE], int index, const vec_t& b, std::vector<wi_connections>& out2wi, std::vector<int>& out2bias) {
        for_(this->parallelize_, 0, this->out_size_, [&](const blocked_range& r) {
            for (int i = r.begin(); i < r.end(); i++) {
                const wi_connections& connections = out2wi[i];
                float_t a = 0.0;

                for (auto connection : connections)// 13.1%
                    a += this->W_[connection.first] * in[connection.second]; // 3.2%

                a *= scale_factor_;
                a += b[out2bias[i]];
                output[index][i] = this->a_.f(a); // 9.6%
            }
        });
    }

    /// BACKPROPAGATION ASSUMES THE WHOLE NETWORK CONSISTS OF AUTO-ENCODERS
    virtual const vec_t& back_propagation(const vec_t& current_delta, int index) {
        back_propagation(current_delta, index,
                          this->prev_AE_->encoder_output(index),
                          this->prev_AE_->activation_function(),
                          this->encoder_prev_delta_[index],
                          encoder_in2wo_,
                          encoder_weight2io_,
                          encoder_bias2out_,
                          this->encoder_db_);

        return this->prev_->back_propagation(this->encoder_prev_delta_[index], index);
    }

    virtual const vec_t& decoder_back_propagation(const vec_t& current_delta, int index) {
        if (this->next_AE_) {
            back_propagation(current_delta, index,
                              this->next_AE_->decoder_output(index),
                              this->next_AE_->activation_function(),
                              this->decoder_prev_delta_[index],
                              decoder_in2wo_,
                              decoder_weight2io_,
                              decoder_bias2out_,
                              this->decoder_db_);
            return this->next_AE_->decoder_back_propagation(this->decoder_prev_delta_[index], index);
        }
        else { // this is the AE_tail
            back_propagation(current_delta, index,
                              this->encoder_output(index),
                              this->activation_function(),
                              this->decoder_prev_delta_[index],
                              decoder_in2wo_,
                              decoder_weight2io_,
                              decoder_bias2out_,
                              this->decoder_db_);
            return this->back_propagation(this->decoder_prev_delta_[index], index);
        }
    }

    void back_propagation(const vec_t& current_delta,
                          int index,
                          const vec_t& prev_out,
                          const activation::function& prev_h,
                          vec_t& prev_delta,
                          std::vector<wo_connections>& in2wo,
                          std::vector<io_connections>& weight2io,
                          std::vector<std::vector<int> >& bias2out,
                          vec_t db[CNN_TASK_SIZE]) {

        for_(this->parallelize_, 0, this->in_size_, [&](const blocked_range& r) {
            for (int i = r.begin(); i != r.end(); i++) {
                const wo_connections& connections = in2wo[i];
                float_t delta = 0.0;

                for (auto connection : connections)
                    delta += this->W_[connection.first] * current_delta[connection.second]; // 40.6%

                prev_delta[i] = delta * scale_factor_ * prev_h.df(prev_out[i]); // 2.1%
            }
        });

        for_(this->parallelize_, 0, weight2io.size(), [&](const blocked_range& r) {
            for (int i = r.begin(); i < r.end(); i++) {
                const io_connections& connections = weight2io[i];
                float_t diff = 0.0;

                for (auto connection : connections) // 11.9%
                    diff += prev_out[connection.first] * current_delta[connection.second];

                this->dW_[index][i] += diff * scale_factor_;
            }
        });

        for (size_t i = 0; i < bias2out.size(); i++) {
            const std::vector<int>& outs = bias2out[i];
            float_t diff = 0.0;

            for (auto o : outs)
                diff += current_delta[o];

            db[index][i] += diff;
        }
    }

    virtual const vec_t& back_propagation_2nd(const vec_t& current_delta2) {
        back_propagation_2nd(current_delta2,
                              this->prev_AE_->encoder_output(0),
                              this->prev_AE_->activation_function(),
                              encoder_weight2io_,
                              encoder_bias2out_,
                              this->encoder_bhessian_,
                              encoder_in2wo_,
                              this->encoder_prev_delta2_,
                              this->in_size_);

        return this->prev_->back_propagation_2nd(this->encoder_prev_delta2_);
    }

    virtual const vec_t& decoder_back_propagation_2nd(const vec_t& current_delta2) {
        if (this->next_AE_)
            back_propagation_2nd(current_delta2,
                              this->next_AE_->decoder_output(0),
                              this->next_AE_->activation_function(),
                              decoder_weight2io_,
                              decoder_bias2out_,
                              this->decoder_bhessian_,
                              decoder_in2wo_,
                              this->decoder_prev_delta2_,
                              this->out_size_);
        else // this is the AE_tail
            back_propagation_2nd(current_delta2,
                              this->encoder_output(0),
                              this->activation_function(),
                              decoder_weight2io_,
                              decoder_bias2out_,
                              this->decoder_bhessian_,
                              decoder_in2wo_,
                              this->decoder_prev_delta2_,
                              this->out_size_);

        return this->next_AE_ ? this->next_AE_->decoder_back_propagation_2nd(this->decoder_prev_delta2_) : this->back_propagation_2nd(this->decoder_prev_delta2_);
    }

    void back_propagation_2nd(const vec_t& current_delta2,
                              const vec_t& prev_out,
                              const activation::function& prev_h,
                              std::vector<io_connections>& weight2io,
                              std::vector<std::vector<int> >& bias2out,
                              vec_t& bhessian,
                              std::vector<wo_connections>& in2wo,
                              vec_t& prev_delta2,
                              int input_size) {

        for (size_t i = 0; i < weight2io.size(); i++) {
            const io_connections& connections = weight2io[i];
            float_t diff = 0.0;

            for (auto connection : connections)
                diff += prev_out[connection.first] * prev_out[connection.first] * current_delta2[connection.second];

            diff *= scale_factor_ * scale_factor_;
            this->Whessian_[i] += diff;
        }

        for (size_t i = 0; i < bias2out.size(); i++) {
            const std::vector<int>& outs = bias2out[i];
            float_t diff = 0.0;

            for (auto o : outs)
                diff += current_delta2[o];

            bhessian[i] += diff;
        }

        for (int i = 0; i < input_size; i++) {
            const wo_connections& connections = in2wo[i];
            prev_delta2[i] = 0.0;

            for (auto connection : connections)
                prev_delta2[i] += this->W_[connection.first] * this->W_[connection.first] * current_delta2[connection.second];

            prev_delta2[i] *= scale_factor_ * scale_factor_ * prev_h.df(prev_out[i]) * prev_h.df(prev_out[i]);
        }
    }

    // remove unused weight to improve cache hits
    void remap() {
        std::map<int, int> swaps;
        int n = 0;

        for (size_t i = 0; i < encoder_weight2io_.size(); i++)
            swaps[i] = encoder_weight2io_[i].empty() ? -1 : n++;

        for (int i = 0; i < this->out_size_; i++) {
            wi_connections& wi = encoder_out2wi_[i];
            for (size_t j = 0; j < wi.size(); j++)
                wi[j].first = static_cast<unsigned short>(swaps[wi[j].first]);
        }

        for (int i = 0; i < this->in_size_; i++) {
            wo_connections& wo = encoder_in2wo_[i];
            for (size_t j = 0; j < wo.size(); j++)
                wo[j].first = static_cast<unsigned short>(swaps[wo[j].first]);
        }

        std::vector<io_connections> weight2io_new(n);
        for (size_t i = 0; i < encoder_weight2io_.size(); i++)
            if(swaps[i] >= 0) weight2io_new[swaps[i]] = encoder_weight2io_[i];

        encoder_weight2io_ = weight2io_new;

        //now the decoder
        swaps.clear();
        n = 0;

        for (size_t i = 0; i < decoder_weight2io_.size(); i++)
            swaps[i] = decoder_weight2io_[i].empty() ? -1 : n++;

        for (int i = 0; i < this->in_size_; i++) {
            wi_connections& wi = decoder_out2wi_[i];
            for (size_t j = 0; j < wi.size(); j++)
                wi[j].first = static_cast<unsigned short>(swaps[wi[j].first]);
        }

        for (int i = 0; i < this->out_size_; i++) {
            wo_connections& wo = decoder_in2wo_[i];
            for (size_t j = 0; j < wo.size(); j++)
                wo[j].first = static_cast<unsigned short>(swaps[wo[j].first]);
        }

        weight2io_new.resize(n);
        for (size_t i = 0; i < decoder_weight2io_.size(); i++)
            if(swaps[i] >= 0) weight2io_new[swaps[i]] = decoder_weight2io_[i];

        decoder_weight2io_ = weight2io_new;
    }

protected:
    std::vector<io_connections> encoder_weight2io_; // weight_id -> [(in_id, out_id)]
    std::vector<wi_connections> encoder_out2wi_; // out_id -> [(weight_id, in_id)]
    std::vector<wo_connections> encoder_in2wo_; // in_id -> [(weight_id, out_id)]
    std::vector<std::vector<int> > encoder_bias2out_;
    std::vector<int> encoder_out2bias_;
    std::vector<io_connections> decoder_weight2io_;
    std::vector<wi_connections> decoder_out2wi_;
    std::vector<wo_connections> decoder_in2wo_;
    std::vector<std::vector<int> > decoder_bias2out_;
    std::vector<int> decoder_out2bias_;
    float_t scale_factor_;
};

} // namespace tiny_cnn
