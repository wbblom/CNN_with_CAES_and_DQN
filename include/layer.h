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
#include "product.h"

namespace tiny_cnn {

// base class of all kind of NN layers
template<typename N>
class layer_base {
public:
    typedef N Network;
    typedef typename Network::Optimizer Optimizer;
    typedef typename Network::LossFunction LossFunction;

    layer_base(int in_dim, int out_dim, int weight_dim) : parallelize_(true), next_(0), prev_(0) {
        set_size(in_dim, out_dim, weight_dim);
    }

    void connect(layer_base<N>* tail) {
        if (this->out_size() != 0 && tail->in_size() != this->out_size())
            throw nn_error("dimension mismatch");
        this->next_ = tail;
        tail->prev_ = this;
    }

    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-parameter"
    virtual void connect_2nd(layer_base<N>* tail) {return;};
    #pragma GCC diagnostic pop

    void set_parallelize(bool parallelize) {
        parallelize_ = parallelize;
    }

    // cannot call from ctor because of pure virtual function call
    // so should call this function explicitly after ctor
    virtual void init_weight() = 0;

    vec_t& weight() { return W_; }

    virtual int in_size() const { return in_size_; }
    virtual int out_size() const { return out_size_; }
    virtual int param_size() = 0;
    virtual int fan_in_size() const = 0;
    virtual int connection_size() const = 0;
    virtual int layer_type() {return 0; };
    /// layer types are
    // -1 input_layer
    // 0 layer_base
    // 1 layer
    // 2 AE_layer

    virtual void save(std::ostream& os) = 0;
    virtual void load(std::istream& is) = 0;

    virtual activation::function& activation_function() = 0;
    virtual const vec_t& forward_propagation(const vec_t& in, int worker_index) = 0;
    virtual const vec_t& back_propagation(const vec_t& current_delta, int worker_index) = 0;
    virtual const vec_t& back_propagation_2nd(const vec_t& current_delta2) = 0;

    // called afrer updating weight
    virtual void post_update() {}

    layer_base<N>* next() { return next_; }
    layer_base<N>* prev() { return prev_; }

    virtual void update_weight(Optimizer *o, int worker_size, int batch_size) = 0;

    vec_t& weight_diff(int index) { return dW_[index]; }

protected:
    int in_size_;
    int out_size_;
    bool parallelize_;

    layer_base<N>* next_;
    layer_base<N>* prev_;
    vec_t W_;          // weight vector
    vec_t dW_[CNN_TASK_SIZE];

    vec_t Whessian_; // diagonal terms of hessian matrix

private:
    virtual void merge(int worker_size, int batch_size) = 0;

    virtual void clear_diff(int worker_size) = 0;

    void set_size(int in_dim, int out_dim, int weight_dim/*, int bias_dim*/) {
        in_size_ = in_dim;
        out_size_ = out_dim;

        W_.resize(weight_dim);
        Whessian_.resize(weight_dim);

        for (auto& dw : dW_)
            dw.resize(weight_dim);
    }
};

template<typename N, typename Activation>
class layer : public layer_base<N> {
public:
    typedef layer_base<N> Base;
    typedef typename Base::Optimizer Optimizer;

    layer(int in_dim, int out_dim, int weight_dim, int bias_dim) : layer_base<N>(in_dim, out_dim, weight_dim) {
        set_size(in_dim, out_dim, bias_dim);
    }

    activation::function& activation_function() { return a_; }

    const vec_t& output(int worker_index) const { return output_[worker_index]; }
    const vec_t& delta(int worker_index) const { return prev_delta_[worker_index]; }
    vec_t& bias() { return b_; }
    virtual int param_size() { return this->W_.size() + b_.size(); }
    virtual int layer_type() {return 1;}

    virtual void connect_2nd(void) {}

    virtual void save(std::ostream& os) {
        for (auto w : this->W_) os << w << " ";
        for (auto b : b_) os << b << " ";
    }

    virtual void load(std::istream& is) {
        for (auto& w : this->W_) is >> w;
        for (auto& b : b_) is >> b;
    }

    void divide_hessian(int denominator) {
        for (auto& w : this->Whessian_) w /= denominator;
        for (auto& b : bhessian_) b /= denominator;
    }

    virtual void update_weight(Optimizer *o, int worker_size, int batch_size) {
		if (this->W_.empty()) {
			return;
		}

        merge(worker_size, batch_size);

        for_(true, 0, this->W_.size(), [&](const blocked_range& r){
            for (int i = r.begin(); i < r.end(); i++)
                o->update(this->dW_[0][i], this->Whessian_[i], &(this->W_[i]));
        });

        int dim_b = b_.size();
        for (int i = 0; i < dim_b; i++)
            o->update(db_[0][i], bhessian_[i], &b_[i]);

        clear_diff(worker_size);
        this->post_update();
    }

    vec_t& bias_diff(int index) { return db_[index]; }

	virtual bool has_same_weights(const layer& rhs, float_t eps) {
        if (this->W_.size() != rhs.W_.size() || b_.size() != rhs.b_.size())
            return false;

        for (size_t i = 0; i < this->W_.size(); i++){
            double var1 = this->W_[i];
            double var2 = rhs.W_[i];
            double bla = std::abs(var1 - var2);
          if (bla > eps) return false;}
        for (size_t i = 0; i < b_.size(); i++)
          if (std::abs(b_[i] - rhs.b_[i]) > eps) return false;

        return true;
	}

    // cannot call from ctor because of pure virtual function call fan_in_size().
    // so should call this function explicitly after ctor
    virtual void init_weight() {
        const float_t weight_base = 0.5 / std::sqrt(this->fan_in_size());

        uniform_rand(this->W_.begin(), this->W_.end(), -weight_base, weight_base);
        uniform_rand(b_.begin(), b_.end(), -weight_base, weight_base);
        std::fill(this->Whessian_.begin(), this->Whessian_.end(), 0.0);
        std::fill(bhessian_.begin(), bhessian_.end(), 0.0);
        clear_diff(CNN_TASK_SIZE);
    }

protected:
    Activation a_;
    vec_t output_[CNN_TASK_SIZE];     // last output of current layer, set by fprop
    vec_t prev_delta_[CNN_TASK_SIZE]; // last delta of previous layer, set by bprop
    vec_t b_;          // bias vector
    vec_t db_[CNN_TASK_SIZE];
    vec_t bhessian_;
    vec_t prev_delta2_; // d^2E/da^2

private:
    virtual void merge(int worker_size, int batch_size) {
        for (int i = 1; i < worker_size; i++)
            vectorize::reduce<float_t>(&(this->dW_[i][0]), this->dW_[i].size(), &(this->dW_[0][0]));
        for (int i = 1; i < worker_size; i++)
            vectorize::reduce<float_t>(&db_[i][0], db_[i].size(), &db_[0][0]);
        //for (int i = 1; i < worker_size; i++) {
        //    std::transform(dW_[0].begin(), dW_[0].end(), dW_[i].begin(), dW_[0].begin(), std::plus<float_t>());
        //    std::transform(db_[0].begin(), db_[0].end(), db_[i].begin(), db_[0].begin(), std::plus<float_t>());
        //}

        std::transform(this->dW_[0].begin(), this->dW_[0].end(), this->dW_[0].begin(), [&](float_t x) { return x / batch_size; });
        std::transform(db_[0].begin(), db_[0].end(), db_[0].begin(), [&](float_t x) { return x / batch_size; });
    }

    virtual void clear_diff(int worker_size) {
        for (int i = 0; i < worker_size; i++) {
            std::fill(this->dW_[i].begin(), this->dW_[i].end(), 0.0);
            std::fill(db_[i].begin(), db_[i].end(), 0.0);
        }
    }
    void set_size(int in_dim, int out_dim, int bias_dim) {
        for (auto& o : output_)
            o.resize(out_dim);
        for (auto& p : prev_delta_)
            p.resize(in_dim);
        b_.resize(bias_dim);
        bhessian_.resize(bias_dim);
        prev_delta2_.resize(in_dim);

        for (auto& db : db_)
            db.resize(bias_dim);
    }
};

template<typename N, typename Activation>
class AE_layer : public layer_base<N> {
public:
    typedef layer_base<N> Base;
    typedef typename Base::Optimizer Optimizer;

    AE_layer(int in_dim, int out_dim, int weight_dim, int encoder_bias_dim, int decoder_bias_dim) : layer_base<N>(in_dim, out_dim, weight_dim) {
        set_size(in_dim, out_dim, encoder_bias_dim, decoder_bias_dim);
    }

    activation::function& activation_function() { return a_; }

    virtual void connect_2nd(layer_base<N>* tail) {
        if (layer_type() == 2) {
            AE_layer<N, Activation>* AE_tail = static_cast<AE_layer<N, Activation>* >(tail);
            next_AE_ = AE_tail;
            AE_tail->prev_AE_ = this;
        }
    }

    const vec_t& encoder_output(int worker_index) const { return encoder_output_[worker_index]; }
    const vec_t& decoder_output(int worker_index) const { return decoder_output_[worker_index]; }
    const vec_t& encoder_delta(int worker_index) const { return encoder_prev_delta_[worker_index]; }
    const vec_t& decoder_delta(int worker_index) const { return decoder_prev_delta_[worker_index]; }
    vec_t& encoder_bias() { return encoder_b_; }
    vec_t& decoder_bias() { return decoder_b_; }
    vec_t& encoder_bias_diff(int index) { return encoder_db_[index]; }
    vec_t& decoder_bias_diff(int index) { return decoder_db_[index]; }
    virtual int param_size() { return this->W_.size() + encoder_b_.size() + decoder_b_.size(); }
    virtual int layer_type() {return 2;}

    virtual void save(std::ostream& os) {
        for (auto w : this->W_) os << w << " ";
        for (auto b : encoder_b_) os << b << " ";
        for (auto b : decoder_b_) os << b << " ";
    }

    virtual void load(std::istream& is) {
        for (auto& w : this->W_) is >> w;
        for (auto& b : encoder_b_) is >> b;
        for (auto& b : decoder_b_) is >> b;
    }

    void divide_hessian(int denominator) {
        for (auto& w : this->Whessian_) w /= denominator;
        for (auto& b : encoder_bhessian_) b /= denominator;
        for (auto& b : decoder_bhessian_) b /= denominator;
    }

    virtual const vec_t& decoder_forward_propagation(const vec_t& in, int index) = 0;
    virtual const vec_t& decoder_back_propagation(const vec_t& current_delta, int index) = 0;
    virtual const vec_t& decoder_back_propagation_2nd(const vec_t& current_delta2) = 0;

    virtual void update_weight(Optimizer *o, int worker_size, int batch_size) {
		if (this->W_.empty()) {
			return;
		}

        merge(worker_size, batch_size);

        for_(true, 0, this->W_.size(), [&](const blocked_range& r){
            for (int i = r.begin(); i < r.end(); i++)
                o->update(this->dW_[0][i], this->Whessian_[i], &(this->W_[i]));
        });

        int dim_b = encoder_b_.size();
        for (int i = 0; i < dim_b; i++)
            o->update(encoder_db_[0][i], encoder_bhessian_[i], &encoder_b_[i]);

        dim_b = decoder_b_.size();
        for (int i = 0; i < dim_b; i++)
            o->update(decoder_db_[0][i], decoder_bhessian_[i], &decoder_b_[i]);

        clear_diff(worker_size);
        this->post_update();
    }

	virtual bool has_same_weights(const AE_layer& rhs, float_t eps) {
        if (this->W_.size() != rhs.W_.size() || encoder_b_.size() != rhs.encoder_b_.size() || decoder_b_.size() != rhs.decoder_b_.size())
            return false;

        for (size_t i = 0; i < this->W_.size(); i++){
            double var1 = this->W_[i];
            double var2 = rhs.W_[i];
            double bla = std::abs(var1 - var2);
          if (bla > eps) return false;}
        for (size_t i = 0; i < encoder_b_.size(); i++)
          if (std::abs(encoder_b_[i] - rhs.encoder_b_[i]) > eps) return false;
        for (size_t i = 0; i < decoder_b_.size(); i++)
          if (std::abs(decoder_b_[i] - rhs.decoder_b_[i]) > eps) return false;

        return true;
	}

    // cannot call from ctor because of pure virtual function call fan_in_size().
    // so should call this function explicitly after ctor
    virtual void init_weight() {
        const float_t weight_base = 0.5 / std::sqrt(this->fan_in_size());

        uniform_rand(this->W_.begin(), this->W_.end(), -weight_base, weight_base);
        uniform_rand(encoder_b_.begin(), encoder_b_.end(), -weight_base, weight_base);
        uniform_rand(decoder_b_.begin(), decoder_b_.end(), -weight_base, weight_base);
        std::fill(this->Whessian_.begin(), this->Whessian_.end(), 0.0);
        std::fill(encoder_bhessian_.begin(), encoder_bhessian_.end(), 0.0);
        std::fill(decoder_bhessian_.begin(), decoder_bhessian_.end(), 0.0);
        clear_diff(CNN_TASK_SIZE);
    }

protected:
    Activation a_;
    vec_t encoder_output_[CNN_TASK_SIZE];     // last output of current layer, set by fprop
    vec_t decoder_output_[CNN_TASK_SIZE];
    vec_t encoder_prev_delta_[CNN_TASK_SIZE]; // last delta of previous layer, set by bprop
    vec_t decoder_prev_delta_[CNN_TASK_SIZE];
    vec_t encoder_b_;          // bias vector
    vec_t decoder_b_;
    vec_t encoder_db_[CNN_TASK_SIZE];
    vec_t decoder_db_[CNN_TASK_SIZE];
    vec_t encoder_bhessian_;
    vec_t decoder_bhessian_;
    vec_t encoder_prev_delta2_; // d^2E/da^2
    vec_t decoder_prev_delta2_;

    AE_layer<N, Activation>* next_AE_;
    AE_layer<N, Activation>* prev_AE_;

private:
    virtual void merge(int worker_size, int batch_size) {
        for (int i = 1; i < worker_size; i++)
            vectorize::reduce<float_t>(&(this->dW_[i][0]), this->dW_[i].size(), &(this->dW_[0][0]));
        for (int i = 1; i < worker_size; i++)
            vectorize::reduce<float_t>(&encoder_db_[i][0], encoder_db_[i].size(), &encoder_db_[0][0]);
        for (int i = 1; i < worker_size; i++)
            vectorize::reduce<float_t>(&decoder_db_[i][0], decoder_db_[i].size(), &decoder_db_[0][0]);
        //for (int i = 1; i < worker_size; i++) {
        //    std::transform(dW_[0].begin(), dW_[0].end(), dW_[i].begin(), dW_[0].begin(), std::plus<float_t>());
        //    std::transform(db_[0].begin(), db_[0].end(), db_[i].begin(), db_[0].begin(), std::plus<float_t>());
        //}

        std::transform(this->dW_[0].begin(), this->dW_[0].end(), this->dW_[0].begin(), [&](float_t x) { return x / batch_size; });
        std::transform(encoder_db_[0].begin(), encoder_db_[0].end(), encoder_db_[0].begin(), [&](float_t x) { return x / batch_size; });
        std::transform(decoder_db_[0].begin(), decoder_db_[0].end(), decoder_db_[0].begin(), [&](float_t x) { return x / batch_size; });
    }

    virtual void clear_diff(int worker_size) {
        for (int i = 0; i < worker_size; i++) {
            std::fill(this->dW_[i].begin(), this->dW_[i].end(), 0.0);
            std::fill(encoder_db_[i].begin(), encoder_db_[i].end(), 0.0);
            std::fill(decoder_db_[i].begin(), decoder_db_[i].end(), 0.0);
        }
    }
    void set_size(int in_dim, int out_dim, int encoder_bias_dim, int decoder_bias_dim) {
        for (auto& o : encoder_output_)
            o.resize(out_dim);
        for (auto& o : decoder_output_)
            o.resize(in_dim);
        for (auto& p : encoder_prev_delta_)
            p.resize(in_dim);
        for (auto& p : decoder_prev_delta_)
            p.resize(out_dim);
        encoder_b_.resize(encoder_bias_dim);
        decoder_b_.resize(decoder_bias_dim);
        encoder_bhessian_.resize(encoder_bias_dim);
        decoder_bhessian_.resize(decoder_bias_dim);
        encoder_prev_delta2_.resize(in_dim);
        decoder_prev_delta2_.resize(out_dim);

        for (auto& db : encoder_db_)
            db.resize(encoder_bias_dim);
        for (auto& db : decoder_db_)
            db.resize(decoder_bias_dim);
    }
};

template<typename N>
class input_layer : public layer<N, activation::identity> {
public:
    typedef layer<N, activation::identity> Base;
    typedef typename Base::Optimizer Optimizer;

    input_layer() : layer<N, activation::identity>(0, 0, 0, 0) {}

    int in_size() const { return this->next_ ? this->next_->in_size(): 0; }
    virtual int layer_type() {return -1;}

    const vec_t& forward_propagation(const vec_t& in, int index) {
        this->output_[index] = in;
        return this->next_ ? this->next_->forward_propagation(in, index) : this->output_[index];
    }

    const vec_t& decoder_forward_propagation(const vec_t& in, int index) {
        // to stop the decoding chain for autoencoder network
        return in; // 15.6%
    }

    const vec_t& back_propagation(const vec_t& current_delta, int /*index*/) {
        return current_delta;
    }

    const vec_t& back_propagation_2nd(const vec_t& current_delta2) {
        return current_delta2;
    }

    int connection_size() const {
        return this->in_size_;
    }

    int fan_in_size() const {
        return 1;
    }
};

template<typename N>
class layers {
public:
    typedef typename N::Optimizer Optimizer;

    layers() {
        add(&first_);
    }

    layers(const layers& rhs) {
        construct(rhs);
    }

    layers<N>& operator = (const layers<N>& rhs) {
        layers_.clear();
        construct(rhs);
        return *this;
    }

    void add(layer_base<N> * new_tail) {
        if (tail()) {
            tail()->connect(new_tail);
            tail()->connect_2nd(new_tail);
        }
        layers_.push_back(new_tail);
    }

    bool empty() const { return layers_.size() == 0; }

    layer_base<N>* head() const { return empty() ? 0 : layers_[0]; }

    layer_base<N>* tail() const { return empty() ? 0 : layers_[layers_.size() - 1]; }

    void reset() {
        for (auto pl : layers_)
            pl->init_weight();
    }

    void divide_hessian(int denominator) {
        for (auto pl : layers_)
            pl->divide_hessian(denominator);
    }

    void update_weights(Optimizer *o, int worker_size, int batch_size) {
        for (auto pl : layers_)
            pl->update_weight(o, worker_size, batch_size);
    }

    void set_parallelize(bool parallelize) {
        for (auto pl : layers_)
            pl->set_parallelize(parallelize);
    }

private:
    void construct(const layers<N>& rhs) {
        add(&first_);
        for (int i = 1; i < (int) rhs.layers_.size(); i++)
            add(rhs.layers_[i]);
    }

    std::vector<layer_base<N>*> layers_;
    input_layer<N> first_;
};

template <typename Char, typename CharTraits, typename N>
std::basic_ostream<Char, CharTraits>& operator << (std::basic_ostream<Char, CharTraits>& os, const layer_base<N>& v) {
    v.save(os);
    return os;
}

template <typename Char, typename CharTraits, typename N>
std::basic_istream<Char, CharTraits>& operator >> (std::basic_istream<Char, CharTraits>& os, layer_base<N>& v) {
    v.load(os);
    return os;
}

} // namespace tiny_cnn
