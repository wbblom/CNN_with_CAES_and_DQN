#include <iostream>
#include "Experience.hpp"
#include "tiny_cnn.h"
#include "caes.hpp"

using namespace std;
//using namespace tiny_cnn;
//using namespace tiny_cnn::activation;

/// Settings
size_t replayMemory_Size = size_t(10000);
size_t State::visual_rows = size_t(300);
size_t State::visual_columns = size_t(100);
//State.visual_channels = size_t(4);
int epsilon = 1000; // out of 10000

int argmax(tiny_cnn::vec_t* in, size_t num){
    int arg = -INFINITY;
    for (size_t idx = 0; idx < num; idx++){
        if (arg < (*in)[idx]){
            arg = (*in)[idx];
        }
    }
    return arg;
}

int main()
{

//    vector<vector<pair<int,int> > > tmp(8);
//    tmp[0].push_back(make_pair(200, 20));
//    cout << tmp[0].size() << endl;



    int window_sizes[] = {2, 2};
    int num_filters[] = {2, 2};
    caes AE(4, 4, 1, 2, window_sizes, num_filters);






//    /// Initialize replay memory D to capacity N
//    Experience* replayMemory = new Experience[replayMemory_Size];
//
//    /// Initialize action-value function Q with random weights
//    // specify loss-function and optimization-algorithm
//    typedef tiny_cnn::network<tiny_cnn::mse, tiny_cnn::gradient_descent> CNN;
//    CNN mynet;
//
//    // tanh, 300x100 input, 8x8 window, 16 feature-maps convolution
//    tiny_cnn::convolutional_layer<CNN, tiny_cnn::activation::tan_h> C1(State::visual_rows, State::visual_columns, 8, 1, 16);
//
//    // tanh, 28x28 input, 4x4 window, 32 feature-maps convolution
//    // Suppose that we have some N×N square neuron layer which is followed by our convolutional layer. If we use an m×m filter ω, our convolutional layer output will be of size (N−m+1)×(N−m+1)
//    tiny_cnn::convolutional_layer<CNN, tiny_cnn::activation::tan_h> C2(300-8+1, 100-8+1, 4, 16, 32);
//
//    // fully-connected layers
//    tiny_cnn::fully_connected_layer<CNN, tiny_cnn::activation::sigmoid> F3((293-4+1)*(93-4+1), 256);
//    tiny_cnn::fully_connected_layer<CNN, tiny_cnn::activation::identity> F4(256, 14);
//
//    // connect all
//    mynet.add(&C1); mynet.add(&C2); mynet.add(&F3); mynet.add(&F4);
//
//    assert(mynet.in_dim() == int(State::visual_rows * State::visual_columns));
//    assert(mynet.out_dim() == 14);
//
//    /// for episode = 1, M do
//    bool running_episodes = true;
//    bool running_time;
//    size_t episode = 0;
//    uint64_t time;
//    int probability_action;
//    Action a_current;
//    State s_current;
//    tiny_cnn::vec_t ss_current; // substate (between autoencoder and DQN)
//    while (running_episodes){
//        episode++;
//        /// Initialise sequence s_1 = {x_1} and preprocessed sequenced φ_1 = φ(s_1)
//        // get state (s_current)
//        // store it
//        // preprocess it (ss_current)
//
//        /// for t = 1, T do
//        running_time = true;
//        time = 0;
//        while (running_time){
//            time++;
//            /// With probability epsilon select a random action a_t
//            probability_action = rand() % 10000;
//            if (probability_action < epsilon){
//                a_current = Action(rand() % Action::NUM_ACTIONS);
//            }
//            /// otherwise select a_t = max_a Q^∗ (φ(s_t), a; θ)
//            else {
//                tiny_cnn::vec_t* prediction;
//                mynet.predict(ss_current, prediction);
//                a_current = Action(argmax(prediction, Action::NUM_ACTIONS));
//            }
//            // store in memory
//
//            /// Execute action a_t in emulator and observe reward r_t and image x_{t+1}
//
//
//            /// Set s_{t+1} = s_t , a_t , x_{t+1} and preprocess φ_{t+1} = φ(s_{t+1})
//
//
//            /// Store transition (φ_t , a_t , r_t , φ_{t+1}) in D
//
//
//            /// Sample random minibatch of transitions (φ_j , a_j , r_j , φ_{j+1}) from D
//
//
//            /// Set y_j = r_j for terminal φ_{j+1}
//            /// Set y_j = r_j + γ max_a Q(φ_{j+1} , a ; θ) for non-terminal φ_{j+1}
//
//
//            /// Perform a gradient descent step on (y_j − Q(φ_j , a_j ; θ)) according to:
//            /// ∇_{θ_i} L_i (θ_i) = E_{s,a∼ρ(·);s∼E} [(r + γ max_a' Q(s', a'; θ_{i−1}) − Q(s, a; θ_i) ∇_{θ_i} Q(s, a; θ_i)]
//
//
//        /// end for
//        }
//    /// end for
//    }
//
//    delete[] replayMemory;
    return 0;
}
