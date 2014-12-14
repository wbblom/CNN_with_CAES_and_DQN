#ifndef STATE_HPP
#define STATE_HPP

#include <cv.hpp>

class State
{
    private:
        // The visual data of the state
        cv::Mat visual;

    public:
        /// Size parameters
        static size_t visual_rows;
        static size_t visual_columns;
        static const size_t visual_channels = 4;

        /// Empty constructor
        State() {
            visual = cv::Mat::zeros(visual_rows, visual_columns, CV_8UC4);
        };

        /// Constructor
        State(cv::Mat camera) {
            camera.copyTo(visual);
        }

        /// Destructor
        virtual ~State() {
            visual.release();
        }

        /// Copy constructor
        State(const State& other) {
            other.visual.copyTo(visual);
        }

        /// Assignment operator
        State& operator=(const State& other) {
            other.visual.copyTo(visual);
            return *this;
        }
};

#endif // STATE_HPP
