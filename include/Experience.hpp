#ifndef EXPERIENCE_HPP
#define EXPERIENCE_HPP

#include "State.hpp"
#include "Action.hpp"

class Experience
{
    private:
        // An experience consists of:
        State    s; // the state from which the action was chosen
        Action   a; // the action taken after considering the old state
        double   r; // the reward for this transition
        uint64_t t; // the discrete time stamp of the experience

    public:
        /// Empty constructor
        Experience() {};

        /// Constructor
        Experience(State state, Action action, double reward, uint64_t time) {
            s = state;
            a = action;
            r = reward;
            t = time;
        }

        /// Destructor
        virtual ~Experience() {
            // no special destructions needed
        }

        /// Copy constructor
        Experience(const Experience& other) {
            s = other.s;
            a = other.a;
            r = other.r;
            t = other.t;
        }

        /// Assignment operator
        Experience& operator=(const Experience& other) {
            s = other.s;
            a = other.a;
            r = other.r;
            t = other.t;
            return *this;
        }
};

#endif // EXPERIENCE_HPP
