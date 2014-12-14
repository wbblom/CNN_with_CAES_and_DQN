#ifndef ACTION_HPP
#define ACTION_HPP

// The possible actions the robot can take
enum Action {nothing=0, forward1, forward2, forward3, backward1, left, right, pickLow, pickForward, pickHigh, placeLow, placeForward, placeHigh, placeBackpack, NUM_ACTIONS};
///WARNING: make sure NUM_ACTIONS is always the last one in the set and the first item is always zero and the only fixed numbered item

#endif // ACTION_HPP
