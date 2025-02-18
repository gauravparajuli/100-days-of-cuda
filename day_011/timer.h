#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <chrono>

#define RED     "\033[31m"      /* Red */
#define DGREEN  "\033[32m"      /* Dark Green */
#define GREEN   "\033[92m"      /* Light Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define RESET   "\033[0m"       /* Reset */

class Timer {
public:
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<float> duration;
};

void startTime(Timer *timer){
    timer->start = std::chrono::high_resolution_clock::now();
}

void stopTime(Timer *timer){
    timer->end = std::chrono::high_resolution_clock::now();
    timer->duration = timer->end - timer->start;
}

void printElapsedTime(const Timer& timer, const std::string& message, const std::string& color = WHITE){
    float milliseconds = timer.duration.count() * 1000.0f; // Convert seconds to milliseconds
    std::cout << color << message << ": " << milliseconds << " milliseconds" << RESET << std::endl;
}

#endif // TIMER_H