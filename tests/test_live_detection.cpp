#include "../tools/live_detection.hpp"
#include <stdexcept>
int main() {
    auto require=[](bool ok){if(!ok)throw std::runtime_error("detection cadence failed");};
    for(unsigned interval : {1u,5u,10u,30u}) {
        live_detection_schedule schedule(interval);
        for(unsigned frame=0;frame<93;++frame) {
            bool due=schedule.due();require(due==(frame%interval==0));
            if(due)schedule.detected(true);
        }
        schedule.reset();require(schedule.due());
        schedule.detected(false);require(schedule.due());
        schedule.detected(true);schedule.reset();require(schedule.due());
    }
}
