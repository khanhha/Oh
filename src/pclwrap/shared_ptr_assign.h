#include <memory>
#include "../pcl/point_types.h"
#include "../pcl/point_cloud.h"

namespace {
    // Workaround for lack of operator= support in Cython.
    template <typename T>
    void sp_assign(std::shared_ptr<T> &p, T *v)
    {
        p = std::shared_ptr<T>(v);
    }
}
