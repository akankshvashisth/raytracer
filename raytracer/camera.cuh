#ifndef __camera_cuh__
#define __camera_cuh__

#include <cuda_library\defines.hpp>
#include "vec.cuh"
#include "ray.cuh"

namespace aks
{
	namespace rx
	{
		template<typename V>
		struct camera
		{
			typedef V vec_type;
			typedef typename vec_type::value_type value_type;
			typedef ray<vec_type> ray_type;

			AKS_FUNCTION_PREFIX_ATTR camera() :
				lower_left_corner(-2, -1, -1)
				, horizontal(4, 0, 0)
				, vertical(0, 2, 0)
				, origin(0, 0, 0) {}

			AKS_FUNCTION_PREFIX_ATTR ray_type get_ray(value_type u, value_type v) const {
				return ray_type(origin, lower_left_corner + horizontal * u + vertical * v - origin);
			}

			vec_type lower_left_corner;
			vec_type horizontal;
			vec_type vertical;
			vec_type origin;
		};

		using camera3f = camera<vec3f>;
	}
}

#endif // __camera_cuh__