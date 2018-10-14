#ifndef __camera_cuh__
#define __camera_cuh__

#include <cuda_library\defines.hpp>
#include "vec.cuh"
#include "ray.cuh"

namespace aks
{
	namespace rx
	{
		template<typename V, typename S>
		AKS_FUNCTION_PREFIX_ATTR V random_in_unit_disk(S s) {
			V p;
			auto sqrtr = sqrt(curand_uniform(s));
			auto theta = curand_uniform(s) * 2 * 3.14159265;
			return V(sqrtr * cos(theta), sqrtr * sin(theta), 0) - V(1, 1, 0) * 2.;
			/*V p;
			do {
				p = V(curand_uniform(s), curand_uniform(s), 0) - V(1, 1, 0) * 2.;
			} while (dot(p, p) >= 1.0);
			return p;*/
		}

		template<typename V>
		struct camera
		{
			typedef V vec_type;
			typedef typename vec_type::value_type value_type;
			typedef ray<vec_type> ray_type;

			AKS_FUNCTION_PREFIX_ATTR camera(vec_type lookfrom, vec_type lookat, vec_type vup, value_type vfov, value_type aspect, value_type aperture, value_type focus_dist) { // vfov is top to bottom in degrees
				lens_radius = aperture / 2;
				value_type theta = (vfov * 3.14159265) / 180.0;
				value_type half_height = tan(theta / 2);
				value_type half_width = aspect * half_height;
				origin = lookfrom;
				w = unit_vector(lookfrom - lookat);
				u = unit_vector(cross(vup, w));
				v = cross(w, u);
				lower_left_corner = origin - u * half_width *focus_dist - v * half_height *focus_dist - w * focus_dist;
				horizontal = u * 2 * half_width*focus_dist;
				vertical = v * 2 * half_height*focus_dist;
			}
			template<typename S>
			AKS_FUNCTION_PREFIX_ATTR ray_type get_ray(value_type s, value_type t, S st) const {
				vec_type rd = lens_radius == 0.0 ? vec3f() : random_in_unit_disk<vec_type, S>(st) * lens_radius;
				vec_type offset = u * rd.x() + v * rd.y();
				return ray_type(origin + offset, lower_left_corner + horizontal * s + vertical * t - origin - offset);
			}

			vec_type origin;
			vec_type lower_left_corner;
			vec_type horizontal;
			vec_type vertical;
			vec_type u, v, w;
			value_type lens_radius;
		};

		using camera3f = camera<vec3f>;
	}
}

#endif // __camera_cuh__