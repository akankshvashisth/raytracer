#ifndef __ray_cuh__
#define __ray_cuh__

#include <cuda_library\defines.hpp>
#include "vec.cuh"

namespace aks
{
	namespace rx
	{
		template<typename V>
		struct ray
		{
			typedef V vec_type;

			AKS_FUNCTION_PREFIX_ATTR ray() {}
			AKS_FUNCTION_PREFIX_ATTR ray(vec_type const& o, vec_type const& d) :m_origin(o), m_direction(d) {}
			AKS_FUNCTION_PREFIX_ATTR vec_type origin() const { return m_origin; }
			AKS_FUNCTION_PREFIX_ATTR vec_type direction() const { return m_direction; }

			template<typename T>
			AKS_FUNCTION_PREFIX_ATTR vec_type at_parameter(T t) const {
				return m_origin + (m_direction * t);
			}

			vec_type m_origin;
			vec_type m_direction;
		};

		using ray3f = ray<vec3f>;
	}
}

#endif // __ray_cuh__