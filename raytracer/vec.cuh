#ifndef __vec_cuh__
#define __vec_cuh__

#include <cuda_library\defines.hpp>

namespace aks
{
	namespace rx
	{
		namespace detail
		{
			template<typename T, typename... Ts>
			AKS_FUNCTION_PREFIX_ATTR void copy(T* data) {}

			template<typename T, typename U, typename... Ts>
			AKS_FUNCTION_PREFIX_ATTR void copy(T* data, U t, Ts... ts)
			{
				data[0] = t;
				copy(data + 1, ts...);
			}
		}

		template<typename T, size_t N>
		struct vec
		{
			using value_type = T;
			enum { dimensions = N };
			static_assert(dimensions > 0, "dimensions > 0");
			template<typename... Ts>
			AKS_FUNCTION_PREFIX_ATTR vec(Ts... ts) {
				static_assert(sizeof...(ts) == dimensions, "sizeof(ts) == dimensions");
				detail::copy(m_data, ts...);
			}

			//AKS_FUNCTION_PREFIX_ATTR vec(T a, T b, T c) {
			//	static_assert(3 == dimensions, "sizeof(ts) == dimensions");
			//	detail::copy(m_data, a, b, c);
			//}

			template<typename U>
			AKS_FUNCTION_PREFIX_ATTR vec(vec<U, N> const& v) {
				auto it0 = data(), end = data() + dimensions; auto it1 = v.data(); for (; it0 != end; ++it0, ++it1) { *it0 = *it1; }
			}

			AKS_FUNCTION_PREFIX_ATTR vec() { for (auto& i : m_data) { i = value_type(); } }

			AKS_FUNCTION_PREFIX_ATTR value_type y() const { return m_data[1]; }
			AKS_FUNCTION_PREFIX_ATTR value_type z() const { return m_data[2]; }
			AKS_FUNCTION_PREFIX_ATTR value_type w() const { return m_data[3]; }
			AKS_FUNCTION_PREFIX_ATTR value_type r() const { return m_data[0]; }
			AKS_FUNCTION_PREFIX_ATTR value_type g() const { return m_data[1]; }
			AKS_FUNCTION_PREFIX_ATTR value_type b() const { return m_data[2]; }
			AKS_FUNCTION_PREFIX_ATTR value_type a() const { return m_data[3]; }
			AKS_FUNCTION_PREFIX_ATTR value_type x() const { return m_data[0]; }

			AKS_FUNCTION_PREFIX_ATTR value_type& x() { return m_data[0]; }
			AKS_FUNCTION_PREFIX_ATTR value_type& y() { return m_data[1]; }
			AKS_FUNCTION_PREFIX_ATTR value_type& z() { return m_data[2]; }
			AKS_FUNCTION_PREFIX_ATTR value_type& w() { return m_data[3]; }
			AKS_FUNCTION_PREFIX_ATTR value_type& r() { return m_data[0]; }
			AKS_FUNCTION_PREFIX_ATTR value_type& g() { return m_data[1]; }
			AKS_FUNCTION_PREFIX_ATTR value_type& b() { return m_data[2]; }
			AKS_FUNCTION_PREFIX_ATTR value_type& a() { return m_data[3]; }

			AKS_FUNCTION_PREFIX_ATTR value_type* data() { return m_data; }
			AKS_FUNCTION_PREFIX_ATTR value_type const * data() const { return m_data; }

			AKS_FUNCTION_PREFIX_ATTR value_type* begin() { return data(); }
			AKS_FUNCTION_PREFIX_ATTR value_type* end() { return data() + dimensions; }

			AKS_FUNCTION_PREFIX_ATTR value_type const * cbegin() { return data(); }
			AKS_FUNCTION_PREFIX_ATTR value_type const * cend() { return data() + dimensions; }

			AKS_FUNCTION_PREFIX_ATTR value_type const * begin() const { return data(); }
			AKS_FUNCTION_PREFIX_ATTR value_type const * end() const { return data() + dimensions; }

			AKS_FUNCTION_PREFIX_ATTR value_type const * cbegin() const { return data(); }
			AKS_FUNCTION_PREFIX_ATTR value_type const * cend() const { return data() + dimensions; }

			AKS_FUNCTION_PREFIX_ATTR size_t size() const { return dimensions; }
			AKS_FUNCTION_PREFIX_ATTR bool empty() const { return false; }

			AKS_FUNCTION_PREFIX_ATTR value_type operator[](int i) const { return data()[i]; }
			AKS_FUNCTION_PREFIX_ATTR value_type& operator[](int i) { return data()[i]; }

			AKS_FUNCTION_PREFIX_ATTR vec operator-() const { return vec(*this) *= value_type(-1); }

			template<typename F>
			AKS_FUNCTION_PREFIX_ATTR vec& map_func(F f) {
				auto it0 = data(), end = data() + dimensions; for (; it0 != end; ++it0) { *it0 *= f(*it0); } return *this;
			}

			template<typename U>
			AKS_FUNCTION_PREFIX_ATTR vec& operator+= (vec<U, dimensions> const& v) {
				auto it0 = data(), end = data() + dimensions; auto it1 = v.data(); for (; it0 != end; ++it0, ++it1) { *it0 += *it1; } return *this;
			}
			template<typename U>
			AKS_FUNCTION_PREFIX_ATTR vec& operator-= (vec<U, dimensions> const& v) {
				auto it0 = data(), end = data() + dimensions; auto it1 = v.data(); for (; it0 != end; ++it0, ++it1) { *it0 -= *it1; } return *this;
			}
			template<typename U>
			AKS_FUNCTION_PREFIX_ATTR vec& operator*= (vec<U, dimensions> const& v) {
				auto it0 = data(), end = data() + dimensions; auto it1 = v.data(); for (; it0 != end; ++it0, ++it1) { *it0 *= *it1; } return *this;
			}
			template<typename U>
			AKS_FUNCTION_PREFIX_ATTR vec& operator/= (vec<U, dimensions> const& v) {
				auto it0 = data(), end = data() + dimensions; auto it1 = v.data(); for (; it0 != end; ++it0, ++it1) { *it0 /= *it1; } return *this;
			}
			template<typename U>
			AKS_FUNCTION_PREFIX_ATTR vec& operator*= (U u) {
				auto it0 = data(), end = data() + dimensions; for (; it0 != end; ++it0) { *it0 *= u; } return *this;
			}
			template<typename U>
			AKS_FUNCTION_PREFIX_ATTR vec& operator/= (U u) {
				auto it0 = data(), end = data() + dimensions; for (; it0 != end; ++it0) { *it0 /= u; } return *this;
			}

			AKS_FUNCTION_PREFIX_ATTR value_type sqlen() const {
				value_type sum = value_type(0);
				for (auto i : m_data) { sum += (i*i); }
				return sum;
			}

			AKS_FUNCTION_PREFIX_ATTR value_type len() const {
				return sqrt(sqlen());
			}

			AKS_FUNCTION_PREFIX_ATTR void make_unit() {
				(*this) /= len();
			}

			AKS_FUNCTION_PREFIX_ATTR vec as_unit() {
				vec v(*this);
				v.make_unit();
				return v;
			}

			value_type m_data[dimensions];
		};

		template<typename T, typename U, size_t N>
		AKS_FUNCTION_PREFIX_ATTR auto operator+(vec<T, N> const& v, vec<U, N> const& u) -> vec<decltype(T() + U()), N>
		{
			typedef vec<decltype(T() + U()), N> return_type;
			return_type ret(v);
			ret += u;
			return ret;
		}

		template<typename T, typename U, size_t N>
		AKS_FUNCTION_PREFIX_ATTR auto operator-(vec<T, N> const& v, vec<U, N> const& u) -> vec<decltype(T() - U()), N>
		{
			typedef vec<decltype(T() - U()), N> return_type;
			return_type ret(v);
			ret -= u;
			return ret;
		}

		template<typename T, typename U, size_t N>
		AKS_FUNCTION_PREFIX_ATTR auto dot(vec<T, N> const& v, vec<U, N> const& u) -> decltype(T() * U())
		{
			typedef decltype(T() * U()) return_type;
			return_type ret = return_type();
			auto it2 = u.cbegin();
			for (auto it = v.cbegin(), end = v.cend(); it != end; ++it, ++it2)
				ret += *it * *it2;
			return ret;
		}

		template<typename T, typename U>
		AKS_FUNCTION_PREFIX_ATTR auto cross(vec<T, 3> const& v, vec<U, 3> const& u) -> vec<decltype(T() * U()), 3>
		{
			typedef decltype(T() * U()) return_type;
			return vec<return_type, 3>(
				v.y()*u.z() - v.z()*u.y(),
				v.z()*u.x() - v.x()*u.z(),
				v.x()*u.y() - v.y()*u.x()
				);
		}

		template<typename T, typename U, size_t N>
		AKS_FUNCTION_PREFIX_ATTR auto elemwise_mult(vec<T, N> const& v, vec<U, N> const& u) -> vec<decltype(T() * U()), N>
		{
			typedef vec<decltype(T() * U()), N> return_type;
			return_type ret;
			auto it2 = u.cbegin();
			auto retit = ret.begin();
			for (auto it = v.cbegin(), end = v.cend(); it != end; ++it, ++it2, ++retit)
				*retit = *it * *it2;
			return ret;
		}

		template<typename T, typename U, size_t N>
		AKS_FUNCTION_PREFIX_ATTR auto elemwise_div(vec<T, N> const& v, vec<U, N> const& u) -> vec<decltype(T() / U()), N>
		{
			typedef vec<decltype(T() * U()), N> return_type;
			return_type ret;
			auto it2 = u.cbegin();
			auto retit = ret.begin();
			for (auto it = v.cbegin(), end = v.cend(); it != end; ++it, ++it2, ++retit)
				*retit = *it / *it2;
			return ret;
		}

		template<typename T, size_t N>
		AKS_FUNCTION_PREFIX_ATTR vec<T, N> unit_vector(vec<T, N> v) {
			v.make_unit();
			return v;
		}

		template<typename T, size_t N>
		AKS_FUNCTION_PREFIX_ATTR vec<T, N> len(vec<T, N> const& v) {
			return v.len();
		}

		template<typename T, size_t N>
		AKS_FUNCTION_PREFIX_ATTR vec<T, N> sqlen(vec<T, N> const& v) {
			return v.sqlen();
		}

		template<typename U, typename T, size_t N>
		AKS_FUNCTION_PREFIX_ATTR vec<T, N> operator*(vec<T, N> v, U u) {
			return v *= u;
		}

		template<typename U, typename T, size_t N>
		AKS_FUNCTION_PREFIX_ATTR vec<T, N> operator/(vec<T, N> v, U u) {
			return v /= u;
		}

		using vec3f = vec<float, 3>;
		using vec3i = vec<int, 3>;
		using vec3d = vec<double, 3>;
		using vec3u = vec<unsigned int, 3>;
		using vec3c = vec<char, 3>;
	}
}

#endif // !__vec_cuh__