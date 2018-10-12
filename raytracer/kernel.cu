#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>

#include "vec.cuh"
#include "ray.cuh"
#include "camera.cuh"
#include <cuda_library\multi_dim_vector.hpp>
#include <string>
#include <random>
#include <cuda_library/cuda_operators.hpp>
#include <curand_kernel.h>
#include <cuda_library/cuda_context.hpp>
#include <cuda_library/cuda_object.hpp>
//#include <curand_uniform.h>

//AKS_FUNCTION_PREFIX_ATTR double randZeroToOne()
//{
//	return rand() / (RAND_MAX + 1.);
//}

namespace aks
{
	namespace rx
	{
		template<typename T, size_t N>
		std::ostream& operator<<(std::ostream& o, vec<T, N> const& v) {
			auto it = v.cbegin(), end = v.cend();
			o << *it++;
			for (; it != end;) o << " " << *it++;
			/*o << "vec[" << N << "]{" << *it++;
			for (; it != end;) o << "," << *it++;
			o << "}";*/
			return o;
		}

		template<typename T, size_t N>
		std::istream& operator >> (std::istream& is, vec<T, N>& v) {
			for (auto it = v.begin(); it != v.end(); ++it)
				is >> *it;
			return is;
		}

		void toPPMFile(std::string const& outfile, multi_dim_vector<vec3f const, 2> const data)
		{
			int const nx = get_max_dim<0>(data), ny = get_max_dim<1>(data);
			std::ofstream myfile(outfile);
			myfile << "P3\n" << nx << " " << ny << "\n255\n";
			for (int j = ny - 1; j >= 0; --j)
				for (int i = 0; i < nx; ++i)
				{
					vec3i v = data(i, j) * 255.99f;
					myfile << v << "\n";
				}
			myfile.close();
		}

		struct hit_record
		{
			AKS_FUNCTION_PREFIX_ATTR hit_record() : was_hit(false), t(), p(), normal() {}
			bool was_hit;
			float t;
			vec3f p;
			vec3f normal;
		};

		struct sphere
		{
			AKS_FUNCTION_PREFIX_ATTR sphere() {}
			AKS_FUNCTION_PREFIX_ATTR sphere(vec3f c, float r) :center(c), radius(r) {}
			AKS_FUNCTION_PREFIX_ATTR hit_record hit(ray3f const& r, float tmin, float tmax) const
			{
				hit_record ret;
				vec3f oc = r.origin() - center;
				float a = dot(r.direction(), r.direction());
				float b = dot(oc, r.direction());
				float c = dot(oc, oc) - radius * radius;
				float dis = b * b - a * c;
				if (dis > 0)
				{
					float temp = (-b - sqrt(dis)) / a;
					if (temp < tmax && temp > tmin) {
						ret.t = temp;
						ret.p = r.at_parameter(temp);
						ret.normal = (ret.p - center) / radius;
						ret.was_hit = true;
						return ret;
					}
					temp = (-b + sqrt(dis)) / a;
					if (temp < tmax && temp > tmin) {
						ret.t = temp;
						ret.p = r.at_parameter(temp);
						ret.normal = (ret.p - center) / radius;
						ret.was_hit = true;
						return ret;
					}
				}
				ret.was_hit = false;
				return ret;
			}
			vec3f center;
			float radius;
		};

		struct spheres
		{
			struct view
			{
				AKS_FUNCTION_PREFIX_ATTR view(multi_dim_vector<sphere, 1> xs) :items(xs) {}
				AKS_FUNCTION_PREFIX_ATTR hit_record hit(ray3f const& r, float tmin, float tmax) const {
					hit_record ret;
					double closest_so_far = tmax;
					for (auto const& s : items)
					{
						hit_record hr = s.hit(r, tmin, closest_so_far);
						if (hr.was_hit) {
							closest_so_far = hr.t;
							ret = hr;
						}
					}
					return ret;
				}
				multi_dim_vector<sphere, 1> items;
			};
			spheres(size_t count) :items(count) {}
			view get_view() {
				return view(items.view());
			}
			host_multi_dim_vector<sphere, 1> items;
		};

		AKS_FUNCTION_PREFIX_ATTR bool hit_sphere(vec3f const& center, float radius, ray3f const& r) {
			vec3f oc = r.origin() - center;
			float a = dot(r.direction(), r.direction());
			float b = 2.f * dot(oc, r.direction());
			float c = dot(oc, oc) - radius * radius;
			float dis = b * b - 4 * a*c;
			return dis > 0;
		}

		__device__ vec3f random_in_unit_sphere(curandState* st)
		{
			//std::random_device rd;
			//std::mt19937 e2(rd());
			//std::uniform_real_distribution<> dist(0, 1);

			vec3f p;
			do {
				p = (vec3f(curand_uniform(st), curand_uniform(st), curand_uniform(st)) * 2.0) - vec3f(1, 1, 1);
			} while (p.sqlen() >= 1.0);
			return p;
		}

		template<typename W, typename R>
		AKS_FUNCTION_PREFIX_ATTR vec3f color(ray3f r, W const& w, R randC) {
			hit_record rec;
			float multiplier = 1.0f;
			for (size_t i = 0; i < 10; ++i) {
				rec = w.hit(r, 0.0001, FLT_MAX);
				if (!rec.was_hit) {
					vec3f unit_direction = unit_vector(r.direction());
					float t = 0.5f * (unit_direction.y() + 1.0);
					return (vec3f(1.0f, 1.0f, 1.0f)*(1.0 - t) + vec3f(0.5, 0.7, 1.0)*t) * multiplier;
				}
				vec3f target = rec.p + rec.normal + random_in_unit_sphere(randC);
				r = ray3f(rec.p, target - rec.p);
				multiplier *= 0.5f;
			}

			//return (hit_r.normal + vec3f(1, 1, 1))*0.5f;
			return vec3f(0, 0, 0);
		}
	}
}

struct InData
{
	int nx;
	int ny;
	int ns;
	aks::rx::camera3f cam;
	aks::rx::spheres::view dworldview;
};

int main()
{
	aks::cuda_context ctxt(aks::cuda_device(0));
	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<> dist(0, 1);

	size_t limit;
	cudaDeviceGetLimit(&limit, cudaLimit::cudaLimitStackSize);
	std::cout << limit << std::endl;
	cudaDeviceSetLimit(cudaLimit::cudaLimitStackSize, 512);
	cudaDeviceGetLimit(&limit, cudaLimit::cudaLimitStackSize);
	std::cout << limit << std::endl;
	cudaDeviceSynchronize();
	cudaDeviceGetLimit(&limit, cudaLimit::cudaLimitStackSize);
	std::cout << limit << std::endl;
	gpu_error_check(aks::last_status());

	using namespace aks::rx;
	using namespace aks;
	vec3d v, v2(2., 3., 4.); // , v3(2, 3), v4(2, 3, 4, 5);
	auto m1 = -v2;
	m1 += v2;
	v2.make_unit();
	std::cout << v << std::endl;
	std::cout << v2 << std::endl;
	std::cout << dot(v2, v2) << std::endl;
	std::cout << v2.len() << std::endl;
	int nx = 1000, ny = 500, ns = 500;
	camera3f cam;
	sphere s(vec3f(0, 0, -1), 0.5);
	sphere s2(vec3f(0, -100.5, -1), 100);
	sphere s3(vec3f(1, 0, -2), 1);
	sphere s4(vec3f(-1, 0, -1), 0.1);
	spheres world(4);
	auto worldview = world.get_view();
	worldview.items(0) = s;
	worldview.items(1) = s2;
	worldview.items(2) = s3;
	worldview.items(3) = s4;

	aks::host_multi_dim_vector< vec3f, 2 > data(nx, ny);
	auto ddata = aks::to_device(data);
	auto dworld = aks::to_device(world.items);
	auto dworldview = spheres::view(dworld.view());
	{
		aks::cuda_sync_context sync_ctxt;
		InData in = { nx, ny, ns, cam, dworldview };
		aks::cuda_object<InData> inData(in);
		auto din = inData.cview();
		aks::naryOpWithIndex(ddata.view(), [din] __device__(int i, int j) {
			unsigned int seed = din->ny * i + j;
			curandState st;
			curand_init(seed, 0, 0, &st);
			vec3f col(0, 0, 0);
			for (int s = 0; s < din->ns; ++s) {
				float u = float(i + curand_uniform(&st)) / float(din->nx);
				float v = float(j + curand_uniform(&st)) / float(din->ny);
				col += color(din->cam.get_ray(u, v), din->dworldview, &st);
			}
			return (col /= float(din->ns)).map_func([](float f) {return sqrt(f); });
		});
	}
	data << ddata;
	auto view = data.view();
	/*
	for (int j = ny - 1; j >= 0; --j)
		for (int i = 0; i < nx; ++i) {
			vec3f col(0, 0, 0);
			for (int s = 0; s < ns; ++s) {
				float u = float(i + randZeroToOne()) / float(nx);
				float v = float(j + randZeroToOne()) / float(ny);
				col += color(cam.get_ray(u, v), worldview);
			}
			view(i, j) = (col /= float(ns)).map_func([](float f) {return sqrt(f); });
		}*/
	std::cout << "[writing]...";
	toPPMFile("D:\\study\\out.ppm", view);
	return 0;
}