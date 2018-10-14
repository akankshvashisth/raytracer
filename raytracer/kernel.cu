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

		struct scatter_record
		{
			bool scatter;
			vec3f attentuation;
			ray3f scattered;
		};

		AKS_FUNCTION_PREFIX_ATTR vec3f reflect(vec3f const& v, vec3f const& n)
		{
			return v - (n * 2. * dot(v, n));
		}

		struct material;

		struct hit_record
		{
			AKS_FUNCTION_PREFIX_ATTR hit_record(material const * m) : was_hit(false), t(), p(), normal(), mat(m) {}
			bool was_hit;
			float t;
			vec3f p;
			vec3f normal;
			material const * mat;
		};

		template<typename T>
		AKS_FUNCTION_PREFIX_ATTR T schlick(T cosine, T ref_idx) {
			T r0 = (1 - ref_idx) / (1 + ref_idx);
			r0 = r0 * r0;
			return r0 + (1 - r0)*pow((1 - cosine), 5);
		}

		struct refracted_record
		{
			bool has_refracted;
			vec3f refracted;
		};

		AKS_FUNCTION_PREFIX_ATTR refracted_record refract(const vec3f& v, const vec3f& n, float ni_over_nt) {
			refracted_record ret;
			vec3f uv = unit_vector(v);
			float dt = dot(uv, n);
			float discriminant = 1.0 - ni_over_nt * ni_over_nt*(1 - dt * dt);
			if (discriminant > 0) {
				ret.refracted = (uv - n * dt) * ni_over_nt - n * sqrt(discriminant);
				ret.has_refracted = true;
			}
			else
				ret.has_refracted = false;
			return ret;
		}

		struct material
		{
			enum type
			{
				metal,
				lambertian,
				dielectric,
				unknown
			};

			AKS_FUNCTION_PREFIX_ATTR static material make_lambertian(const vec3f& a)
			{
				return material(a, 0.0, 0.0, type::lambertian);
			}

			AKS_FUNCTION_PREFIX_ATTR static material make_null()
			{
				return material(vec3f(), 0.0, 0.0, type::unknown);
			}

			AKS_FUNCTION_PREFIX_ATTR static material make_metal(const vec3f& a, float f)
			{
				return material(a, f, 0.0, type::metal);
			}

			AKS_FUNCTION_PREFIX_ATTR static material make_dielectric(float ref_index)
			{
				return material(vec3f(), 0.0, ref_index, type::dielectric);
			}

			template<typename S>
			AKS_FUNCTION_PREFIX_ATTR scatter_record scatter(ray3f const& r_in, hit_record const& rec, S s) const
			{
				switch (mat_type)
				{
				case aks::rx::material::metal:
					return metal_scatter(r_in, rec, s);
				case aks::rx::material::lambertian:
					return lambertian_scatter(rec, s);
				case aks::rx::material::dielectric:
					return dielectric_scatter(r_in, rec, s);
				default:
					return { false, vec3f(), ray3f() };
				}
			}

			vec3f albedo;
			float scattering;
			float ref_index;
			type mat_type;

		private:
			AKS_FUNCTION_PREFIX_ATTR material(vec3f const& a, float ascattering, float ari, type mtype) : albedo(a), scattering(ascattering), ref_index(ari), mat_type(mtype)
			{
				if (scattering > 1)
					scattering = 1;
				if (scattering < 0)
					scattering = 0;
			}

			template<typename S>
			AKS_FUNCTION_PREFIX_ATTR scatter_record metal_scatter(const aks::rx::ray3f & r_in, const aks::rx::hit_record & rec, S &s) const
			{
				aks::rx::scatter_record ret;
				ret = { true, albedo, ray3f{ rec.p, reflect(unit_vector(r_in.direction()), rec.normal) + random_in_unit_sphere(s) * scattering } };
				ret.scatter = dot(ret.scattered.direction(), rec.normal) > 0;
				return ret;
			}

			template<typename S>
			AKS_FUNCTION_PREFIX_ATTR scatter_record lambertian_scatter(const aks::rx::hit_record & rec, S &s) const
			{
				aks::rx::scatter_record ret;
				vec3f target = rec.p + rec.normal + random_in_unit_sphere(s);
				ret = { true, albedo, ray3f{ rec.p, target - rec.p } };
				return ret;
			}

			template<typename S>
			AKS_FUNCTION_PREFIX_ATTR scatter_record dielectric_scatter(const aks::rx::ray3f & r_in, const aks::rx::hit_record & rec, S &s) const
			{
				aks::rx::scatter_record ret;
				vec3f reflected;
				vec3f outward_normal;
				reflected = reflect(r_in.direction(), rec.normal);
				float ni_over_nt;
				ret.attentuation = vec3f(1.0, 1.0, 1.0);
				//vec3f refracted;
				float reflect_prob;
				float cosine;
				if (dot(r_in.direction(), rec.normal) > 0) {
					outward_normal = -rec.normal;
					ni_over_nt = ref_index;
					//         cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
					cosine = dot(r_in.direction(), rec.normal) / r_in.direction().len();
					cosine = sqrt(1 - ref_index * ref_index *(1 - cosine * cosine));
				}
				else {
					outward_normal = rec.normal;
					ni_over_nt = 1.0 / ref_index;
					cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().len();
				}
				refracted_record refr = refract(r_in.direction(), outward_normal, ni_over_nt);
				if (refr.has_refracted)
					reflect_prob = schlick(cosine, ref_index);
				else
					reflect_prob = 1.0;
				if (curand_uniform(s) < reflect_prob)
					ret.scattered = ray3f(rec.p, reflected);
				else
					ret.scattered = ray3f(rec.p, refr.refracted);
				ret.scatter = true;
				return ret;
			}
		};

		struct sphere
		{
			AKS_FUNCTION_PREFIX_ATTR sphere() :center(), radius(0), mat(material::make_null()) {}
			AKS_FUNCTION_PREFIX_ATTR sphere(vec3f c, float r, material m) : center(c), radius(r), mat(m) {}
			AKS_FUNCTION_PREFIX_ATTR hit_record hit(ray3f const& r, float tmin, float tmax) const
			{
				hit_record ret(&mat);
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
			material mat;
		};

		struct spheres
		{
			struct view
			{
				AKS_FUNCTION_PREFIX_ATTR view(multi_dim_vector<sphere, 1> xs) :items(xs) {}
				AKS_FUNCTION_PREFIX_ATTR hit_record hit(ray3f const& r, float tmin, float tmax) const {
					hit_record ret(nullptr);
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
			double u = (curand_uniform(st) * 2.0) - 1.0;
			double t = curand_uniform(st) * 2 * 3.14159265;
			double sqrt1minususq = sqrt(1 - u * u);
			return vec3f(sqrt1minususq * cos(t), sqrt1minususq * sin(t), u);

			/*vec3f p;
			do {
				p = (vec3f(curand_uniform(st), curand_uniform(st), curand_uniform(st)) * 2.0) - vec3f(1, 1, 1);
			} while (p.sqlen() >= 1.0);
			return p;*/
		}

		template<typename W, typename R>
		AKS_FUNCTION_PREFIX_ATTR vec3f color(size_t max_depth, ray3f r, W const& w, R randC) {
			vec3f ret;
			hit_record rec(nullptr);
			vec3f attentuation(1.0f, 1.0f, 1.0f);
			for (size_t i = 0; i < max_depth; ++i) {
				rec = w.hit(r, 0.0001, FLT_MAX);
				if (!rec.was_hit) {
					vec3f unit_direction = unit_vector(r.direction());
					float t = 0.5f * (unit_direction.y() + 1.0);
					ret = elemwise_mult(attentuation, (vec3f(1.0f, 1.0f, 1.0f)*(1.0 - t) + vec3f(0.6, 0.8, 1.0)*t));
					break;
				}
				else
				{
					scatter_record sr = rec.mat->scatter(r, rec, randC);
					if (sr.scatter) {
						r = sr.scattered;
						attentuation = elemwise_mult(attentuation, sr.attentuation);
					}
				}
			}
			return ret;
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
	size_t max_ray_depth;
};

int main()
{
	//{
	//	std::vector<int> xs = { 1,127,512,786,1024,2000 };
	//	for (auto x : xs) {
	//		for (auto y : xs) {
	//			for (auto z : xs) {
	//				auto res = aks::calculateDims(x, y, z);
	//				std::cout << "{" << x << "," << y << "," << z << "} = "
	//					<< "{"
	//					<< std::get<0>(res).x
	//					<< "," << std::get<0>(res).y
	//					<< "," << std::get<0>(res).z
	//					<< "},{"
	//					<< std::get<1>(res).x
	//					<< "," << std::get<1>(res).y
	//					<< "," << std::get<1>(res).z
	//					<< "}(" << std::get<1>(res).x * std::get<1>(res).y * std::get<1>(res).z << ")"
	//					<< std::endl;
	//			}
	//		}
	//	}
	//	return 0;
	//}

	aks::cuda_context ctxt(aks::cuda_device(0));
	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<> dist(0, 1);

	size_t limit;
	cudaDeviceGetLimit(&limit, cudaLimit::cudaLimitStackSize);
	std::cout << limit << std::endl;
	cudaDeviceSetLimit(cudaLimit::cudaLimitStackSize, 1024);
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

	int nx = 2000, ny = 1500, ns = 10;
	size_t max_ray_depth = 20;
	aks::point<size_t, 2> tile, start;
	tile.x = nx;
	tile.y = ny;

	vec3f lookfrom(13, 2, 3);
	vec3f lookat(0, 0, 0);
	float dist_to_focus = 10.0;
	float aperture = 0.1;

	camera3f cam(lookfrom, lookat, vec3f(0, 1, 0), 20, float(nx) / float(ny), aperture, dist_to_focus);
	std::vector<sphere> s;
	{
		s.push_back(sphere(vec3f(0, -1000, 0), 1000, material::make_lambertian(vec3f(0.5, 0.5, 0.5))));
		for (int a = -11; a < 11; ++a)
		{
			for (int b = -11; b < 11; ++b)
			{
				float choose_mat = dist(e2);
				vec3f center(a + 0.9 * dist(e2), 0.2, b + 0.9 * dist(e2));
				if ((center - vec3f(4, 0.2, 0)).len() > 0.9) {
					if (choose_mat < 0.8) {
						s.push_back(sphere(center, 0.2, material::make_lambertian(vec3f(dist(e2)*dist(e2), dist(e2)*dist(e2), dist(e2)*dist(e2)))));
					}
					else if (choose_mat < 0.95) {
						s.push_back(sphere(center, 0.2, material::make_metal(vec3f(0.5*(1 + dist(e2)), 0.5*(1 + dist(e2)), 0.5*(1 + dist(e2))), 0.5 * dist(e2))));
					}
					else {
						s.push_back(sphere(center, 0.2, material::make_dielectric(1.5)));
					}
				}
			}
		}
		s.push_back(sphere(vec3f(0, 1, 0), 1.0, material::make_dielectric(1.5)));
		s.push_back(sphere(vec3f(-4, 1, 0), 1.0, material::make_lambertian(vec3f(0.4, 0.2, 0.1))));
		s.push_back(sphere(vec3f(4, 1, 0), 1.0, material::make_metal(vec3f(0.7, 0.6, 0.5), 0.0)));

		/*worldview.items(0) = sphere(vec3f(0, 0, -1), 0.5, material::make_lambertian(vec3f(0.1, 0.2, 0.5)));
		worldview.items(1) = sphere(vec3f(0, -100.5, -1), 100, material::make_lambertian(vec3f(0.8, 0.8, 0.0)));
		worldview.items(2) = sphere(vec3f(1, 0, -1), 0.5, material::make_metal(vec3f(0.8, 0.6, 0.2), 0.0));
		worldview.items(3) = sphere(vec3f(-1, 0, -1), 0.5, material::make_dielectric(1.5));
		worldview.items(4) = sphere(vec3f(-1, 1, -1), 0.45, material::make_dielectric(1.5));*/
	}
	spheres world(s.size());
	auto worldview = world.get_view();
	for (int i = 0; i < s.size(); ++i)
		worldview.items(i) = s[i];
	std::cout << s.size() << std::endl;

	aks::host_multi_dim_vector< vec3f, 2 > data(nx, ny);
	auto ddata = aks::to_device(data);
	auto dworld = aks::to_device(world.items);
	auto dworldview = spheres::view(dworld.view());

	for (size_t x = 0; x < nx; x += tile.x)
		for (size_t y = 0; y < ny; y += tile.y)
		{
			start.x = x;
			start.y = y;
			{
				aks::cuda_sync_context sync_ctxt;
				InData in = { nx, ny, ns, cam, dworldview, max_ray_depth };
				aks::cuda_object<InData> inData(in);
				auto din = inData.cview();
				aks::naryOpWithIndexTiled(ddata.view(), tile, start, [din] __device__(int i, int j) {
					unsigned int seed = din->ny * i + j;
					curandState st;
					curand_init(seed, 0, 0, &st);
					vec3f col(0, 0, 0);
					for (int s = 0; s < din->ns; ++s) {
						float u = float(i + curand_uniform(&st)) / float(din->nx);
						float v = float(j + curand_uniform(&st)) / float(din->ny);
						col += color(din->max_ray_depth, din->cam.get_ray(u, v, &st), din->dworldview, &st);
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
			std::cout << "[writing]..." << x << "," << y << "\n";
			toPPMFile("D:\\study\\out.ppm", view);
		}
	return 0;
}