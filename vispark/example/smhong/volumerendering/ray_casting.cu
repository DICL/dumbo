#include "/home/smhong/vislib/translator/helper_math.h"
#define GPU inline __device__
#define uchar unsigned char

#define BORDER_REPLICATE 1
#define BORDER_REFLECT 2
#define BORDER_REFLECT_101 3
#define BORDER_WRAP 4

#include<stdio.h>
#include<vector_types.h>
#include<vector_functions.h>
__device__ int DEVICE_NUMBER;
__device__ float modelview[4][4];
__device__ float inv_modelview[4][4]; 
__device__ int slider[4];
__device__ int slider1[4];

__device__ float TF_bandwidth;
__device__ int front_back;

__device__ float4 planes[2048];
__device__ float3 curves[2048];
__device__ int plane_index[2048];
__device__ int plane_cnt;

__device__ float3 cut_planes[2048];


__device__ float2 wr_start;

__device__ int target_plane;


// STREAMING
__device__ int middle;
__device__ int stride;

texture<float, 2> depthMap;
texture<float4, 2> colorMap;

// clamp
GPU float clamp(float f, float a, float b);
GPU float2 clamp(float2 v, float a, float b);
GPU float3 clamp(float3 v, float a, float b);
GPU float3 clamp(float3 v, float3 a, float3 b);
GPU float4 clamp(float4 v, float a, float b);
GPU float4 clamp(float4 v, float4 a, float4 b);

// rgba class
class RGBA{
public:
    unsigned char r, g, b, a;
	GPU RGBA(float3 rgb, float a_in){
		r = clamp(rgb.x, 0.0f, 255.0f);
		g = clamp(rgb.y, 0.0f, 255.0f);
		b = clamp(rgb.z, 0.0f, 255.0f);
		a = clamp(a_in, 0.0f, 255.0f);
	}
	GPU RGBA(float4 rgba){
		r = clamp(rgba.x, 0.0f, 255.0f);
		g = clamp(rgba.y, 0.0f, 255.0f);
		b = clamp(rgba.z, 0.0f, 255.0f);
		a = clamp(rgba.w, 0.0f, 255.0f);
	}
	GPU RGBA(float r_in, float g_in, float b_in, float a_in){
		r = clamp(r_in, 0.0f, 255.0f);
		g = clamp(g_in, 0.0f, 255.0f);
		b = clamp(b_in, 0.0f, 255.0f);
		a = clamp(a_in, 0.0f, 255.0f);
	}
	GPU RGBA(float c){
		r = clamp(c, 0.0f, 255.0f);
		g = clamp(c, 0.0f, 255.0f);
		b = clamp(c, 0.0f, 255.0f);
		a = clamp(255.0f, 0.0f, 255.0f);
	}
	GPU RGBA(){
        r = g = b = 0;
        a = 1;
    }

};
// rgb class
class RGB{
public:
    unsigned char r, g, b;
	GPU RGB(float3 rgb)
	{
		r = clamp(rgb.x, 0.0f, 255.0f);
		g = clamp(rgb.y, 0.0f, 255.0f);
		b = clamp(rgb.z, 0.0f, 255.0f);
	}
	GPU RGB(float4 rgb)
	{
		r = clamp(rgb.x, 0.0f, 255.0f);
		g = clamp(rgb.y, 0.0f, 255.0f);
		b = clamp(rgb.z, 0.0f, 255.0f);
	}

	GPU RGB(float r_in, float g_in, float b_in)
	{
		r = clamp(r_in, 0.0f, 255.0f);
		g = clamp(g_in, 0.0f, 255.0f);
		b = clamp(b_in, 0.0f, 255.0f);
	}
	GPU RGB(float a)
    {
        r = g = b = clamp(a, 0.0f, 255.0f);
    }
	GPU RGB()
    {
        r = g = b = 0;
    }

};

class VIVALDI_DATA_RANGE{
public:
	int4 data_start, data_end;
	int4 full_data_start, full_data_end;
	int4 buffer_start, buffer_end;
	int data_halo;
	int buffer_halo;
	
};


template <typename T>
class Slice{
public:
	GPU Slice(T* data) 
	{
		target_z = target_plane;
		data_ptr = data;
	}

public:
	int target_z;
	T* data_ptr;
	
};

// data type converters 
////////////////////////////////////////////////////////////////////////////////
GPU float convert(char1 a){
	return float(a.x);
}
GPU float convert(uchar1 a){
	return float(a.x);
}
GPU float convert(short1 a){
	return float(a.x);
}
GPU float convert(ushort1 a){
	return float(a.x);
}
GPU float convert(int1 a){
	return float(a.x);
}
GPU float convert(uint1 a){
	return float(a.x);
}
GPU float convert(float1 a){
	return float(a.x);
}
GPU float convert(double1 a){
	return float(a.x);
}
GPU float convert(double a){
	return float(a);
}

GPU float2 convert(char2 a){
    return make_float2(a.x, a.y);
}
GPU float2 convert(uchar2 a){
    return make_float2(a.x, a.y);
}
GPU float2 convert(short2 a){
    return make_float2(a.x,a.y);
}
GPU float2 convert(ushort2 a){
    return make_float2(a.x,a.y);
}
GPU float2 convert(int2 a){
    return make_float2(a.x,a.y);
}
GPU float2 convert(uint2 a){
    return make_float2(a.x,a.y);
}
GPU float2 convert(float2 a){
    return make_float2(a.x,a.y);
}
GPU float2 convert(double2 a){
    return make_float2(a.x,a.y);
}

GPU float3 convert(RGB a){
    return make_float3(a.r,a.g,a.b);
}
GPU float3 convert(char3 a){
	return make_float3(a.x, a.y, a.z);
}
GPU float3 convert(uchar3 a){
	return make_float3(a.x, a.y, a.z);
}
GPU float3 convert(short3 a){
	return make_float3(a.x, a.y, a.z);
}
GPU float3 convert(ushort3 a){
    return make_float3(a.x, a.y, a.z);
}
GPU float3 convert(int3 a){
    return make_float3(a.x, a.y, a.z);
}
GPU float3 convert(uint3 a){
	return make_float3(a.x, a.y, a.z);
}
GPU float3 convert(float3 a){
    return make_float3(a.x, a.y, a.z);
}
GPU float3 convert(double3 a){
    return make_float3(a.x, a.y, a.z);
}

GPU float4 convert(RGBA a){
    return make_float4(a.r,a.g,a.b,a.a);
}
GPU float4 convert(char4 a){
    return make_float4(a.x, a.y, a.z, a.w);
}
GPU float4 convert(uchar4 a){
     return make_float4(a.x, a.y, a.z, a.w);
}
GPU float4 convert(short4 a){
    return make_float4(a.x, a.y, a.z, a.w);
}
GPU float4 convert(ushort4 a){
    return make_float4(a.x, a.y, a.z, a.w);
}
GPU float4 convert(int4 a){
    return make_float4(a.x, a.y, a.z, a.w);
}
GPU float4 convert(uint4 a){
    return make_float4(a.x, a.y, a.z, a.w);
}
GPU float4 convert(float4 a){
    return make_float4(a.x, a.y, a.z, a.w);
}
GPU float4 convert(double4 a){
    return make_float4(a.x, a.y, a.z, a.w);
}

// data_type init
//////////////////////////////////////////////////////////////
GPU float initial(uchar1 a){
	return 0.0;
}
GPU float initial(char1 a){
	return 0.0;
}
GPU float initial(ushort1 a){
	return 0.0;
}
GPU float initial(short1 a){
	return 0.0;
}
GPU float initial(int1 a){
	return 0.0;
}
GPU float initial(uint1 a){
	return 0.0;
}
GPU float initial(float1 a){
	return 0.0;
}
GPU float initial(float a){
	return 0.0;
}
GPU float initial(double1 a){
	return 0.0;
}


GPU float2 initial(char2 a){
	return make_float2(0);
}
GPU float2 initial(uchar2 a){
	return make_float2(0);
}
GPU float2 initial(short2 a){
	return make_float2(0);
}
GPU float2 initial(ushort2 a){
	return make_float2(0);
}
GPU float2 initial(int2 a){
	return make_float2(0);
}
GPU float2 initial(uint2 a){
	return make_float2(0);
}
GPU float2 initial(float2 a){
	return make_float2(0);
}
GPU float2 initial(double2 a){
	return make_float2(0);
}

GPU float3 initial(RGB a){
    return make_float3(0);
}
GPU float3 initial(char3 a){
    return make_float3(0);
}
GPU float3 initial(uchar3 a){
    return make_float3(0);
}
GPU float3 initial(short3 a){
    return make_float3(0);
}
GPU float3 initial(ushort3 a){
    return make_float3(0);
}
GPU float3 initial(int3 a){
    return make_float3(0);
}
GPU float3 initial(uint3 a){
    return make_float3(0);
}
GPU float3 initial(float3 a){
    return make_float3(0);
}
GPU float3 initial(double3 a){
    return make_float3(0);
}

GPU float4 initial(RGBA a){
	return make_float4(0);
}
GPU float4 initial(char4 a){
	return make_float4(0);
}
GPU float4 initial(uchar4 a){
	return make_float4(0);
}
GPU float4 initial(short4 a){
    return make_float4(0);
}
GPU float4 initial(ushort4 a){
    return make_float4(0);
}
GPU float4 initial(int4 a){
    return make_float4(0);
}
GPU float4 initial(uint4 a){
    return make_float4(0);
}
GPU float4 initial(float4 a){
    return make_float4(0);
}
GPU float4 initial(double4 a){
    return make_float4(0);
}


GPU float initial2(float a){
	return 1.0;
}

// float functions
////////////////////////////////////////////////////////////////////////////////

GPU float length(float a){
	if(a < 0){
		return -a;
	}
	return a;
}

//f = value, a = min, b = max
GPU float step(float edge, float x){
    return x < edge ? 0 : 1;
}

GPU float rect(float edge0, float edge1, float x){
    return edge0 <= x && x <= edge1 ? 1 : 0;
}

// float2 functions
////////////////////////////////////////////////////////////////////////////////

// negate
GPU float2 operator-(float2 a){
    return make_float2(-a.x, -a.y);
}

// floor
GPU float2 floor(const float2 v){
    return make_float2(floor(v.x), floor(v.y));
}

// reflect
GPU float2 reflect(float2 i, float2 n){
	return i - 2.0f * n * dot(n,i);
}

// float3 functions
////////////////////////////////////////////////////////////////////////////////

// floor
GPU float3 floor(const float3 v){
    return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

// float4 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors


// negate
GPU float4 operator-(float4 a){
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

// floor
GPU float4 floor(const float4 v){
    return make_float4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}


// Frame
////////////////////////////////////////////////////////////////////////////////

class Frame{
public:
    float3 x, y, z, origin;
    
    GPU void setDefault(float3 position)
    {
        origin = position;
        x = make_float3(1, 0, 0);
        y = make_float3(0, 1, 0);
        z = make_float3(0, 0, 1);
    }
    
    GPU void lookAt(float3 position, float3 target, float3 up)
    {
        origin = position;
        z = normalize(position - target);
        x = normalize(cross(up, z));
        y = normalize(cross(z, x));
    }
    
    GPU float3 getVectorToWorld(float3 v)
    {
        return v.x * x + v.y * y + v.z * z;
    }
    
    GPU float3 getPointToWorld(float3 p)
    {
        return p.x * x + p.y * y + p.z * z + origin;
    }
};

// transfer2
////////////////////////////////////////////////////////////////////////////////

GPU float transfer2(
    float x0, float f0,
    float x1, float f1,
    float x)
{
    if (x < x0) return 0;
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    return 0;
}

GPU float2 transfer2(
    float x0, float2 f0,
    float x1, float2 f1,
    float x)
{
    if (x < x0) return make_float2(0);
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    return make_float2(0);
}

GPU float3 transfer2(
    float x0, float3 f0,
    float x1, float3 f1,
    float x)
{
    if (x < x0) return make_float3(0);
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    return make_float3(0);
}

GPU float4 transfer2(
    float x0, float4 f0,
    float x1, float4 f1,
    float x)
{
    if (x < x0) return make_float4(0);
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    return make_float4(0);
}


// transfer3
////////////////////////////////////////////////////////////////////////////////

GPU float transfer3(
    float x0, float f0,
    float x1, float f1,
    float x2, float f2,
    float x)
{
    if (x < x0) return 0;
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    if (x < x2) return lerp(f1, f2, (x - x1) / (x2 - x0));
    return 0;
}

GPU float2 transfer3(
    float x0, float2 f0,
    float x1, float2 f1,
    float x2, float2 f2,
    float x)
{
    if (x < x0) return make_float2(0);
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    if (x < x2) return lerp(f1, f2, (x - x1) / (x2 - x0));
    return make_float2(0);
}

GPU float3 transfer3(
    float x0, float3 f0,
    float x1, float3 f1,
    float x2, float3 f2,
    float x)
{
    if (x < x0) return make_float3(0);
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    if (x < x2) return lerp(f1, f2, (x - x1) / (x2 - x0));
    return make_float3(0);
}

GPU float4 transfer3(
    float x0, float4 f0,
    float x1, float4 f1,
    float x2, float4 f2,
    float x)
{
    if (x < x0) return make_float4(0);
    if (x < x1) return lerp(f0, f1, (x - x0) / (x1 - x0));
    if (x < x2) return lerp(f1, f2, (x - x1) / (x2 - x0));
    return make_float4(0);
}

// helper textures for cubic interpolation and random numbers
////////////////////////////////////////////////////////////////////////////////

texture<float4, 2, cudaReadModeElementType> hgTexture;
texture<float4, 2, cudaReadModeElementType> dhgTexture;
texture<int, 2, cudaReadModeElementType> randomTexture;


GPU float3 hg(float a){
    // float a2 = a * a;
    // float a3 = a2 * a;
    // float w0 = (-a3 + 3*a2 - 3*a + 1) / 6;
    // float w1 = (3*a3 - 6*a2 + 4) / 6;
    // float w2 = (-3*a3 + 3*a2 + 3*a + 1) / 6;
    // float w3 = a3 / 6;
    // float g = w2 + w3;
    // float h0 = (1.0f + a) - w1 / (w0 + w1);
    // float h1 = (1.0f - a) + w3 / (w2 + w3);
    // return make_float3(h0, h1, g);
    return make_float3(tex2D(hgTexture, a, 0));
}
GPU float3 dhg(float a){
    return make_float3(tex2D(dhgTexture, a, 0));
}

//iterators
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// class make_laplacian_iter{}
class line_iter{
public:
	float3 S,E,P,step;
	float len;
	int cnt, full_cnt;

	// Temporal
	float depth;
	GPU line_iter(){
	}
	GPU line_iter(float3 from, float3 to, float d){
		S = from;
		E = to;
		
		step = normalize(to-from)*d;
	
		full_cnt =(int)round(length(to-from)/d);
		cnt = 0;

		len = length(to-from);
		
		P = S;
		if( S.x == E.x &&
			S.y == E.y &&
			S.z == E.z )step = make_float3(0,0,1);
			
	}
	GPU float3 begin(){
		return S;
	}
	GPU bool valid(){
		if(cnt < full_cnt)return true;
		return false;
	}
	GPU float3 next(){
		cnt += 1;
		P = S + cnt*step;
		return P;
	}
	GPU float3 direction(){
		return normalize(step);
	}
	GPU float3 prev() {
		return P - step;
	}
	GPU float3 tmp_next() {
		float3 T = S + (cnt+1)*step;
		return T;
	}
	GPU float get_depth() {
		return depth;
	}
	GPU void set_depth(float in) {
		depth = in;
	}
};
class plane_iter{
public:
	float2 S;
	float d;
	int max_step, step;
	int width;
	float x,y;
	GPU plane_iter(){
	}
	GPU plane_iter(float2 point, float size){
		S = point;
		d = size;
		width = 1+2*size;
		max_step = width*width;
	}
	GPU plane_iter(int x, int y, float size){
		S = make_float2(x,y);
		d = size;
		width = 1+2*size;
		max_step = width*width;
	}
	GPU plane_iter(float x, float y, float size){
		S = make_float2(x,y);
		d = size;
		width = 1+2*size;
		max_step = width*width;
	}

	GPU float2 begin(){
		step = 0;
		x = 0;
		y = 0;
		return S + make_float2(-d,-d);
	}
	GPU bool hasNext(){
		if(max_step == step)return false;
		return true;
	}
	GPU bool valid(){
		if(max_step <= step)return false;
		return true;
	}
	GPU float2 next(){
		step++;
		x++;
		if( x == width){ x=0; y++;}
		float2 P = S + make_float2( x - d, y - d);
		return P;
	}
};
class cube_iter{
public:
	float3 S;
	int d;
	int width;
	int max_step, step;
	float x,y,z;
	GPU cube_iter(){
	}
	GPU cube_iter(float3 point, float size){
		S = point;
		d = size;
		width = 1+2*size;
		max_step = (width)*(width)*(width);
	}
	GPU cube_iter(int x,int y,int z, float radius){
		S = make_float3(x,y,z);
		d = radius;
		width = 1+2*radius;
		max_step = (width)*(width)*(width);
	
	}
	GPU cube_iter(float x, float y, float z, float size){
		S = make_float3(x,y,z);
		d = size;
		width = 1+2*size;
		max_step = (width)*(width)*(width);
	
	}
	GPU float3 begin(){
		step = 0;
		x = 0;
		y = 0;
		z = 0;
		return S + make_float3(-d, -d, -d);
	}
	GPU bool hasNext(){
		if(max_step == step)return false;
		return true;
	}
	GPU bool valid(){
		if(max_step == step)return false;
		return true;
	}
	GPU float3 next(){
		step++;
		x++;
		if( x == width){ x=0; y++;}
		if( y == width){ y=0; z++;}

		float3 P = S + make_float3( x - d, y - d, z - d);
		return P;
	}
};

// data query functions
//////////////////////////////////////////////////////////////////////////////
#define INF            __int_as_float(0x7f800000)

GPU int2 float2_to_int2(float2 a){
	return make_int2(int(a.x), int(a.y));
}
GPU int3 float3_to_int3(float3 a){
    return make_int3(int(a.x), int(a.y), int(a.z));
}

// BORDER handling functions
GPU int border_replicate(int x, int pivot){
	// aaaaaa|abcdefgh|hhhhhhh
	return pivot;
}
GPU int border_reflect(int x, int pivot){
	// fedcba|abcdefgh|hgfedcb
	int a;
	a = 0;
	if(x < pivot) a = -1;
	if(x > pivot) a = 1;
		
	return 2*pivot-x + a;
}
GPU int border_reflect_101(int x, int pivot){
	// gfedcb|abcdefgh|gfedcba
	return 2*pivot-x;
}
GPU int border_wrap(int x, int pivot, int w){
	// cdefgh|abcdefgh|abcdefg
	if(x > pivot)return x - w;
	if(x < pivot)return x + w;
	return pivot;
}
GPU int border_constant(){
// iiiiii|abcdefgh|iiiiiii  with some specified 'i'
	return 0;
}

GPU int border_switch(int x, int pivot, int w, int border){

	switch(border){
		case BORDER_REPLICATE: // replicate
			return border_replicate(x, pivot);
		case BORDER_REFLECT: // reflect
			return border_reflect(x, pivot);
		case BORDER_REFLECT_101: // reflect_101
			return border_reflect_101(x, pivot);
		case BORDER_WRAP: // border_wrap
			return border_wrap(x, pivot, w);
		default: // border_constant
			return border_constant();
	}
}
GPU int check_in_border(int p, int start, int end){
	if(p > end-1)return 1; // right
	else if(p < start)return -1; // left
	return 0; // middle
}
GPU int border_handling(int p, int start, int end, int border){
	int flag = -1;
	while(flag != 0){
		flag = check_in_border(p, start, end);
		
		if(flag == 1){
			p = border_switch(p, end-1, end - start, border);
		}else if(flag == -1){
			p = border_switch(p, start, end - start, border);
		}
	}
	return p;
}

// Range check
GPU bool range_check(int p, int data_start, int data_end){
	if(data_start <= p && p < data_end){
		return true;
	}
	return false;
}

// 1D data query function
////////////////////////////////////////////////////////////////////////////////
template<typename R,typename T> GPU R point_query_1d(T* data, float x, int border, VIVALDI_DATA_RANGE* sdr){

//	int data_start = int(sdr->data_start.x);

	int full_data_start = int(sdr->full_data_start.x);
	int full_data_end = int(sdr->full_data_end.x);
	
	int buffer_start = int(sdr->buffer_start.x);
	
	R rt;
		// input coordinate is world coordinate
	
	// border handling
	int flag1; // -1, 0, 1
	flag1 = check_in_border(x, full_data_start, full_data_end);
	
	bool flag; // border 
	flag = (1 <= border && border <= 4);

	if( (!flag && (flag1 == 0)) || (flag)){
		// Border calculation
		x = border_handling(x, full_data_start, full_data_end, border);
		
		// to Buffer coordinate 
		x = x - buffer_start;
		
		rt = convert(data[int(x)]);
	}else{
		rt = initial(data[0]);
	}
	return rt;
}
template<typename R,typename T> GPU R point_query_1d(T* data, float x, VIVALDI_DATA_RANGE* sdr){
	return point_query_1d<R>(data, x, 0, sdr);
}

// 2D data query functions
////////////////////////////////////////////////////////////////////////////////

// STREAMING
GPU void change_plane(int relative_cnt) {
	return ;
}

template<typename R,typename T> GPU R s_point_query_2d(T* data, float2 p, int offset, int border, VIVALDI_DATA_RANGE* sdr){
	int4 data_start = sdr->data_start;
	int4 data_end = sdr->data_end;

	int4 full_data_start = sdr->full_data_start;
	int4 full_data_end = sdr->full_data_end;

	int data_halo = sdr->data_halo;

	int x = p.x;
	int y = p.y;
	
	int Y = data_end.y - data_start.y;
	int X = data_end.x - data_start.x;
	R rt;

	int flag1, flag2; // -1, 0, 1
	flag1 = check_in_border(x, full_data_start.x, full_data_end.x);
	flag2 = check_in_border(y, full_data_start.y, full_data_end.y);


	bool flag; // border 
	flag = (1 <= border && border <= 4);
	if( (!flag && (flag1 == 0 && flag2 == 0)) || (flag)){
	
		// Border calculation
		x = border_handling(x, full_data_start.x, full_data_end.x, border);
		y = border_handling(y, full_data_start.y, full_data_end.y, border);
		
		// Data range check
		
		bool flag_x = range_check(x, data_start.x, data_end.x);
		bool flag_y = range_check(y, data_start.y, data_end.y);
		
		if(flag_x && flag_y){
			// to Buffer coordinate 
			x = x - data_start.x;
			y = y - data_start.y;
			int target_z = (middle + offset) % (data_halo * 2 + 1);
			if(target_z < 0)
				target_z = (data_halo * 2 + 1) + target_z;
			rt = convert(data[target_z * Y * X + y * X + x]);
			//rt = convert(data[9 * Y * X + y * X + x]);
			
		}else{
			rt = initial(data[0]);
		}
		
	}else{
		rt = initial(data[0]);
	}
	


	
	return rt;
	// Data coordinate input
	// border check
	//int flag1, flag2; // -1, 0, 1
	//flag1 = check_in_border(x, full_data_start.x, full_data_end.x);
	//flag2 = check_in_border(y, full_data_start.y, full_data_end.y);
	
	//bool flag; // border 
	//flag = (1 <= border && border <= 4);
	
	//if( (!flag && (flag1 == 0 && flag2 == 0)) || (flag)){
	
		// Border calculation
		//x = border_handling(x, full_data_start.x, full_data_end.x, border);
		//y = border_handling(y, full_data_start.y, full_data_end.y, border);
		
		// Data range check
		
		//bool flag_x = range_check(x, data_start.x, data_end.x);
		//bool flag_y = range_check(y, data_start.y, data_end.y);
		
		//if(flag_x && flag_y){
			// to Buffer coordinate 
			//x = x - data_start.x;
			//y = y - data_start.y;
			
			//rt = convert(data[y*X + x]);
		//}else{
			//rt = initial(data[0]);
		//}
		
	//}else{
		//rt = initial(data[0]);
	//}
	
	//return rt;
}
template<typename R,typename T> GPU R s_point_query_2d(T* image, float2 p, int offset, VIVALDI_DATA_RANGE* sdr){
	return s_point_query_2d<R>(image, p, offset, 0, sdr);
}
template<typename R,typename T> GPU R s_point_query_2d(T* image, float x, float y, int offset, int border, VIVALDI_DATA_RANGE* sdr){
	return s_point_query_2d<R>(image, make_float2(x,y), offset, border, sdr);
}
template<typename R,typename T> GPU R s_point_query_2d(T* image, float x, float y, int offset, VIVALDI_DATA_RANGE* sdr){
	return s_point_query_2d<R>(image, make_float2(x,y), offset, 0, sdr);
}


template<typename R,typename T> GPU R point_query_2d(T* data, float2 p, int border, VIVALDI_DATA_RANGE* sdr){
	int4 data_start = sdr->data_start;
	int4 data_end = sdr->data_end;

	int4 full_data_start = sdr->full_data_start;
	int4 full_data_end = sdr->full_data_end;

	int x = p.x;
	int y = p.y;
	
	int X = data_end.x - data_start.x;
	R rt;
	
	// Data coordinate input
	// border check
	int flag1, flag2; // -1, 0, 1
	flag1 = check_in_border(x, full_data_start.x, full_data_end.x);
	flag2 = check_in_border(y, full_data_start.y, full_data_end.y);
	
	bool flag; // border 
	flag = (1 <= border && border <= 4);
	
	if( (!flag && (flag1 == 0 && flag2 == 0)) || (flag)){
	
		// Border calculation
		x = border_handling(x, full_data_start.x, full_data_end.x, border);
		y = border_handling(y, full_data_start.y, full_data_end.y, border);
		
		// Data range check
		
		bool flag_x = range_check(x, data_start.x, data_end.x);
		bool flag_y = range_check(y, data_start.y, data_end.y);
		
		if(flag_x && flag_y){
			// to Buffer coordinate 
			x = x - data_start.x;
			y = y - data_start.y;
			
			rt = convert(data[y*X + x]);
		}else{
			rt = initial(data[0]);
		}
		
	}else{
		rt = initial(data[0]);
	}
	
	return rt;
}
template<typename R,typename T> GPU R point_query_2d(T* image, float2 p, VIVALDI_DATA_RANGE* sdr){
	return point_query_2d<R>(image, p, 0, sdr);
}
template<typename R,typename T> GPU R point_query_2d(T* image, float x, float y, int border, VIVALDI_DATA_RANGE* sdr){
	return point_query_2d<R>(image, make_float2(x,y), border, sdr);
}
template<typename R,typename T> GPU R point_query_2d(T* image, float x, float y, VIVALDI_DATA_RANGE* sdr){
	return point_query_2d<R>(image, make_float2(x,y), 0, sdr);
}


template<typename R,typename T> GPU R linear_query_2d(T* image, float2 p, int border, VIVALDI_DATA_RANGE* sdr){
	//range check
	float x = p.x;
	float y = p.y;

	int fx = floor(x);
	int fy = floor(y);
	int cx = ceil(x);
	int cy = ceil(y);

	float dx = x - fx;
	float dy = y - fy;

	R iv = initial(image[0])*0;
	
	R q00 = point_query_2d<R>(image, fx, fy, border, sdr);
	R q01 = point_query_2d<R>(image, cx, fy, border, sdr);
	R q10 = point_query_2d<R>(image, fx, cy, border, sdr);
	R q11 = point_query_2d<R>(image, cx, cy, border, sdr);
	
	// lerp along x
	R q0 = lerp(q00, q01, dx);
	R q1 = lerp(q10, q11, dx);

	// lerp along y
	R q = lerp(q0, q1, dy);
	return q;
}
template<typename R, typename T> GPU R linear_query_2d(T* image, float2 p, VIVALDI_DATA_RANGE* sdr){
	return linear_query_2d<R>(image, p, 0, sdr);
}
template<typename R, typename T> GPU R linear_query_2d(T* image, float x, float y, int border, VIVALDI_DATA_RANGE* sdr){
	return linear_query_2d<R>(image, make_float2(x,y), border, sdr);
}
template<typename R, typename T> GPU R linear_query_2d(T* image, float x, float y, VIVALDI_DATA_RANGE* sdr){
	return linear_query_2d<R>(image, make_float2(x,y), 0, sdr);
}

template<typename R, typename T> GPU float2 linear_gradient_2d(T* image, float2 p, VIVALDI_DATA_RANGE* sdr){

	int4 data_start = sdr->data_start;
	int4 data_end = sdr->data_end;

	int halo = sdr->data_halo;

	float x = p.x - data_start.x;
    float y = p.y - data_start.y;
    int X = data_end.x - data_start.x;
    int Y = data_end.y - data_start.y;

    float2 rbf = make_float2(0);
  
	if( x < halo)return rbf;
    if( y < halo)return rbf;
    if( x >= X-halo)return rbf;
    if( y >= Y-halo)return rbf;
 
	float delta = 1.0f;

	R xf, xb;
	xf = linear_query_2d<R>(image, make_float2(p.x + delta, p.y), sdr);
	xb = linear_query_2d<R>(image, make_float2(p.x - delta, p.y), sdr);
	float dx = length(xf-xb);

	R yf, yb;
	yf = linear_query_2d<R>(image, make_float2(p.x, p.y + delta), sdr);
	yb = linear_query_2d<R>(image, make_float2(p.x, p.y - delta), sdr);
	float dy = length(yf-yb);

	return make_float2(dx,dy)/(2*delta);
}
template<typename R, typename T> GPU float2 linear_gradient_2d(T* image, float x, float y, VIVALDI_DATA_RANGE* sdr){
	return linear_gradient_2d<R>(image, make_float2(x,y), sdr);
}
template<typename R, typename T> GPU float2 linear_gradient_2d(T* image, int x, int y, VIVALDI_DATA_RANGE* sdr){
	return linear_gradient_2d<R>(image, make_float2(x,y), sdr);
}

// 3D data query functions
////////////////////////////////////////////////////////////////////////////////

template<typename R, typename T> GPU R checker(T* image, float3 p, int offset, int border, VIVALDI_DATA_RANGE* sdr){
	int4 data_start = sdr->data_start;
	int4 data_end = sdr->data_end;

	int4 full_data_start = sdr->full_data_start;
	int4 full_data_end = sdr->full_data_end;
	
	int4 buffer_start = sdr->buffer_start;
	int4 buffer_end = sdr->buffer_end;

	int x = p.x;
	int y = p.y;
	int z = p.z;

	// check in 
	int data_halo = sdr->data_halo;

	R rt;
	if(true) {
		int X = buffer_end.x - buffer_start.x;
		int Y = buffer_end.y - buffer_start.y;

		// Data coordinate input
		// border check
		int flag1, flag2, flag3; // -1, 0, 1
		flag1 = check_in_border(x, full_data_start.x, full_data_end.x);
		flag2 = check_in_border(y, full_data_start.y, full_data_end.y);
		flag3 = check_in_border(z, full_data_start.z, full_data_end.z);

		bool flag; // border 
		flag = (1 <= border && border <= 4);

		if( (!flag && (flag1 == 0 && flag2 == 0 && flag3 == 0)) || (flag)){
			bool flag_x = range_check(x, data_start.x, data_end.x);
			bool flag_y = range_check(y, data_start.y, data_end.y);
			bool flag_z = range_check(z, data_start.z, data_end.z);

			if(flag_x && flag_y && flag_z){		
	
				rt = 100;
			}
			
			}else{
				rt = 0;
			}
	}else{
		rt = 0;
	}
	
	return rt;
}



template<typename R,typename T> GPU R checker(T* image, float3 p, int offset, VIVALDI_DATA_RANGE* sdr){
	return checker<R>(image, p, offset, 0, sdr);
}
template<typename R,typename T> GPU R checker(T* image, float x, float y, float z, int offset, int border, VIVALDI_DATA_RANGE* sdr){
	return checker<R>(image, make_float3(x,y,z), offset, border, sdr);
}
template<typename R,typename T> GPU R checker(T* image, float x, float y, float z,int offset, VIVALDI_DATA_RANGE* sdr){
	return checker<R>(image, make_float3(x,y,z), offset, 0, sdr);
}




template<typename R, typename T> GPU R s_point_query_3d(T* image, float3 p, int offset, int border, VIVALDI_DATA_RANGE* sdr){
	int4 data_start = sdr->data_start;
	int4 data_end = sdr->data_end;

	int4 full_data_start = sdr->full_data_start;
	int4 full_data_end = sdr->full_data_end;
	
	int4 buffer_start = sdr->buffer_start;
	int4 buffer_end = sdr->buffer_end;

	int x = p.x;
	int y = p.y;
	int z = p.z;

	// check in 
	int data_halo = sdr->data_halo;

	R rt;
	if(true) {
		int X = buffer_end.x - buffer_start.x;
		int Y = buffer_end.y - buffer_start.y;

		// Data coordinate input
		// border check
		int flag1, flag2, flag3; // -1, 0, 1
		flag1 = check_in_border(x, full_data_start.x, full_data_end.x);
		flag2 = check_in_border(y, full_data_start.y, full_data_end.y);
		flag3 = check_in_border(z, full_data_start.z, full_data_end.z);

		bool flag; // border 
		flag = (1 <= border && border <= 4);

		if( (!flag && (flag1 == 0 && flag2 == 0 && flag3 == 0)) || (flag)){
			// Border calculation
			x = border_handling(x, full_data_start.x, full_data_end.x, border);
			y = border_handling(y, full_data_start.y, full_data_end.y, border);
			z = border_handling(z, full_data_start.z, full_data_end.z, border);

			bool flag_x = range_check(x, data_start.x, data_end.x);
			bool flag_y = range_check(y, data_start.y, data_end.y);
			bool flag_z = range_check(z, data_start.z, data_end.z);

			if(flag_x && flag_y && flag_z){		
				// to Buffer coordinate 
				x = x - buffer_start.x;
				y = y - buffer_start.y;
				z = z - buffer_start.z;

				int target_z = z + offset;
				if( target_z >= middle - data_halo && target_z <= middle + data_halo + stride) {
					target_z = target_z % (data_halo * 2 + stride);
					if(target_z < 0)
						target_z = (data_halo * 2 + stride) + target_z;

					rt = convert(image[target_z*Y*X + y*X + x]);
				}else{
					rt = initial(image[0]);
				}

				//int target_z = (middle + offset) % (data_halo * 2 + 1);
				//if(target_z < 0)
					//target_z = (data_halo * 2 + 1) + target_z;

				//rt = convert(image[z*Y*X + y*X + x]);
			}else{
				rt = initial(image[0]);
			}
		}else{
			rt = initial(image[0]);
		}
	}else{
		rt = initial(image[0]);
	}
	
	return rt;
}


template<typename R,typename T> GPU R s_point_query_3d(T* image, float3 p, int offset, VIVALDI_DATA_RANGE* sdr){
	return s_point_query_3d<R>(image, p, offset, 0, sdr);
}
template<typename R,typename T> GPU R s_point_query_3d(T* image, float x, float y, float z, int offset, int border, VIVALDI_DATA_RANGE* sdr){
	return s_point_query_3d<R>(image, make_float3(x,y,z), offset, border, sdr);
}
template<typename R,typename T> GPU R s_point_query_3d(T* image, float x, float y, float z,int offset, VIVALDI_DATA_RANGE* sdr){
	return s_point_query_3d<R>(image, make_float3(x,y,z), offset, 0, sdr);
}


template<typename R, typename T> GPU R point_query_3d(T* image, float3 p, int border, VIVALDI_DATA_RANGE* sdr){
	int4 data_start = sdr->data_start;
	int4 data_end = sdr->data_end;

	int4 full_data_start = sdr->full_data_start;
	int4 full_data_end = sdr->full_data_end;
	
	int4 buffer_start = sdr->buffer_start;
	int4 buffer_end = sdr->buffer_end;

	int x = p.x;
	int y = p.y;
	int z = p.z;

	// check in 

	R rt;
		int X = buffer_end.x - buffer_start.x;
		int Y = buffer_end.y - buffer_start.y;

		// Data coordinate input
		// border check
		int flag1, flag2, flag3; // -1, 0, 1
		flag1 = check_in_border(x, full_data_start.x, full_data_end.x);
		flag2 = check_in_border(y, full_data_start.y, full_data_end.y);
		flag3 = check_in_border(z, full_data_start.z, full_data_end.z);

		bool flag; // border 
		flag = (1 <= border && border <= 4);

		if( (!flag && (flag1 == 0 && flag2 == 0 && flag3 == 0)) || (flag)){
			// Border calculation
			x = border_handling(x, full_data_start.x, full_data_end.x, border);
			y = border_handling(y, full_data_start.y, full_data_end.y, border);
			z = border_handling(z, full_data_start.z, full_data_end.z, border);

			bool flag_x = range_check(x, data_start.x, data_end.x);
			bool flag_y = range_check(y, data_start.y, data_end.y);
			bool flag_z = range_check(z, data_start.z, data_end.z);

			if(flag_x && flag_y && flag_z){		
				// to Buffer coordinate 
				x = x - buffer_start.x;
				y = y - buffer_start.y;
				z = z - buffer_start.z;

				rt = convert(image[z*Y*X + y*X + x]);
			}else{
				rt = initial(image[0]);
			}
		}else{
			rt = initial(image[0]);
		}
	
	return rt;
}
template<typename R, typename T> GPU R point_query_3d(T* image, float3 p, VIVALDI_DATA_RANGE* sdr){
	return point_query_3d<R>(image, p, 0, sdr);
}
template<typename R, typename T> GPU R point_query_3d(T* image, float x, float y, float z, int border, VIVALDI_DATA_RANGE* sdr){
	return point_query_3d<R>(image, make_float3(x,y,z), border, sdr);
}
template<typename R, typename T> GPU R point_query_3d(T* image, float x, float y, float z, VIVALDI_DATA_RANGE* sdr){
    return point_query_3d<R>(image, make_float3(x,y,z), 0, sdr);
}
template<typename R, typename T> GPU R point_query_3d(T* image, float3 p, float3 nearest, VIVALDI_DATA_RANGE* sdr){
    return point_query_3d<R>(image, p, nearest, 0, sdr);
}
// 3D data query functions
////////////////////////////////////////////////////////////////////////////////
template<typename R, typename T> GPU R point_query_3d(T* image, float3 p, float3 nearest, int border, VIVALDI_DATA_RANGE* sdr){
	int4 data_start = sdr->data_start;
	int4 data_end = sdr->data_end;

	int4 full_data_start = sdr->full_data_start;
	int4 full_data_end = sdr->full_data_end;
	
	int4 buffer_start = sdr->buffer_start;
	int4 buffer_end = sdr->buffer_end;

	int x = p.x;
	int y = p.y;
	int z = p.z;

	// check in 
	bool sample_flag = false;
	bool plane_flag = false;
	int glob_cnt = 0;
	for(int i=0; i < plane_cnt; i++) {
		int cur_plane_cnt = plane_index[i];
		bool local_sample_flag = true;
		bool local_plane_flag = false;
		for(int j=0; j < cur_plane_cnt; j++) {
		//for(int j=0; j < ; j++) {
			float4 loc_plane = planes[glob_cnt+j];
			float loc_dist = dot(loc_plane, make_float4(x,y,z,1));
			float plane_dist = length(p-nearest);
			if(loc_dist > 0.0001) {
				local_sample_flag = true && local_sample_flag;
				local_plane_flag = false | local_plane_flag;
			//} else if (abs(loc_dist) < 0.0001 && plane_dist < 0.0001 && !sample_flag && local_sample_flag) {
			} else if (plane_dist < 0.0001 && !sample_flag) {
			//} else if (abs(loc_dist) < 0.0001 && (!sample_flag||local_sample_flag)) {
				local_plane_flag = true;
				local_sample_flag = true;
			} else {
				local_sample_flag = false;
				local_plane_flag = false;
			}
		}
		glob_cnt += cur_plane_cnt;
		sample_flag = local_sample_flag | sample_flag ;
		plane_flag = local_plane_flag | plane_flag;
	}

	if(plane_cnt == 0) 
		sample_flag = true;


	R rt;
	if(sample_flag) {
		int X = buffer_end.x - buffer_start.x;
		int Y = buffer_end.y - buffer_start.y;

		// Data coordinate input
		// border check
		int flag1, flag2, flag3; // -1, 0, 1
		flag1 = check_in_border(x, full_data_start.x, full_data_end.x);
		flag2 = check_in_border(y, full_data_start.y, full_data_end.y);
		flag3 = check_in_border(z, full_data_start.z, full_data_end.z);

		bool flag; // border 
		flag = (1 <= border && border <= 4);

		if( (!flag && (flag1 == 0 && flag2 == 0 && flag3 == 0)) || (flag)){
			// Border calculation
			x = border_handling(x, full_data_start.x, full_data_end.x, border);
			y = border_handling(y, full_data_start.y, full_data_end.y, border);
			z = border_handling(z, full_data_start.z, full_data_end.z, border);

			bool flag_x = range_check(x, data_start.x, data_end.x);
			bool flag_y = range_check(y, data_start.y, data_end.y);
			bool flag_z = range_check(z, data_start.z, data_end.z);

			if(flag_x && flag_y && flag_z){		
				// to Buffer coordinate 
				x = x - buffer_start.x;
				y = y - buffer_start.y;
				z = z - buffer_start.z;

				rt = convert(image[z*Y*X + y*X + x]);
			}else{
				rt = initial(image[0]);
			}
		}else{
			rt = initial(image[0]);
		}
	}else{
		rt = initial(image[0]);
	}
	if(plane_flag)
		return -1 * rt;
	
	return rt;
}






GPU float d_lerp(float a, float b, float t){
    return (b - a) * t;
}

template<typename R, typename T> GPU R linear_query_3d(T* volume, float3 p, int border, VIVALDI_DATA_RANGE* sdr){
    //tri linear interpolation
    float x,y,z;
    x = p.x;
    y = p.y;
    z = p.z;
	
    int fx = floor(x);
    int fy = floor(y);
    int fz = floor(z);
    int cx = ceil(x);
    int cy = ceil(y);
    int cz = ceil(z);
	
	R q000 = point_query_3d<R>(volume, fx, fy, fz, border, sdr);
	R q001 = point_query_3d<R>(volume, fx, fy, cz, border, sdr);
	R q010 = point_query_3d<R>(volume, fx, cy, fz, border, sdr);
	R q011 = point_query_3d<R>(volume, fx, cy, cz, border, sdr);
	R q100 = point_query_3d<R>(volume, cx, fy, fz, border, sdr);
	R q101 = point_query_3d<R>(volume, cx, fy, cz, border, sdr);
	R q110 = point_query_3d<R>(volume, cx, cy, fz, border, sdr);
	R q111 = point_query_3d<R>(volume, cx, cy, cz, border, sdr);

    float dx = x - fx;
    float dy = y - fy;
    float dz = z - fz;

    // lerp along x
    R q00 = lerp(q000, q001, dx);
    R q01 = lerp(q010, q011, dx);
    R q10 = lerp(q100, q101, dx);
    R q11 = lerp(q110, q111, dx);

    // lerp along y
    R q0 = lerp(q00, q01, dy);
    R q1 = lerp(q10, q11, dy);

    // lerp along z
    R q = lerp(q0, q1, dz);
    return q;
}
template<typename R, typename T> GPU R linear_query_3d(T* volume, float3 p, VIVALDI_DATA_RANGE* sdr){
	return linear_query_3d<R>(volume, p, 0, sdr);
}
template<typename R, typename T> GPU R linear_query_3d(T* volume, float x, float y, float z, int border, VIVALDI_DATA_RANGE* sdr){
	return linear_query_3d<R>(volume, make_float3(x,y,z), border, sdr);
}
template<typename R, typename T> GPU R linear_query_3d(T* volume, float x, float y, float z, VIVALDI_DATA_RANGE* sdr){
	return linear_query_3d<R>(volume, make_float3(x,y,z), 0, sdr);
}

template<typename R, typename T> GPU float3 linear_gradient_3d(T* volume, float3 p, VIVALDI_DATA_RANGE* sdr){
    float3 rbf = make_float3(0);
	float delta = 1.0f;
	//float delta = 0.01f;
	R dx = linear_query_3d<R>(volume, make_float3(p.x + delta, p.y, p.z), sdr) 
	     - linear_query_3d<R>(volume, make_float3(p.x - delta, p.y, p.z), sdr);
	R dy = linear_query_3d<R>(volume, make_float3(p.x, p.y + delta, p.z), sdr)
         - linear_query_3d<R>(volume, make_float3(p.x, p.y - delta, p.z), sdr);
	R dz = linear_query_3d<R>(volume, make_float3(p.x, p.y, p.z + delta), sdr)
	     - linear_query_3d<R>(volume, make_float3(p.x, p.y, p.z - delta), sdr);

//	float dxl = length(dx);
//	float dyl = length(dy);
//	float dzl = length(dz);
//	return make_float3(dxl, dyl, dzl) / (2 * delta);
//	return make_float3(dxl) / (2 * delta);
//	return make_float3(dx);
	return make_float3(dx, dy, dz) / (2 * delta);
}
template<typename R, typename T> GPU float3 linear_gradient_3d(T* volume, int x, int y, int z, VIVALDI_DATA_RANGE* sdr){
	return linear_gradient_3d<R>(volume, make_float3(x,y,z), sdr);
}
template<typename R, typename T> GPU float3 linear_gradient_3d(T* volume, float x, float y, float z, VIVALDI_DATA_RANGE* sdr){
	return linear_gradient_3d<R>(volume, make_float3(x,y,z), sdr);
}

template<typename R,typename T> GPU T cubic_query_3d(T* volume, float3 p, VIVALDI_DATA_RANGE* sdr){
	float3 alpha = 255 * (p - floor(p - 0.5f) - 0.5f);

	float3 hgx = hg(alpha.x);
	float3 hgy = hg(alpha.y);
	float3 hgz = hg(alpha.z);

	// 8 linear queries
	R q000 = linear_query_3d<R>(volume, p.x - hgx.x, p.y - hgy.x, p.z - hgz.x, sdr);
	R q001 = linear_query_3d<R>(volume, p.x - hgx.x, p.y - hgy.x, p.z + hgz.y, sdr);
	R q010 = linear_query_3d<R>(volume, p.x - hgx.x, p.y + hgy.y, p.z - hgz.x, sdr);
	R q011 = linear_query_3d<R>(volume, p.x - hgx.x, p.y + hgy.y, p.z + hgz.y, sdr);
	R q100 = linear_query_3d<R>(volume, p.x + hgx.y, p.y - hgy.x, p.z - hgz.x, sdr);
	R q101 = linear_query_3d<R>(volume, p.x + hgx.y, p.y - hgy.x, p.z + hgz.y, sdr);
	R q110 = linear_query_3d<R>(volume, p.x + hgx.y, p.y + hgy.y, p.z - hgz.x, sdr);
	R q111 = linear_query_3d<R>(volume, p.x + hgx.y, p.y + hgy.y, p.z + hgz.y, sdr);

	// lerp along z
	R q00 = lerp(q000, q001, hgz.z);
	R q01 = lerp(q010, q011, hgz.z);
	R q10 = lerp(q100, q101, hgz.z);
	R q11 = lerp(q110, q111, hgz.z);

	// lerp along y
	R q0 = lerp(q00, q01, hgy.z);
	R q1 = lerp(q10, q11, hgy.z);

	// lerp along x
	R q = lerp(q0, q1, hgx.z);
	return q;
}

template<typename R,typename T> GPU float3 cubic_gradient_3d(T* data, float3 p, VIVALDI_DATA_RANGE* sdr){

    float3 rbf = make_float3(0);
    
	float3 alpha = 255 * (p - floor(p - 0.5f) - 0.5f);

	float3 hgx = hg(alpha.x);
	float3 hgy = hg(alpha.y);
	float3 hgz = hg(alpha.z);

	float3 dhgx = dhg(alpha.x);
	float3 dhgy = dhg(alpha.y);
	float3 dhgz = dhg(alpha.z);

	// compute x-derivative
	
	R q000 = linear_query_3d<R>(data, p.x - dhgx.x, p.y - hgy.x, p.z - hgz.x, sdr);
	R q001 = linear_query_3d<R>(data, p.x - dhgx.x, p.y - hgy.x, p.z + hgz.y, sdr);
	R q010 = linear_query_3d<R>(data, p.x - dhgx.x, p.y + hgy.y, p.z - hgz.x, sdr);
	R q011 = linear_query_3d<R>(data, p.x - dhgx.x, p.y + hgy.y, p.z + hgz.y, sdr);
	R q100 = linear_query_3d<R>(data, p.x + dhgx.y, p.y - hgy.x, p.z - hgz.x, sdr);
	R q101 = linear_query_3d<R>(data, p.x + dhgx.y, p.y - hgy.x, p.z + hgz.y, sdr);
	R q110 = linear_query_3d<R>(data, p.x + dhgx.y, p.y + hgy.y, p.z - hgz.x, sdr);
	R q111 = linear_query_3d<R>(data, p.x + dhgx.y, p.y + hgy.y, p.z + hgz.y, sdr);

	R q00 = lerp(q000, q001, hgz.z);
	R q01 = lerp(q010, q011, hgz.z);
	R q10 = lerp(q100, q101, hgz.z);
	R q11 = lerp(q110, q111, hgz.z);

	R q0 = lerp(q00, q01, hgy.z);
	R q1 = lerp(q10, q11, hgy.z);

	float gradientX = d_lerp(q0, q1, dhgx.z);

	// compute y-derivative
	q000 = linear_query_3d<R>(data, p.x - hgx.x, p.y - dhgy.x, p.z - hgz.x, sdr);
	q001 = linear_query_3d<R>(data, p.x - hgx.x, p.y - dhgy.x, p.z + hgz.y, sdr);
	q010 = linear_query_3d<R>(data, p.x - hgx.x, p.y + dhgy.y, p.z - hgz.x, sdr);
	q011 = linear_query_3d<R>(data, p.x - hgx.x, p.y + dhgy.y, p.z + hgz.y, sdr);
	q100 = linear_query_3d<R>(data, p.x + hgx.y, p.y - dhgy.x, p.z - hgz.x, sdr);
	q101 = linear_query_3d<R>(data, p.x + hgx.y, p.y - dhgy.x, p.z + hgz.y, sdr);
	q110 = linear_query_3d<R>(data, p.x + hgx.y, p.y + dhgy.y, p.z - hgz.x, sdr);
	q111 = linear_query_3d<R>(data, p.x + hgx.y, p.y + dhgy.y, p.z + hgz.y, sdr);

	q00 = lerp(q000, q001, hgz.z);
	q01 = lerp(q010, q011, hgz.z);
	q10 = lerp(q100, q101, hgz.z);
	q11 = lerp(q110, q111, hgz.z);

	q0 = d_lerp(q00, q01, dhgy.z);
	q1 = d_lerp(q10, q11, dhgy.z);

	float gradientY = lerp(q0, q1, hgx.z);

	// compute z-derivative
	q000 = linear_query_3d<R>(data, p.x - hgx.x, p.y - hgy.x, p.z - dhgz.x, sdr);
	q001 = linear_query_3d<R>(data, p.x - hgx.x, p.y - hgy.x, p.z + dhgz.y, sdr);
	q010 = linear_query_3d<R>(data, p.x - hgx.x, p.y + hgy.y, p.z - dhgz.x, sdr);
	q011 = linear_query_3d<R>(data, p.x - hgx.x, p.y + hgy.y, p.z + dhgz.y, sdr);
	q100 = linear_query_3d<R>(data, p.x + hgx.y, p.y - hgy.x, p.z - dhgz.x, sdr);
	q101 = linear_query_3d<R>(data, p.x + hgx.y, p.y - hgy.x, p.z + dhgz.y, sdr);
	q110 = linear_query_3d<R>(data, p.x + hgx.y, p.y + hgy.y, p.z - dhgz.x, sdr);
	q111 = linear_query_3d<R>(data, p.x + hgx.y, p.y + hgy.y, p.z + dhgz.y, sdr);

	q00 = d_lerp(q000, q001, dhgz.z);
	q01 = d_lerp(q010, q011, dhgz.z);
	q10 = d_lerp(q100, q101, dhgz.z);
	q11 = d_lerp(q110, q111, dhgz.z);

	q0 = lerp(q00, q01, hgy.z);
	q1 = lerp(q10, q11, hgy.z);

	float gradientZ = lerp(q0, q1, hgx.z);

	return make_float3(gradientX, gradientY, gradientZ);
}

//rotate functions
///////////////////////////////////////////////////////////////////////////////////
GPU float arccos(float angle){
	return acos(angle);
}
GPU float arcsin(float angle){
	return asin(angle);
}
GPU float norm(float3 a){
	float val = 0;
	val += a.x*a.x + a.y*a.y + a.z*a.z;
	val = sqrt(val);
	return val;
}

GPU float3 matmul(float3* mat, float3 vec){
    float x = mat[0].x*vec.x + mat[1].x*vec.y + mat[2].x*vec.z;
    float y = mat[0].y*vec.x + mat[1].y*vec.y + mat[2].y*vec.z;
    float z = mat[0].z*vec.x + mat[1].z*vec.y + mat[2].z*vec.z;

    return make_float3(x, y, z);
}
GPU void getInvMat(float3* mat, float3* ret) {
    double det = mat[0].x*(mat[1].y*mat[2].z-mat[1].z*mat[2].y)-mat[0].y*(mat[1].x*mat[2].z-mat[1].z*mat[2].x)+mat[0].z*(mat[1].x*mat[2].y-mat[1].y*mat[2].x);

    if(det!=0) {
        double invdet = 1/det;
        float a00 = (mat[1].y*mat[2].z-mat[2].y*mat[1].z)*invdet;
        float a01 = (mat[0].z*mat[2].y-mat[0].y*mat[2].z)*invdet;
        float a02 = (mat[0].y*mat[1].z-mat[0].z*mat[1].y)*invdet;
        float a10 = (mat[1].z*mat[2].x-mat[1].x*mat[2].z)*invdet;
        float a11 = (mat[0].x*mat[2].z-mat[0].z*mat[2].x)*invdet;
        float a12 = (mat[1].x*mat[0].z-mat[0].x*mat[1].z)*invdet;
        float a20 = (mat[1].x*mat[2].y-mat[2].x*mat[1].y)*invdet;
        float a21 = (mat[2].x*mat[0].y-mat[0].x*mat[2].y)*invdet;
        float a22 = (mat[0].x*mat[1].y-mat[1].x*mat[0].y)*invdet;
        ret[0] = make_float3(a00, a01, a02);
        ret[1] = make_float3(a10, a11, a12);
        ret[2] = make_float3(a20, a21, a22);
    }
    else {
        ret[0] = make_float3(0);
        ret[1] = make_float3(0);
        ret[2] = make_float3(0);
    }
}
GPU float getDistance(float3* mat, float3 vec){
    float3 tmp_mat  = matmul(mat, vec);
    if(tmp_mat.z>200000 ) return -8765;
    if(tmp_mat.z<0) tmp_mat.z = 0;
    if(tmp_mat.y < 0 || tmp_mat.y > 1) return -8765;
    if(tmp_mat.x < 0 || tmp_mat.x > 1) return -8765;

    if(tmp_mat.y+tmp_mat.x > 1.0000) return -8765;

    return tmp_mat.z;
}

GPU float2 getCrossedInterval(float3 origin, float3 direction, float3* tmp){
	float3 min = tmp[0];
	float3 max = tmp[1];

	float tmin=-9999.0, tmax=9999.0, tymin=-9999.0, tymax=9999.0, tzmin=-9999.0, tzmax=9999.0;
	if (direction.x > 0) {
		tmin = (min.x - origin.x) / direction.x;
		tmax = (max.x - origin.x) / direction.x;
	}
	else if(direction.x < 0) { 
		tmin = (max.x - origin.x) / direction.x;
		tmax = (min.x - origin.x) / direction.x;
	}
	if (direction.y > 0) {
		tymin = (min.y - origin.y) / direction.y;
		tymax = (max.y - origin.y) / direction.y;
	}
	else if(direction.y < 0) {
		tymin = (max.y - origin.y) / direction.y;
		tymax = (min.y - origin.y) / direction.y;
	}
	if (direction.z > 0) {
		tzmin = (min.z - origin.z) / direction.z;
		tzmax = (max.z - origin.z) / direction.z;
	}
	else if(direction.z < 0) {
		tzmin = (max.z - origin.z) / direction.z;
		tzmax = (min.z - origin.z) / direction.z;
	}
	float start, end;
	start = (tmin < tymin)?((tymin < tzmin)?tzmin:tymin):((tmin < tzmin)?tzmin:tmin);
	end = (tmax > tymax)?((tymax > tzmax)?tzmax:tymax):((tmax > tzmax)?tzmax:tmax);
	if((origin.x > min.x) && (origin.x < max.x) && (origin.y > min.y) && (origin.y < max.y) && (origin.z > min.z) && (origin.z < max.z)) {
		end = start;
		start = 0;
	}
		
		
	return make_float2(start, end);
}	

GPU float2 intersectSlab(float p, float d, float2 slab){
    if (fabs(d) < 0.0001f) return make_float2(-INF, INF);

    float x1 = (slab.x - p) / d;
    float x2 = (slab.y - p) / d;

    if (x1 <= x2) return make_float2(x1, x2);
        else return make_float2(x2, x1);
}

GPU float2 intersectIntervals(float2 a, float2 b){
    if (a.x > b.x)
    {
        float2 temp = a; a = b; b = temp;
    }
    
    if (b.x > a.y) return make_float2(INF, -INF);
    return make_float2(b.x, min(a.y, b.y));
}

GPU float2 intersectUnitCube(float3 p, float3 d, float3 *tmp){
    //float2 slab = make_float2(-1, 1);
	
	float3 min = tmp[0];
	float3 max = tmp[1];
	
	float2 slabx = make_float2(min.x, max.x);
	float2 slaby = make_float2(min.y, max.y);
	float2 slabz = make_float2(min.z, max.z);
	
    float2 tx = intersectSlab(p.x, d.x, slabx);
    float2 ty = intersectSlab(p.y, d.y, slaby);
    float2 tz = intersectSlab(p.z, d.z, slabz);
    
	// parallel test
	
	if(tx.x == -INF){
		if( p.x < min.x || max.x <= p.x)
			return make_float2(INF, -INF);
	}
	if(ty.x == -INF){
		if( p.y < min.y || max.y <= p.y)
			return make_float2(INF, -INF);
	}
	if(tz.x == -INF){
		if( p.z < min.z || max.z <= p.z)
			return make_float2(INF, -INF);
	}
	
	return intersectIntervals(tx, intersectIntervals(ty, tz));
	//return make_float2(slaby.y, 0);
}

template<typename T> GPU line_iter perspective_iter(T* volume, float x, float y, float step, float near, VIVALDI_DATA_RANGE* sdr){
	
    int4 start = sdr->data_start;
    int4 end = sdr->data_end;
	
	float data_halo = sdr->data_halo;
	
	float3 ray_direction = make_float3(x,y,near);
	float3 ray_origin = make_float3(0);

	start = start +  make_int4(data_halo);
	end = end - make_int4(data_halo);

	float3 min_max[2];
	min_max[0] = make_float3(start.x, start.y, start.z);
	min_max[1] = make_float3(end.x, end.y, end.z);

	float o_x, o_y, o_z;
    o_x = inv_modelview[0][0] * ray_origin.x + inv_modelview[0][1] * ray_origin.y + inv_modelview[0][2] * ray_origin.z + inv_modelview[0][3];
    o_y = inv_modelview[1][0] * ray_origin.x + inv_modelview[1][1] * ray_origin.y + inv_modelview[1][2] * ray_origin.z + inv_modelview[1][3];
    o_z = inv_modelview[2][0] * ray_origin.x + inv_modelview[2][1] * ray_origin.y + inv_modelview[2][2] * ray_origin.z + inv_modelview[2][3];

	ray_origin = make_float3(o_x, o_y, o_z);

    o_x = inv_modelview[0][0] * ray_direction.x + inv_modelview[0][1] * ray_direction.y + inv_modelview[0][2] * ray_direction.z;// + inv_modelview[0][3];
    o_y = inv_modelview[1][0] * ray_direction.x + inv_modelview[1][1] * ray_direction.y + inv_modelview[1][2] * ray_direction.z;// + inv_modelview[1][3];
    o_z = inv_modelview[2][0] * ray_direction.x + inv_modelview[2][1] * ray_direction.y + inv_modelview[2][2] * ray_direction.z;// + inv_modelview[2][3];
	
	ray_direction = normalize(make_float3(o_x, o_y, o_z));
	
    float2 interval = intersectUnitCube(ray_origin, ray_direction, min_max);
	
//	float val;
//	val = interval.x;
//	return line_iter(make_float3(val), make_float3(val,val,val+1), 1.0);
	
	if(interval.x == INF) return line_iter(make_float3(0), make_float3(0), 1.0);
    float3 S = ray_origin + interval.x * ray_direction;
    float3 E = ray_origin + interval.y * ray_direction;

	return line_iter(S,E,step);
}
template<typename T> GPU float3 cut_plane_loc(T* volume, float2 p, float step, float3 norm, float3 point, VIVALDI_DATA_RANGE* vdr) { 
 
    float3 ray_origin = make_float3(0), ray_direction = make_float3(0, 0, 1); 
    ray_origin.x = inv_modelview[0][0] * p.x + inv_modelview[0][1] * p.y + inv_modelview[0][3]; 
    ray_origin.y = inv_modelview[1][0] * p.x + inv_modelview[1][1] * p.y + inv_modelview[1][3]; 
    ray_origin.z = inv_modelview[2][0] * p.x + inv_modelview[2][1] * p.y + inv_modelview[2][3]; 
 
    float tx, ty, tz; 
     
    tx = inv_modelview[0][0] * ray_direction.x + inv_modelview[0][1] * ray_direction.y + inv_modelview[0][2] * ray_direction.z; 
    ty = inv_modelview[1][0] * ray_direction.x + inv_modelview[1][1] * ray_direction.y + inv_modelview[1][2] * ray_direction.z; 
    tz = inv_modelview[2][0] * ray_direction.x + inv_modelview[2][1] * ray_direction.y + inv_modelview[2][2] * ray_direction.z; 
     
    ray_direction = normalize(make_float3(tx, ty, tz)); 
 
    float upper = dot((point - ray_origin), norm), lower = dot(ray_direction, norm); 
    float dist; 
    if(fabs(lower) > 0.0001f) 
        dist = upper / lower; 
    else dist = 0; 
    float3 sample_point = ray_origin + ((int)(dist/step)*step)*ray_direction; 
 
    return sample_point; 
} 
 
template<typename T> GPU float3 cut_plane_loc(T* volume, float x, float y, float step, float3 norm, float3 point, VIVALDI_DATA_RANGE* vdr) { 
    return cut_plane_loc(volume, make_float2(x, y), step, norm, point, vdr); 
} 

template<typename T> GPU float4 cut_plane_loc(T* volume, float2 p, float step, float4 plane, VIVALDI_DATA_RANGE* vdr) { 
 
	// initialization
    int4 start = vdr->data_start;
    int4 end = vdr->data_end;
	
    float3 ray_origin = make_float3(0), ray_direction = make_float3(0, 0, 1); 
    ray_origin.x = inv_modelview[0][0] * p.x + inv_modelview[0][1] * p.y + inv_modelview[0][3]; 
    ray_origin.y = inv_modelview[1][0] * p.x + inv_modelview[1][1] * p.y + inv_modelview[1][3]; 
    ray_origin.z = inv_modelview[2][0] * p.x + inv_modelview[2][1] * p.y + inv_modelview[2][3]; 
 
    float tx, ty, tz; 
     
    tx = inv_modelview[0][0] * ray_direction.x + inv_modelview[0][1] * ray_direction.y + inv_modelview[0][2] * ray_direction.z; 
    ty = inv_modelview[1][0] * ray_direction.x + inv_modelview[1][1] * ray_direction.y + inv_modelview[1][2] * ray_direction.z; 
    tz = inv_modelview[2][0] * ray_direction.x + inv_modelview[2][1] * ray_direction.y + inv_modelview[2][2] * ray_direction.z; 
     
    ray_direction = normalize(make_float3(tx, ty, tz)); 
	float3 norm = normalize(make_float3(plane.x, plane.y, plane.z));
 
    float upper = - dot(ray_origin, norm) - plane.w, lower = dot(ray_direction, norm); 
    float dist; 
    if(fabs(lower) > 0.0001f) 
        dist = upper / lower; 
    else dist = 0; 
      
	float3 min_max[2];
	min_max[0] = make_float3(start.x, start.y, start.z);
	min_max[1] = make_float3(end.x, end.y, end.z);
	float2 interval = intersectUnitCube(ray_origin, ray_direction, min_max);

    
    float3 s_point = ray_origin + ((int)(dist/step)*step)*ray_direction; 
    //float3 sample_point = ray_origin + dist*ray_direction; 
	if(interval.x <dist && interval.y > dist)
    	return make_float4(s_point.x, s_point.y, s_point.z, dist);
	else 
    	return make_float4(0, 0, 0, 100000);
} 
 
template<typename T> GPU float4 cut_plane_loc(T* volume, float x, float y, float step, float4 plane, VIVALDI_DATA_RANGE* vdr) { 
    return cut_plane_loc(volume, make_float2(x, y), step, plane, vdr); 
} 

template<typename T> GPU float3 set_planes(T* volume, float2 p, float step, VIVALDI_DATA_RANGE* vdr) 
{
	int cnt=0;
	float4 nearest = make_float4(-100, -100, -100, 1000000);
	for(int i=0; i<plane_cnt; i++) {
		int cur_plane_cnt = plane_index[i];
		float4 cp_loc;
		if(cur_plane_cnt < 0) {
			//cur_plane_cnt = -cur_plane_cnt + 2;
			//for(int j=0; j < cur_plane_cnt; j++) {
				//float4 cur = planes[cnt];
				//float4 next = planes[cnt];


				//float3 cur_top = make_float3(
			//}
			

		} else {
			for(int j=0; j < cur_plane_cnt; j++) {
				bool flag = true;
				float4 loc_plane = planes[cnt+j];
				cp_loc = cut_plane_loc(volume, p, step, loc_plane, vdr);
				if(nearest.w > cp_loc.w) {
					for(int k=0; k<cur_plane_cnt; k++) {
						if(k==j) continue;
						float4 other_plane = planes[cnt+k];
						float dist=dot(other_plane, make_float4(cp_loc.x, cp_loc.y, cp_loc.z, 1));
						if(dist < 0)
							flag = false;
					}	
					if(flag) {
						nearest = cp_loc;
						//nearest.z = cnt+j;
					}
				}
			}
			cnt += cur_plane_cnt;
		}
	}
	return make_float3(nearest.x, nearest.y, nearest.z);
}
template<typename T> GPU float3 set_planes(T* volume, float x, float y, float step, VIVALDI_DATA_RANGE* vdr) 
{
	return set_planes(volume, make_float2(x,y), step, vdr);
}

// Orthogonal_iter with pre-computing
template<typename T> GPU line_iter orthogonal_iter(T* volume, float2 p, float step, float3 ray_direction, VIVALDI_DATA_RANGE* sdr){

	// initialization
    int4 start = sdr->data_start;
    int4 end = sdr->data_end;
	
	int data_halo = sdr->data_halo;
	
    //float3 ray_direction = make_float3(0,0,-1);
    float3 ray_origin = make_float3(p.x, p.y ,1696*3);

	//start = start +  make_int4(data_halo);
	//end = end - make_int4(data_halo);
                                                                                                                                                                                                                                             
	float3 min_max[2];
	min_max[0] = make_float3(start.x, start.y, start.z);
	min_max[1] = make_float3(end.x, end.y, end.z);

	float o_x, o_y, o_z;
    o_x = inv_modelview[0][0] * p.x + inv_modelview[0][1] * p.y + inv_modelview[0][2] * 0 + inv_modelview[0][3];
    o_y = inv_modelview[1][0] * p.x + inv_modelview[1][1] * p.y + inv_modelview[1][2] * 0 + inv_modelview[1][3];
    o_z = inv_modelview[2][0] * p.x + inv_modelview[2][1] * p.y + inv_modelview[2][2] * 0 + inv_modelview[2][3];

	ray_origin = make_float3(o_x, o_y, o_z);

    o_x = inv_modelview[0][0] * ray_direction.x + inv_modelview[0][1] * ray_direction.y + inv_modelview[0][2] * ray_direction.z;// + inv_modelview[0][3];
    o_y = inv_modelview[1][0] * ray_direction.x + inv_modelview[1][1] * ray_direction.y + inv_modelview[1][2] * ray_direction.z;// + inv_modelview[1][3];
    o_z = inv_modelview[2][0] * ray_direction.x + inv_modelview[2][1] * ray_direction.y + inv_modelview[2][2] * ray_direction.z;// + inv_modelview[2][3];
	
	//ray_direction = normalize(make_float3(o_x, o_y, o_z));

    float2 interval = intersectUnitCube(ray_origin, ray_direction, min_max);
	
//	float val;
//	val = end.z;
//	return line_iter(make_float3(val), make_float3(val,val,val+1), 1.0);
	
	if(interval.x == INF) return line_iter(make_float3(0), make_float3(0), 1.0);
    float3 S = ray_origin + ((int)(interval.x/step)*step)*ray_direction;
    float3 E = ray_origin + ((int)(interval.y/step)*step)*ray_direction;

	return line_iter(S,E,step);
}


template<typename T> GPU line_iter orthogonal_iter(T* volume, float x, float y, float step, float3 ray_direction, VIVALDI_DATA_RANGE* sdr){
	return orthogonal_iter(volume, make_float2(x,y), step, ray_direction, sdr);
}



// Orthogonal_iter with pre-computing
template<typename T> GPU line_iter orthogonal_iter(T* volume, float2 p, float step, VIVALDI_DATA_RANGE* sdr){

	// initialization
    int4 start = sdr->data_start;
    int4 end = sdr->data_end;
	
	int data_halo = sdr->data_halo;
	
    float3 ray_direction = make_float3(0,0,-1);
    float3 ray_origin = make_float3(p.x, p.y ,1696*3);

	//start = start +  make_int4(data_halo);
	//end = end - make_int4(data_halo);
     
    start = start -  make_int4(1);
	end = end + make_int4(1);
         
	float3 min_max[2];
	min_max[0] = make_float3(start.x, start.y, start.z);
	min_max[1] = make_float3(end.x, end.y, end.z);

	float o_x, o_y, o_z;
    o_x = inv_modelview[0][0] * p.x + inv_modelview[0][1] * p.y + inv_modelview[0][2] * 0 + inv_modelview[0][3];
    o_y = inv_modelview[1][0] * p.x + inv_modelview[1][1] * p.y + inv_modelview[1][2] * 0 + inv_modelview[1][3];
    o_z = inv_modelview[2][0] * p.x + inv_modelview[2][1] * p.y + inv_modelview[2][2] * 0 + inv_modelview[2][3];

	ray_origin = make_float3(o_x, o_y, o_z);

    o_x = inv_modelview[0][0] * ray_direction.x + inv_modelview[0][1] * ray_direction.y + inv_modelview[0][2] * ray_direction.z;// + inv_modelview[0][3];
    o_y = inv_modelview[1][0] * ray_direction.x + inv_modelview[1][1] * ray_direction.y + inv_modelview[1][2] * ray_direction.z;// + inv_modelview[1][3];
    o_z = inv_modelview[2][0] * ray_direction.x + inv_modelview[2][1] * ray_direction.y + inv_modelview[2][2] * ray_direction.z;// + inv_modelview[2][3];
	
	ray_direction = normalize(make_float3(o_x, o_y, o_z));

    float2 interval = intersectUnitCube(ray_origin, ray_direction, min_max);
	
//	float val;
//	val = end.z;
//	return line_iter(make_float3(val), make_float3(val,val,val+1), 1.0);
	
	if(interval.x == INF) return line_iter(make_float3(0), make_float3(0), 1.0);
    float3 S = ray_origin + ((int)(interval.x/step)*step)*ray_direction;
    float3 E = ray_origin + ((int)(interval.y/step)*step)*ray_direction;

	return line_iter(S,E,step);
}


template<typename T> GPU line_iter orthogonal_iter(T* volume, float x, float y, float step, VIVALDI_DATA_RANGE* sdr){
	return orthogonal_iter(volume, make_float2(x,y), step, sdr);
}

// Domain Specific functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
GPU float3 phong(float3 L, float3 pos, float3 N, float3 omega, float3 kd, float3 ks, float n, float3 amb){
	float3 color;
	float a, b, c;

    a = inv_modelview[0][0] * L.x + inv_modelview[0][1] * L.y + inv_modelview[0][2] * L.z + inv_modelview[0][3];
    b = inv_modelview[1][0] * L.x + inv_modelview[1][1] * L.y + inv_modelview[1][2] * L.z + inv_modelview[1][3];
    c = inv_modelview[2][0] * L.x + inv_modelview[2][1] * L.y + inv_modelview[2][2] * L.z + inv_modelview[2][3];
	L.x = a;
	L.y = b;
	L.z = c;
	L = normalize(L-pos);
	//ambient
	color = amb;

	// diffuse
    float lobe =  max(dot(N, L), 0.0f);
    
	color += kd * lobe;

    // specular
    if (n > 0)
    {
        float3 R = reflect(-L, N);
        lobe = pow(fmaxf(dot(R, omega), 0), n);
        color += ks * lobe;
    }
 
    // clamping is a hack, but looks better
    return fminf(color, make_float3(1));
}
GPU float3 phong(float3 L, float3 N, float3 omega, float3 kd, float3 ks, float n, float3 amb){
	float3 color;
	
	//ambient
	color = amb;

	// diffuse
    float lobe =  max(dot(N, L), 0.0f);
	//float lobe =  max(dot(-N, L), 0.0f);
    
	color += kd * lobe;

    // specular
    if (n > 0)
    {
        float3 R = reflect(-L, N);
        lobe = pow(fmaxf(dot(R, omega), 0), n);
        color += ks * lobe;
    }
 
    // clamping is a hack, but looks better
    return fminf(color, make_float3(1));
}
GPU float3 diffuse(float3 L, float3 N, float3 kd){
    float lobe = max(dot(N, L), 0.0f);
	return kd * lobe;
}
template<typename R,typename T> GPU R laplacian(T* image, float2 p, VIVALDI_DATA_RANGE* sdr){

	float x = p.x;
	float y = p.y;

	//parallel variables
	R a =  point_query_2d<R>(image, x, y, sdr);
	R u =  point_query_2d<R>(image, x, y+1, sdr);
	R d =  point_query_2d<R>(image, x, y-1, sdr);
	R l =  point_query_2d<R>(image, x-1, y, sdr);
	R r =  point_query_2d<R>(image, x+1, y, sdr);
	R ret = u+d+l+r-4.0*a;
	return ret;

}
template<typename R,typename T> GPU R laplacian(T* image, float x, float y, VIVALDI_DATA_RANGE* sdr){

	//parallel variables
	return laplacian<R>(image, make_float2(x,y), sdr);

}

extern "C"{
// memory copy functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// halo memset function
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
__global__ void halo_memeset( float3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
								int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
            {
                //parallel variables
                int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
                int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
                int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;

                int x = x_hschoi + x_start;
                int y = y_hschoi + y_start;
                int z = z_hschoi + z_start;

                if(x_end <= x || y_end <= y || z_end <= z)return;
                int idx =
                (z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)
                + (y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)
                + (x-rb_DATA_RANGE->start.x);


				int3 full_data_start = make_int3(rb_DATA_RANGE->full_data_start);
				int3 full_data_end = make_int3(rb_DATA_RANGE->full_data_end);

				int3 start = make_int3(rb_DATA_RANGE->start);
				int3 end = make_int3(rb_DATA_RANGE->end);

				x = x - start.x;
				y = y - start.y;
				z = z - start.z;
				int buffer_X = end.x - start.x;
				int buffer_Y = end.y - start.y;
				int buffer_Z = end.z - start.z;

				if (!(full_data_start.x <= x && x < full_data_end.x &&
					  full_data_start.y <= y && y < full_data_end.y &&
	  				  full_data_start.y <= y && y < full_data_end.z)){
					r[idx] = initial(rb[0]);
				}
            }

}
*/
}

__device__ float4 alpha_compositing_emph(float4 origin, float4 next)
{
	float a = origin.w/2.0;
	float r = origin.x;
	float g = origin.y;
	float b = origin.z;
	
	float x, y, z, w;
	w = a + (1-a/255) * next.w;
	
	w = (w > 255)? 255 : w;
	
	x = r*a/255 + (1-a*2/255) * next.x * next.w/255.0f;
	y = g*a/255 + (1-a*2/255) * next.y * next.w/255.0f;
	z = b*a/255 + (1-a*2/255) * next.z * next.w/255.0f;
	//x = r*a/255 + (1-a/255) * next.x * next.w/255.0f;
	//y = g*a/255 + (1-a/255) * next.y * next.w/255.0f;
	//z = b*a/255 + (1-a/255) * next.z * next.w/255.0f;

	return make_float4(x,y,z,w);
}

__device__ float4 alpha_compositing(float4 origin, float4 next, float aa) 
{ 
    float a = origin.w; 
    float r = aa*origin.x; 
    float g = aa*origin.y; 
    float b = aa*origin.z; 
    aa = (aa<0)?0:aa; 
     
    float x, y, z, w, w_p; 
    //w = a + (1-a/255) * next.w; 
    w = a *(1-next.w/255) + next.w; 
    w_p = aa*a*(1-next.w/255) + next.w; 
    //w = a + (1-a/255) * next.w; 
    //w_p = aa*a + (1-aa*a/255) * next.w; 
     
    w = (w > 255)? 255 : w; 
     
    x = r + (1-aa*a/255.0) * next.x * next.w/255.0; 
    y = g + (1-aa*a/255.0) * next.y * next.w/255.0; 
    z = b + (1-aa*a/255.0) * next.z * next.w/255.0; 
 
    return make_float4(x*w/w_p,y*w/w_p,z*w/w_p,w); 
}


__device__ float4 alpha_compositing(float4 origin, float4 next)
{
	float a = origin.w;
	float r = origin.x;
	float g = origin.y;
	float b = origin.z;
	
	float x, y, z, w;
	w = a + (1-a/255) * next.w;
	
	w = (w > 255)? 255 : w;
	
	x = r + (1-a/255) * next.x * next.w/255.0f;
	y = g + (1-a/255) * next.y * next.w/255.0f;
	z = b + (1-a/255) * next.z * next.w/255.0f;
	//x = r*a/255 + (1-a/255) * next.x * next.w/255.0f;
	//y = g*a/255 + (1-a/255) * next.y * next.w/255.0f;
	//z = b*a/255 + (1-a/255) * next.z * next.w/255.0f;

	return make_float4(x,y,z,w);
}

__device__ float4 blending(float4 origin, float4 next)
{
	float x = 0.0;
    float y = 0.0;
    float z = 0.0;
    float w = 0.0;

    w = max(origin.w,next.w);

    if (w > 0)
    {
        x = (origin.x*origin.w + next.x*next.w)/(origin.w+next.w);
        y = (origin.y*origin.w + next.y*next.w)/(origin.w+next.w);
        z = (origin.z*origin.w + next.z*next.w)/(origin.w+next.w);
    }

	return make_float4(x,y,z,w);
}

__device__ float4 alpha_blending(float4 origin, float4 next)
{
	float a = origin.w;
	float r = origin.x;
	float g = origin.y;
	float b = origin.z;
	
	float x, y, z, w;
	w = origin.w + (1-origin.w) * next.w;
	w = (w > 1.0)? 1.0: w;

/*
	x = (r + (1-a) * next.x);
	y = (g + (1-a) * next.y);
	z = (b + (1-a) * next.z);
*/

	x = (r*a + (1-a) * next.x * next.w)/w;
	y = (g*a + (1-a) * next.y * next.w)/w;
	z = (b*a + (1-a) * next.z * next.w)/w;

	//x = r*a/255 + (1-a/255) * next.x * next.w/255.0f;
	//y = g*a/255 + (1-a/255) * next.y * next.w/255.0f;
	//z = b*a/255 + (1-a/255) * next.z * next.w/255.0f;

	return make_float4(x,y,z,w);
}



__device__ float4 alpha_compositing_new(float4 origin, float4 next)
{
	float a = origin.w;
	float r = origin.x;
	float g = origin.y;
	float b = origin.z;
	
	float x, y, z, w;
	w = a + (1-a) * next.w;
	
	w = (w > 1.0)? 1.0: w;
	
	x = r + (1-a) * next.x * next.w;
	y = g + (1-a) * next.y * next.w;
	z = b + (1-a) * next.z * next.w;
	//x = r*a/255 + (1-a/255) * next.x * next.w/255.0f;
	//y = g*a/255 + (1-a/255) * next.y * next.w/255.0f;
	//z = b*a/255 + (1-a/255) * next.z * next.w/255.0f;

	return make_float4(x,y,z,w);
}

__device__ float4 alpha_compositing_wo_alpha_new(float4 origin, float4 next)
{
        float a = origin.w;
        float r = origin.x;
        float g = origin.y;
        float b = origin.z;
	//if(origin.w == 1) return make_float4(255,0,0,0);

        float x, y, z, w;
	w = a + (1-a) * next.w;

	w = (w > 1.0)? 1.0 : w;

    x = r + (1-a) * next.x;
    y = g + (1-a) * next.y;
    z = b + (1-a) * next.z;

	return make_float4(x,y,z,w);
}


__device__ float4 alpha_compositing_wo_alpha(float4 origin, float4 next)
{
        float a = origin.w;
        float r = origin.x;
        float g = origin.y;
        float b = origin.z;
	//if(origin.w == 1) return make_float4(255,0,0,0);

        float x, y, z, w;
	w = a + (1-a/255) * next.w;

	w = (w > 255)? 255 : w;

    x = r + (1-a/255) * next.x;
    y = g + (1-a/255) * next.y;
    z = b + (1-a/255) * next.z;

	return make_float4(x,y,z,w);
}
__device__ float4 background_white(float4 origin)
{
	float a = origin.w;
	float r = origin.x;
	float g = origin.y;
	float b = origin.z;

	float x, y, z, w;
	w = a ;


//	x = r + (1-a/255.0f) * 255.0f;
//	y = g + (1-a/255.0f) * 255.0f;
//	z = b + (1-a/255.0f) * 255.0f;


	x = r + (1.0-a);
	y = g + (1.0-a);
	z = b + (1.0-a);

	return make_float4(x,y,z,w);
}

__device__ float4 background_white(float4 origin, int x, int y)
{
	float4 buffer_color = make_float4(0, 0, 0, 0);
	//float dist=0;
	if(wr_start.x > -99999990) {
		//dist  =  tex2D(depthMap, x - wr_start.x, y- wr_start.y);
	 	buffer_color = tex2D(colorMap, x - wr_start.x, y - wr_start.y);
		if(buffer_color.x + buffer_color.y + buffer_color.z > 1)
			buffer_color.w = 255.0;
		else
			buffer_color.w = 0.0;
	}

	buffer_color = alpha_compositing(origin, buffer_color);

	return background_white(buffer_color);
}

__device__ float4 detach(float4 origin)
{
	float a = origin.w;
	float r = origin.x;
	float g = origin.y;
	float b = origin.z;

	float x,y,z,w;
	w = a;
	x = r - (1-a/255.0)*255.0;
	y = g - (1-a/255.0)*255.0;
	z = b - (1-a/255.0)*255.0;

	return make_float4(x,y,z,w);
}


texture<float4, 2> TFF;

#include <stdio.h>


__device__ float4 transfer(float a, double thre)
{
	if(a < 0 && abs(a) > thre)
		return make_float4(-a/TF_bandwidth*255, -a/TF_bandwidth*255, -a/TF_bandwidth*255, 255);
	float4 tmp = tex2D(TFF, a/TF_bandwidth* 255, 0);
	float4 tmp_col = make_float4(tmp.x*255.0, tmp.y*255.0, tmp.z*255.0, tmp.w*255);
	//float4 tmp_col = make_float4(tmp.x*255.0, tmp.y*255.0, tmp.z*255.0, 0);


	return tmp_col;
}
__device__ float4 transfer(float a) 
{
	return transfer(a, 0);
}
__device__ float4 transfer(float2 a)
{
	return transfer(a.x);
}
__device__ float4 transfer(float3 a)
{
	return transfer(a.x);
}
__device__ float4 transfer(float4 a)
{
	return transfer(a.x);
}
texture<float4, 2> TFF1;
texture<float4, 2> TFF2;
texture<float4, 2> TFF3;
texture<float4, 2> TFF4;
__device__ float4 transfer(float a, int chan)
{
	float4 tmp;
	if(chan == 0) 
		tmp = tex2D(TFF, a/TF_bandwidth * 255, 0);
	else if(chan == 1) 
		tmp = tex2D(TFF1, a/TF_bandwidth * 255, 0);
	else if(chan == 2) 
		tmp = tex2D(TFF2, a/TF_bandwidth * 255, 0);
	else if(chan == 3) 
		tmp = tex2D(TFF3, a/TF_bandwidth * 255, 0);
	else if(chan == 4) 
		tmp = tex2D(TFF4, a/TF_bandwidth * 255, 0);

	float4 tmp_col = make_float4(tmp.x*255.0, tmp.y*255.0, tmp.z*255.0, tmp.w*255);

	return tmp_col;
}
__device__ float4 transfer(float2 a, int chan)
{
	return transfer(a.x, chan);
}
__device__ float4 transfer(float3 a, int chan)
{
	return transfer(a.x, chan);
}
__device__ float4 transfer(float4 a, int chan)
{
	return transfer(a.x, chan);
}
__device__ float4 ch_binder(float4 a, float4 b)
{
	return (a + b) / 2.0f;
}
__device__ int floor_tmp(float a)
{
	return floor(a);
}
extern "C"{

	__global__ void writing_1d_char(char1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	char1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_1d<float>(A, x, A_DATA_RANGE);
		char1 a = make_char1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_char2(char2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	char2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_1d<float2>(A, x, A_DATA_RANGE);
		char2 a = make_char2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_char3(char3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	char3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_1d<float3>(A, x, A_DATA_RANGE);
		char3 a = make_char3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_char4(char4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	char4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_1d<float4>(A, x, A_DATA_RANGE);
		char4 a = make_char4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_uchar(uchar1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_1d<float>(A, x, A_DATA_RANGE);
		uchar1 a = make_uchar1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_uchar2(uchar2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_1d<float2>(A, x, A_DATA_RANGE);
		uchar2 a = make_uchar2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_uchar3(uchar3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_1d<float3>(A, x, A_DATA_RANGE);
		uchar3 a = make_uchar3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_uchar4(uchar4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_1d<float4>(A, x, A_DATA_RANGE);
		uchar4 a = make_uchar4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_short(short1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_1d<float>(A, x, A_DATA_RANGE);
		short1 a = make_short1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_short2(short2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_1d<float2>(A, x, A_DATA_RANGE);
		short2 a = make_short2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_short3(short3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_1d<float3>(A, x, A_DATA_RANGE);
		short3 a = make_short3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_short4(short4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_1d<float4>(A, x, A_DATA_RANGE);
		short4 a = make_short4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_ushort(ushort1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	ushort1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_1d<float>(A, x, A_DATA_RANGE);
		ushort1 a = make_ushort1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_ushort2(ushort2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	ushort2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_1d<float2>(A, x, A_DATA_RANGE);
		ushort2 a = make_ushort2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_ushort3(ushort3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	ushort3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_1d<float3>(A, x, A_DATA_RANGE);
		ushort3 a = make_ushort3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_ushort4(ushort4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	ushort4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_1d<float4>(A, x, A_DATA_RANGE);
		ushort4 a = make_ushort4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_int(int1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_1d<float>(A, x, A_DATA_RANGE);
		int1 a = make_int1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_int2(int2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_1d<float2>(A, x, A_DATA_RANGE);
		int2 a = make_int2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_int3(int3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_1d<float3>(A, x, A_DATA_RANGE);
		int3 a = make_int3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_int4(int4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_1d<float4>(A, x, A_DATA_RANGE);
		int4 a = make_int4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_uint(uint1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uint1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_1d<float>(A, x, A_DATA_RANGE);
		uint1 a = make_uint1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_uint2(uint2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uint2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_1d<float2>(A, x, A_DATA_RANGE);
		uint2 a = make_uint2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_uint3(uint3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uint3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_1d<float3>(A, x, A_DATA_RANGE);
		uint3 a = make_uint3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_uint4(uint4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uint4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_1d<float4>(A, x, A_DATA_RANGE);
		uint4 a = make_uint4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_float(float1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_1d<float>(A, x, A_DATA_RANGE);
		float1 a = make_float1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_float2(float2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_1d<float2>(A, x, A_DATA_RANGE);
		float2 a = make_float2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_float3(float3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_1d<float3>(A, x, A_DATA_RANGE);
		float3 a = make_float3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_float4(float4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_1d<float4>(A, x, A_DATA_RANGE);
		float4 a = make_float4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_double(double1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	double1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_1d<float>(A, x, A_DATA_RANGE);
		double1 a = make_double1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_double2(double2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	double2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_1d<float2>(A, x, A_DATA_RANGE);
		double2 a = make_double2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_double3(double3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	double3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_1d<float3>(A, x, A_DATA_RANGE);
		double3 a = make_double3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_1d_double4(double4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	double4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_1d<float4>(A, x, A_DATA_RANGE);
		double4 a = make_double4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_char(char* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	char* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_2d<float>(A, x, y, A_DATA_RANGE);
		char a = (val);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_char2(char2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	char2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_2d<float2>(A, x, y, A_DATA_RANGE);
		char2 a = make_char2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_char3(char3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	char3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_2d<float3>(A, x, y, A_DATA_RANGE);
		char3 a = make_char3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_char4(char4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	char4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_2d<float4>(A, x, y, A_DATA_RANGE);
		char4 a = make_char4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_uchar(uchar* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_2d<float>(A, x, y, A_DATA_RANGE);
		uchar a = (val);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_uchar2(uchar2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_2d<float2>(A, x, y, A_DATA_RANGE);
		uchar2 a = make_uchar2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_uchar3(uchar3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_2d<float3>(A, x, y, A_DATA_RANGE);
		uchar3 a = make_uchar3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_uchar4(uchar4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_2d<float4>(A, x, y, A_DATA_RANGE);
		uchar4 a = make_uchar4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_short(short* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_2d<float>(A, x, y, A_DATA_RANGE);
		short a = (val);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_short2(short2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_2d<float2>(A, x, y, A_DATA_RANGE);
		short2 a = make_short2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_short3(short3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_2d<float3>(A, x, y, A_DATA_RANGE);
		short3 a = make_short3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_short4(short4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_2d<float4>(A, x, y, A_DATA_RANGE);
		short4 a = make_short4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_ushort(ushort* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	ushort* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_2d<float>(A, x, y, A_DATA_RANGE);
		ushort a = (val);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_ushort2(ushort2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	ushort2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_2d<float2>(A, x, y, A_DATA_RANGE);
		ushort2 a = make_ushort2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_ushort3(ushort3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	ushort3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_2d<float3>(A, x, y, A_DATA_RANGE);
		ushort3 a = make_ushort3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_ushort4(ushort4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	ushort4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_2d<float4>(A, x, y, A_DATA_RANGE);
		ushort4 a = make_ushort4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_int(int* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_2d<float>(A, x, y, A_DATA_RANGE);
		int a = (val);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_int2(int2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_2d<float2>(A, x, y, A_DATA_RANGE);
		int2 a = make_int2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_int3(int3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_2d<float3>(A, x, y, A_DATA_RANGE);
		int3 a = make_int3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_int4(int4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_2d<float4>(A, x, y, A_DATA_RANGE);
		int4 a = make_int4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_uint(uint* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uint* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_2d<float>(A, x, y, A_DATA_RANGE);
		uint a = (val);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_uint2(uint2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uint2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_2d<float2>(A, x, y, A_DATA_RANGE);
		uint2 a = make_uint2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_uint3(uint3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uint3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_2d<float3>(A, x, y, A_DATA_RANGE);
		uint3 a = make_uint3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_uint4(uint4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uint4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_2d<float4>(A, x, y, A_DATA_RANGE);
		uint4 a = make_uint4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_float(float* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_2d<float>(A, x, y, A_DATA_RANGE);
		float a = (val);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_float2(float2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_2d<float2>(A, x, y, A_DATA_RANGE);
		float2 a = make_float2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_float3(float3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_2d<float3>(A, x, y, A_DATA_RANGE);
		float3 a = make_float3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_float4(float4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_2d<float4>(A, x, y, A_DATA_RANGE);
		float4 a = make_float4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_double(double* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	double* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_2d<float>(A, x, y, A_DATA_RANGE);
		double a = (val);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_double2(double2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	double2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_2d<float2>(A, x, y, A_DATA_RANGE);
		double2 a = make_double2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_double3(double3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	double3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_2d<float3>(A, x, y, A_DATA_RANGE);
		double3 a = make_double3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_double4(double4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	double4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_2d<float4>(A, x, y, A_DATA_RANGE);
		double4 a = make_double4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_RGB(RGB* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	RGB* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_2d<float3>(A, x, y, A_DATA_RANGE);
		RGB a = RGB(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_2d_RGBA(RGBA* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	RGBA* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_2d<float4>(A, x, y, A_DATA_RANGE);
		RGBA a = RGBA(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_char(char1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	char1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_3d<float>(A, x, y, z, A_DATA_RANGE);
		char1 a = make_char1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_char2(char2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	char2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_3d<float2>(A, x, y, z, A_DATA_RANGE);
		char2 a = make_char2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_char3(char3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	char3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_3d<float3>(A, x, y, z, A_DATA_RANGE);
		char3 a = make_char3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_char4(char4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	char4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_3d<float4>(A, x, y, z, A_DATA_RANGE);
		char4 a = make_char4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_uchar(uchar1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_3d<float>(A, x, y, z, A_DATA_RANGE);
		uchar1 a = make_uchar1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_uchar2(uchar2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_3d<float2>(A, x, y, z, A_DATA_RANGE);
		uchar2 a = make_uchar2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_uchar3(uchar3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_3d<float3>(A, x, y, z, A_DATA_RANGE);
		uchar3 a = make_uchar3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_uchar4(uchar4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uchar4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_3d<float4>(A, x, y, z, A_DATA_RANGE);
		uchar4 a = make_uchar4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_short(short1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_3d<float>(A, x, y, z, A_DATA_RANGE);
		short1 a = make_short1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_short2(short2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_3d<float2>(A, x, y, z, A_DATA_RANGE);
		short2 a = make_short2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_short3(short3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_3d<float3>(A, x, y, z, A_DATA_RANGE);
		short3 a = make_short3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_short4(short4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	short4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_3d<float4>(A, x, y, z, A_DATA_RANGE);
		short4 a = make_short4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_ushort(ushort1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	ushort1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_3d<float>(A, x, y, z, A_DATA_RANGE);
		ushort1 a = make_ushort1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_ushort2(ushort2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	ushort2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_3d<float2>(A, x, y, z, A_DATA_RANGE);
		ushort2 a = make_ushort2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_ushort3(ushort3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	ushort3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_3d<float3>(A, x, y, z, A_DATA_RANGE);
		ushort3 a = make_ushort3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_ushort4(ushort4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	ushort4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_3d<float4>(A, x, y, z, A_DATA_RANGE);
		ushort4 a = make_ushort4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_int(int1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_3d<float>(A, x, y, z, A_DATA_RANGE);
		int1 a = make_int1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_int2(int2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_3d<float2>(A, x, y, z, A_DATA_RANGE);
		int2 a = make_int2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_int3(int3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_3d<float3>(A, x, y, z, A_DATA_RANGE);
		int3 a = make_int3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_int4(int4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	int4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_3d<float4>(A, x, y, z, A_DATA_RANGE);
		int4 a = make_int4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_uint(uint1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uint1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_3d<float>(A, x, y, z, A_DATA_RANGE);
		uint1 a = make_uint1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_uint2(uint2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uint2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_3d<float2>(A, x, y, z, A_DATA_RANGE);
		uint2 a = make_uint2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_uint3(uint3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uint3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_3d<float3>(A, x, y, z, A_DATA_RANGE);
		uint3 a = make_uint3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_uint4(uint4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	uint4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_3d<float4>(A, x, y, z, A_DATA_RANGE);
		uint4 a = make_uint4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_float(float1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_3d<float>(A, x, y, z, A_DATA_RANGE);
		float1 a = make_float1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_float2(float2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_3d<float2>(A, x, y, z, A_DATA_RANGE);
		float2 a = make_float2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_float3(float3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_3d<float3>(A, x, y, z, A_DATA_RANGE);
		float3 a = make_float3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_float4(float4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	float4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_3d<float4>(A, x, y, z, A_DATA_RANGE);
		float4 a = make_float4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_double(double1* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	double1* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float val = point_query_3d<float>(A, x, y, z, A_DATA_RANGE);
		double1 a = make_double1(val);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_double2(double2* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	double2* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float2 val = point_query_3d<float2>(A, x, y, z, A_DATA_RANGE);
		double2 a = make_double2(val.x, val.y);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_double3(double3* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	double3* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float3 val = point_query_3d<float3>(A, x, y, z, A_DATA_RANGE);
		double3 a = make_double3(val.x, val.y, val.z);
		rb[idx] = a;
	}
						

	__global__ void writing_3d_double4(double4* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	double4* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float4 val = point_query_3d<float4>(A, x, y, z, A_DATA_RANGE);
		double4 a = make_double4(val.x, val.y, val.z, val.w);
		rb[idx] = a;
	}
						
}


template<typename R, typename T> GPU R point_query_3d_channel(T* image, int x, int y, int z,VIVALDI_DATA_RANGE* sdr, int channel, int num_channel){
	int4 data_start = sdr->data_start;
	int4 data_end = sdr->data_end;

	int4 full_data_start = sdr->full_data_start;
	int4 full_data_end = sdr->full_data_end;
	
	int4 buffer_start = sdr->buffer_start;
	int4 buffer_end = sdr->buffer_end;

	// check in 
    int border =0;
    int halo = sdr->data_halo;

	R rt;
		int X = buffer_end.x - buffer_start.x;
		int Y = buffer_end.y - buffer_start.y;
		int Z = buffer_end.z - buffer_start.z;

		// Data coordinate input
		// border check
		int flag1, flag2, flag3; // -1, 0, 1
		flag1 = check_in_border(x, full_data_start.x, full_data_end.x);
		flag2 = check_in_border(y, full_data_start.y, full_data_end.y);
		flag3 = check_in_border(z, full_data_start.z, full_data_end.z);

		bool flag; // border 
		flag = (1 <= border && border <= 4);

		if( (!flag && (flag1 == 0 && flag2 == 0 && flag3 == 0)) || (flag)){
			// Border calculation
			x = border_handling(x, full_data_start.x, full_data_end.x, border);
			y = border_handling(y, full_data_start.y, full_data_end.y, border);
			z = border_handling(z, full_data_start.z, full_data_end.z, border);

			bool flag_x = range_check(x, data_start.x, data_end.x);
			bool flag_y = range_check(y, data_start.y, data_end.y);
			bool flag_z = range_check(z, data_start.z, data_end.z);

			if(flag_x && flag_y && flag_z){		
				// to Buffer coordinate 
				x = x - buffer_start.x;
				y = y - buffer_start.y;
				z = z - buffer_start.z;

                z += halo;

				rt = convert(image[num_channel*(z*Y*X + y*X + x)+channel]);
				//rt = convert(image[channel*(Z*Y*X) +z*Y*X + y*X + x]);
			}else{
				rt = initial(image[0]);
			}
		}else{
			rt = initial(image[0]);
		}
	
	return rt;
}



template<typename R, typename T> GPU R point_query_3d_channel(T* image, float3 p, VIVALDI_DATA_RANGE* sdr, int channel, int num_channel){
	int4 data_start = sdr->data_start;
	int4 data_end = sdr->data_end;

	int4 full_data_start = sdr->full_data_start;
	int4 full_data_end = sdr->full_data_end;
	
	int4 buffer_start = sdr->buffer_start;
	int4 buffer_end = sdr->buffer_end;

	int x = p.x;
	int y = p.y;
	int z = p.z;

	// check in 
    int border =0;

	R rt;
		int X = buffer_end.x - buffer_start.x;
		int Y = buffer_end.y - buffer_start.y;
		int Z = buffer_end.z - buffer_start.z;

		// Data coordinate input
		// border check
		int flag1, flag2, flag3; // -1, 0, 1
		flag1 = check_in_border(x, full_data_start.x, full_data_end.x);
		flag2 = check_in_border(y, full_data_start.y, full_data_end.y);
		flag3 = check_in_border(z, full_data_start.z, full_data_end.z);

		bool flag; // border 
		flag = (1 <= border && border <= 4);

		if( (!flag && (flag1 == 0 && flag2 == 0 && flag3 == 0)) || (flag)){
			// Border calculation
			x = border_handling(x, full_data_start.x, full_data_end.x, border);
			y = border_handling(y, full_data_start.y, full_data_end.y, border);
			z = border_handling(z, full_data_start.z, full_data_end.z, border);

			bool flag_x = range_check(x, data_start.x, data_end.x);
			bool flag_y = range_check(y, data_start.y, data_end.y);
			bool flag_z = range_check(z, data_start.z, data_end.z);

			if(flag_x && flag_y && flag_z){		
				// to Buffer coordinate 
				x = x - buffer_start.x;
				y = y - buffer_start.y;
				z = z - buffer_start.z;

				rt = convert(image[num_channel*(z*Y*X + y*X + x)+channel]);
				//rt = convert(image[channel*(Z*Y*X) +z*Y*X + y*X + x]);
			}else{
				rt = initial(image[0]);
			}
		}else{
			rt = initial(image[0]);
		}
	
	return rt;
}

template<typename R, typename T> GPU R linear_query_3d_channel(T* volume, float3 p, VIVALDI_DATA_RANGE* sdr, int channel, int num_channel){
    //tri linear interpolation
    int border =0;

    float x,y,z;
    x = p.x;
    y = p.y;
    z = p.z;
	
    int fx = floor(x);
    int fy = floor(y);
    int fz = floor(z);
    int cx = ceil(x);
    int cy = ceil(y);
    int cz = ceil(z);
	
	R q000 = point_query_3d_channel<R>(volume, fx, fy, fz, sdr,channel,num_channel);
	R q001 = point_query_3d_channel<R>(volume, fx, fy, cz, sdr,channel,num_channel);
	R q010 = point_query_3d_channel<R>(volume, fx, cy, fz, sdr,channel,num_channel);
	R q011 = point_query_3d_channel<R>(volume, fx, cy, cz, sdr,channel,num_channel);
	R q100 = point_query_3d_channel<R>(volume, cx, fy, fz, sdr,channel,num_channel);
	R q101 = point_query_3d_channel<R>(volume, cx, fy, cz, sdr,channel,num_channel);
	R q110 = point_query_3d_channel<R>(volume, cx, cy, fz, sdr,channel,num_channel);
	R q111 = point_query_3d_channel<R>(volume, cx, cy, cz, sdr,channel,num_channel);

    float dx = x - fx;
    float dy = y - fy;
    float dz = z - fz;

    // lerp along x
    R q00 = lerp(q000, q001, dx);
    R q01 = lerp(q010, q011, dx);
    R q10 = lerp(q100, q101, dx);
    R q11 = lerp(q110, q111, dx);

    // lerp along y
    R q0 = lerp(q00, q01, dy);
    R q1 = lerp(q10, q11, dy);

    // lerp along z
    R q = lerp(q0, q1, dz);
    //return q;
    return q000;
}


extern "C"{

__global__ void set_inv_modelview(float* rb, float* view1)
{
    for (int j = 0 ; j < 4 ; j++) 
        for (int i = 0 ; i < 4 ; i++) 
            inv_modelview[j][i] = view1[j*4+i];
}

__global__ void read_inv_modelview(float* rb, float* view1)
{
    for (int j = 0 ; j < 4 ; j++) 
        for (int i = 0 ; i < 4 ; i++) 
            rb[j*4+i] = inv_modelview[j][i];
}


//

__global__ void render_test(float* rb, float* volume, VIVALDI_DATA_RANGE* volume_DATA_RANGE, int width, int height, float4* transfer_func, float* inv_model, int xoff, int yoff, int num_channel) {

    int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (idx_x == 0 && idx_y == 0)
        printf("%f %f %f %f \n",transfer_func[0].x,transfer_func[0].y,transfer_func[0].z,transfer_func[0].w);

    rb[(idx_y * width + idx_x)*4 + 0] = transfer_func[0].x;
	rb[(idx_y * width + idx_x)*4 + 1] = transfer_func[0].y;
    rb[(idx_y * width + idx_x)*4 + 2] = transfer_func[0].z;
	rb[(idx_y * width + idx_x)*4 + 3] = transfer_func[0].w;
}


__global__ void render(float* rb, float* volume, VIVALDI_DATA_RANGE* volume_DATA_RANGE, int width, int height, float4* transfer_func, float* inv_model, int xoff, int yoff, int num_channel, float3* ray_direction) {

    int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_y = threadIdx.y + blockDim.y * blockIdx.y;

    if(width <= idx_x || height <= idx_y)return;
   

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        for (int j = 0 ; j < 4 ; j++)
            for (int i = 0 ; i < 4 ; i++)
            {   
                int inv_idx = j*4+i;
                inv_modelview[j][i] = inv_model[inv_idx];
            }
    }


    __syncthreads();
    
    line_iter line_iter;

    float4 color;

    float val_diff; 
    float4 top_col, bot_col;   

    //char *attris[11]  = {"rho","prs","tev","xdt","ydt","zdt","snd","grd","mat","v02","v03"};

    color = make_float4(0);
    //line_iter = orthogonal_iter(volume, idx_x+xoff, idx_y+yoff, 0.1, ray_direction[0], volume_DATA_RANGE);
    line_iter = orthogonal_iter(volume, idx_x+xoff, idx_y+yoff, 0.1, ray_direction[0], volume_DATA_RANGE);

    for(float3 elem = line_iter.begin(); line_iter.valid(); elem = line_iter.next()){
 
        float4 combined_color = make_float4(0,0,0,0);
        float4 tmp_val = make_float4(0);
        float4 tmp_val1 = make_float4(0);
        float4 tmp_val2 = make_float4(0);
        float4 tmp_val3 = make_float4(0);
        float4 tmp_val4 = make_float4(0);
        float4 tmp_val5 = make_float4(0);
        float4 tmp_val6 = make_float4(0);
        float4 tmp_val7 = make_float4(0);
        float sum_w = 0;
        
        float v02  = point_query_3d_channel<float>(volume, elem, volume_DATA_RANGE, 9, num_channel);
       
        float v02_min = 0;
        float v02_max = 1;
        if (v02 > v02_min && v02 <v02_max){ 
            v02 = (v02 - v02_min)/(v02_max-v02_min)*256.0;

            val_diff = v02- floor(v02);
            top_col = transfer_func[5*256+(int) ceil(v02)];
            bot_col = transfer_func[5*256+(int)floor(v02)];
            tmp_val = (top_col * (1-val_diff) + bot_col * val_diff);
            tmp_val.w = tmp_val.w*tmp_val.w;
        }

        float v03  = point_query_3d_channel<float>(volume, elem, volume_DATA_RANGE, 10, num_channel);
       
        float v03_min = 0;
        float v03_max = 1;
        if (v03 > v03_min && v03 <v03_max){ 
            v03 = (v03 - v03_min)/(v03_max-v03_min)*256.0;

            val_diff = v03- floor(v03);
            top_col = transfer_func[6*256+(int) ceil(v03)];
            bot_col = transfer_func[6*256+(int)floor(v03)];
            tmp_val6 = (top_col * (1-val_diff) + bot_col * val_diff);
            tmp_val6.w = tmp_val6.w * tmp_val6.w;
        }


        float rho  = point_query_3d_channel<float>(volume, elem, volume_DATA_RANGE, 0, num_channel);

        float rho_min = 0.002004;
        float rho_max = 1.014865; 
        
        if (rho > rho_min && rho <rho_max){ 
            rho = (rho - rho_min)/(rho_max-rho_min)*256.0;

            val_diff = rho- floor(rho);
            top_col = transfer_func[1*256+(int) ceil(rho)];
            bot_col = transfer_func[1*256+(int)floor(rho)];
            tmp_val1 = (top_col * (1-val_diff) + bot_col * val_diff)*1.0;
        }

        float tev  = point_query_3d_channel<float>(volume, elem, volume_DATA_RANGE, 2, num_channel);
        float tev_min = 0.012058;
        float tev_max = 0.3058619;


        if (tev > tev_min && tev < tev_max){
            tev = (tev - tev_min)/(tev_max-tev_min)*256.0;

            val_diff = tev- floor(tev);
            top_col = transfer_func[2*256+(int) ceil(tev)];
            bot_col = transfer_func[2*256+(int)floor(tev)];
            tmp_val2 = (top_col * (1-val_diff) + bot_col * val_diff);
            tmp_val2.w /= 30;
        }
        
        float mat  = point_query_3d_channel<float>(volume, elem, volume_DATA_RANGE, 8, num_channel);
 
        float mat_min = 1.001570;
        float mat_max = 3.000000; 
        
        if (mat > mat_min && mat <mat_max){ 
            mat = (mat - mat_min)/(mat_max-mat_min)*256.0;

            val_diff = mat- floor(mat);
            top_col = transfer_func[0*256+(int) ceil(mat)];
            bot_col = transfer_func[0*256+(int)floor(mat)];
            tmp_val3 = (top_col * (1-val_diff) + bot_col * val_diff);
//            tmp_val3.w /= 5;
        }

        float prs  = point_query_3d_channel<float>(volume, elem, volume_DATA_RANGE, 1, num_channel);
        float prs_min = 13966.178711;
        float prs_max = 513453952;

        if (prs > prs_min && prs <prs_max){ 
            prs = (prs - prs_min)/(prs_max-prs_min)*256.0;
    
            val_diff = prs- floor(prs);
            top_col = transfer_func[3*256+(int) ceil(prs)];
            bot_col = transfer_func[3*256+(int)floor(prs)];
            tmp_val4 = (top_col * (1-val_diff) + bot_col * val_diff);
        }


        float snd  = point_query_3d_channel<float>(volume, elem, volume_DATA_RANGE, 6, num_channel);
        float snd_min = 89.637283;
        float snd_max = 161575.781250;

        if (snd > snd_min && snd <snd_max){ 
            snd = (snd - snd_min)/(snd_max-snd_min)*256.0;
    
            val_diff = snd- floor(snd);
            top_col = transfer_func[4*256+(int) ceil(snd)];
            bot_col = transfer_func[4*256+(int)floor(snd)];
            tmp_val5 = (top_col * (1-val_diff) + bot_col * val_diff);
            tmp_val5.w /= 5;
        }


        if (combined_color.w < tmp_val.w)
            combined_color.w = tmp_val.w;
        if (combined_color.w < tmp_val1.w)
            combined_color.w = tmp_val1.w;
        if (combined_color.w < tmp_val2.w)
            combined_color.w = tmp_val2.w;
        if (combined_color.w < tmp_val3.w)
            combined_color.w = tmp_val3.w;
        if (combined_color.w < tmp_val4.w)
            combined_color.w = tmp_val4.w;
        if (combined_color.w < tmp_val5.w)
            combined_color.w = tmp_val5.w;
        if (combined_color.w < tmp_val6.w)
            combined_color.w = tmp_val6.w;



        if (combined_color.w > 0)
        {
            combined_color.x += tmp_val.x*tmp_val.w;
            combined_color.x += tmp_val1.x*tmp_val1.w;
            combined_color.x += tmp_val2.x*tmp_val2.w;
            combined_color.x += tmp_val3.x*tmp_val3.w;
            combined_color.x += tmp_val4.x*tmp_val4.w;
            combined_color.x += tmp_val5.x*tmp_val5.w;
            combined_color.x += tmp_val6.x*tmp_val6.w;
   
            combined_color.y += tmp_val.y*tmp_val.w;
            combined_color.y += tmp_val1.y*tmp_val1.w;
            combined_color.y += tmp_val2.y*tmp_val2.w;
            combined_color.y += tmp_val3.y*tmp_val3.w;
            combined_color.y += tmp_val4.y*tmp_val4.w;
            combined_color.y += tmp_val5.y*tmp_val5.w;
            combined_color.y += tmp_val6.y*tmp_val6.w;
 
            combined_color.z += tmp_val.z*tmp_val.w;
            combined_color.z += tmp_val1.z*tmp_val1.w;
            combined_color.z += tmp_val2.z*tmp_val2.w;
            combined_color.z += tmp_val3.z*tmp_val3.w;
            combined_color.z += tmp_val4.z*tmp_val4.w;
            combined_color.z += tmp_val5.z*tmp_val5.w;
            combined_color.z += tmp_val6.z*tmp_val6.w;
 
            sum_w += tmp_val.w;
            sum_w += tmp_val1.w;
            sum_w += tmp_val2.w;
            sum_w += tmp_val3.w;
            sum_w += tmp_val4.w;
            sum_w += tmp_val6.w;
 
            combined_color.x /= sum_w;
            combined_color.y /= sum_w;
            combined_color.z /= sum_w;
        }
            
        color = alpha_compositing_new(color, combined_color);

    }


	if (false)
    {
    	rb[(idx_y * width + idx_x)*4 + 0] = 255;
	    rb[(idx_y * width + idx_x)*4 + 1] = 0;
    	rb[(idx_y * width + idx_x)*4 + 2] = 0;
	    rb[(idx_y * width + idx_x)*4 + 3] = 255;
    }
    else 
    {
    	rb[(idx_y * width + idx_x)*4 + 0] = color.x;
	    rb[(idx_y * width + idx_x)*4 + 1] = color.y;
    	rb[(idx_y * width + idx_x)*4 + 2] = color.z;
    	rb[(idx_y * width + idx_x)*4 + 3] = color.w;
    }
    return;

}

__global__ void composite_uchar(unsigned char* output, float* input, int num_img, int dim_x, int dim_y) {
    int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_y = threadIdx.y + blockDim.y * blockIdx.y;

    if(dim_x <= idx_x || dim_y <= idx_y) return;

	
	int index   = (idx_y * dim_x + idx_x) * 4;
	//int index   = (1*dim_x*dim_y + idx_y * dim_x + idx_x) * 4;
	float r     = input[index + 0];
	float g     = input[index + 1];
	float b     = input[index + 2];
	float alpha = input[index + 3];

	//if(idx_x == 0 && idx_y == 0)
	//	printf("R: %f, G: %f, B: %f\n", r, g, b);

	float4 result = make_float4(r, g, b, alpha);

	//result = alpha_compositing(make_float4(0, 0, 0, 0), result);

    for(int i=1; i<num_img; i++) {
		// Channel = 4 (r, g, b, alpha)
		index = (i * dim_x * dim_y + idx_y * dim_x + idx_x) * 4;
        //val = input[idx_y * dim_x + idx_x];
		r     = input[index + 0];
		g     = input[index + 1];
		b     = input[index + 2];
		alpha = input[index + 3];

	//	if(idx_x == 0 && idx_y == 0)
	//		printf("cnt : %d | R: %f, G: %f, B: %f\n", i, r, g, b);
		float4 curr_col = make_float4(r, g, b, alpha);

		result = alpha_compositing_wo_alpha_new(result, curr_col);
		//result = alpha_compositing_new(result, curr_col);
    }
    result = background_white(result);

	output[(idx_y * dim_x + idx_x)*3 + 0] = result.x*255;
	output[(idx_y * dim_x + idx_x)*3 + 1] = result.y*255;
	output[(idx_y * dim_x + idx_x)*3 + 2] = result.z*255;

    return ;
}




__global__ void composite(float* output, float* input, int num_img, int dim_x, int dim_y) {
    int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_y = threadIdx.y + blockDim.y * blockIdx.y;

    if(dim_x <= idx_x || dim_y <= idx_y) return;

	
	int index   = (idx_y * dim_x + idx_x) * 4;
	//int index   = (1*dim_x*dim_y + idx_y * dim_x + idx_x) * 4;
	float r     = input[index + 0];
	float g     = input[index + 1];
	float b     = input[index + 2];
	float alpha = input[index + 3];

	//if(idx_x == 0 && idx_y == 0)
	//	printf("R: %f, G: %f, B: %f\n", r, g, b);

	float4 result = make_float4(r, g, b, alpha);

	//result = alpha_compositing(make_float4(0, 0, 0, 0), result);

    for(int i=1; i<num_img; i++) {
		// Channel = 4 (r, g, b, alpha)
		index = (i * dim_x * dim_y + idx_y * dim_x + idx_x) * 4;
        //val = input[idx_y * dim_x + idx_x];
		r     = input[index + 0];
		g     = input[index + 1];
		b     = input[index + 2];
		alpha = input[index + 3];

	//	if(idx_x == 0 && idx_y == 0)
	//		printf("cnt : %d | R: %f, G: %f, B: %f\n", i, r, g, b);
		float4 curr_col = make_float4(r, g, b, alpha);

		result = alpha_compositing_wo_alpha_new(result, curr_col);
		//result = alpha_compositing_new(result, curr_col);
    }
	output[(idx_y * dim_x + idx_x)*4 + 0] = result.x;
	output[(idx_y * dim_x + idx_x)*4 + 1] = result.y;
	output[(idx_y * dim_x + idx_x)*4 + 2] = result.z;
	output[(idx_y * dim_x + idx_x)*4 + 3] = result.w;

    return ;
}




}
