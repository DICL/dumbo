
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <fstream>

#include <cstdio>
#include <cstdint>

#include <vector>

using namespace std;




//----------------------------------------------------------------------------
// Vec.h 
//----------------------------------------------------------------------------
struct Vec3i{
 	union {
		struct {
			int x, y, z;
		};
		int data[3];
	};

	__host__ __device__ Vec3i() = default;
	__host__ __device__ Vec3i(int x, int y, int z): x(x), y(y), z(z) {}
	__host__ __device__ Vec3i(int v): x(v), y(v), z(v) {}

	__host__ __device__ Vec3i &operator+=(const Vec3i &r) { x+=r.x; y+=r.y; z+=r.z; return *this; }
	__host__ __device__ Vec3i &operator-=(const Vec3i &r) { x-=r.x; y-=r.y; z-=r.z; return *this; }
	__host__ __device__ Vec3i &operator*=(const Vec3i &r) { x*=r.x; y*=r.y; z*=r.z; return *this; }
	__host__ __device__ Vec3i &operator/=(const Vec3i &r) { x/=r.x; y/=r.y; z/=r.z; return *this; }

	int &operator[](int i) { return data[i]; }
	int operator[](int i) const { return data[i]; }

};

struct Vec3f{
	union {
		struct {
			float x, y, z;
		};
		float data[3];
	};

	__host__ __device__ Vec3f() = default;
	__host__ __device__ Vec3f(float x, float y, float z): x(x), y(y), z(z) {}
	__host__ __device__ Vec3f(float v): x(v), y(v), z(v) {}

	__host__ __device__ Vec3f &operator+=(const Vec3f &r) { x+=r.x; y+=r.y; z+=r.z; return *this; }
	__host__ __device__ Vec3f &operator-=(const Vec3f &r) { x-=r.x; y-=r.y; z-=r.z; return *this; }
	__host__ __device__ Vec3f &operator*=(const Vec3f &r) { x*=r.x; y*=r.y; z*=r.z; return *this; }
	__host__ __device__ Vec3f &operator/=(const Vec3f &r) { x/=r.x; y/=r.y; z/=r.z; return *this; }

	__host__ __device__ float &operator[](int i) { return data[i]; }
	__host__ __device__ float operator[](int i) const { return data[i]; }

};

static inline Vec3f operator-(const Vec3f &l, const Vec3f &r) { return {l.x - r.x, l.y - r.y, l.z - r.z}; }
static inline Vec3f operator/(const Vec3f &l, const Vec3f &r) { return {l.x / r.x, l.y / r.y, l.z / r.z}; }

static inline Vec3f cross(const Vec3f &v1, const Vec3f &v2) { return Vec3f(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x); } 
//static inline Vec3f normalize(const Vec3f &v) { return v / Vec3f(sqrtf(v.x*v.x + v.y+v.y + v.z*v.z)); }
__host__ __device__ Vec3f normalize(const Vec3f &v) { return v / Vec3f(sqrtf(v.x*v.x + v.y+v.y + v.z*v.z)); }

//static inline Vec3f operator-(const Vec3f &l, const Vec3f &r) { return {l.x - r.x, l.y - r.y, l.z - r.z}; }
//static inline Vec3f operator/(const Vec3f &l, const Vec3f &r) { return {l.x / r.x, l.y / r.y, l.z / r.z}; }

//static inline Vec3f cross(const Vec3f &v1, const Vec3f &v2) { return Vec3f(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x); } 
//static inline Vec3f normalize(const Vec3f &v) { return v / Vec3f(std::sqrt(v.x*v.x + v.y+v.y + v.z*v.z)); }
//----------------------------------------------------------------------------
// Geometry
//----------------------------------------------------------------------------


struct Vertex {
	Vec3f position;
	Vec3f normal;
//	Vec3f color;
};
//float* voxels;

//Vertex vertices[MAXIMUM_NUM];
//unsigned int vertices_num = 0;

__constant__ uint64_t marching_cube_tris[256] = {
	0ULL, 33793ULL, 36945ULL, 159668546ULL,
	18961ULL, 144771090ULL, 5851666ULL, 595283255635ULL,
	20913ULL, 67640146ULL, 193993474ULL, 655980856339ULL,
	88782242ULL, 736732689667ULL, 797430812739ULL, 194554754ULL,
	26657ULL, 104867330ULL, 136709522ULL, 298069416227ULL,
	109224258ULL, 8877909667ULL, 318136408323ULL, 1567994331701604ULL,
	189884450ULL, 350847647843ULL, 559958167731ULL, 3256298596865604ULL,
	447393122899ULL, 651646838401572ULL, 2538311371089956ULL, 737032694307ULL,
	29329ULL, 43484162ULL, 91358498ULL, 374810899075ULL,
	158485010ULL, 178117478419ULL, 88675058979ULL, 433581536604804ULL,
	158486962ULL, 649105605635ULL, 4866906995ULL, 3220959471609924ULL,
	649165714851ULL, 3184943915608436ULL, 570691368417972ULL, 595804498035ULL,
	124295042ULL, 431498018963ULL, 508238522371ULL, 91518530ULL,
	318240155763ULL, 291789778348404ULL, 1830001131721892ULL, 375363605923ULL,
	777781811075ULL, 1136111028516116ULL, 3097834205243396ULL, 508001629971ULL,
	2663607373704004ULL, 680242583802939237ULL, 333380770766129845ULL, 179746658ULL,
	42545ULL, 138437538ULL, 93365810ULL, 713842853011ULL,
	73602098ULL, 69575510115ULL, 23964357683ULL, 868078761575828ULL,
	28681778ULL, 713778574611ULL, 250912709379ULL, 2323825233181284ULL,
	302080811955ULL, 3184439127991172ULL, 1694042660682596ULL, 796909779811ULL,
	176306722ULL, 150327278147ULL, 619854856867ULL, 1005252473234484ULL,
	211025400963ULL, 36712706ULL, 360743481544788ULL, 150627258963ULL,
	117482600995ULL, 1024968212107700ULL, 2535169275963444ULL, 4734473194086550421ULL,
	628107696687956ULL, 9399128243ULL, 5198438490361643573ULL, 194220594ULL,
	104474994ULL, 566996932387ULL, 427920028243ULL, 2014821863433780ULL,
	492093858627ULL, 147361150235284ULL, 2005882975110676ULL, 9671606099636618005ULL,
	777701008947ULL, 3185463219618820ULL, 482784926917540ULL, 2900953068249785909ULL,
	1754182023747364ULL, 4274848857537943333ULL, 13198752741767688709ULL, 2015093490989156ULL,
	591272318771ULL, 2659758091419812ULL, 1531044293118596ULL, 298306479155ULL,
	408509245114388ULL, 210504348563ULL, 9248164405801223541ULL, 91321106ULL,
	2660352816454484ULL, 680170263324308757ULL, 8333659837799955077ULL, 482966828984116ULL,
	4274926723105633605ULL, 3184439197724820ULL, 192104450ULL, 15217ULL,
	45937ULL, 129205250ULL, 129208402ULL, 529245952323ULL,
	169097138ULL, 770695537027ULL, 382310500883ULL, 2838550742137652ULL,
	122763026ULL, 277045793139ULL, 81608128403ULL, 1991870397907988ULL,
	362778151475ULL, 2059003085103236ULL, 2132572377842852ULL, 655681091891ULL,
	58419234ULL, 239280858627ULL, 529092143139ULL, 1568257451898804ULL,
	447235128115ULL, 679678845236084ULL, 2167161349491220ULL, 1554184567314086709ULL,
	165479003923ULL, 1428768988226596ULL, 977710670185060ULL, 10550024711307499077ULL,
	1305410032576132ULL, 11779770265620358997ULL, 333446212255967269ULL, 978168444447012ULL,
	162736434ULL, 35596216627ULL, 138295313843ULL, 891861543990356ULL,
	692616541075ULL, 3151866750863876ULL, 100103641866564ULL, 6572336607016932133ULL,
	215036012883ULL, 726936420696196ULL, 52433666ULL, 82160664963ULL,
	2588613720361524ULL, 5802089162353039525ULL, 214799000387ULL, 144876322ULL,
	668013605731ULL, 110616894681956ULL, 1601657732871812ULL, 430945547955ULL,
	3156382366321172ULL, 7644494644932993285ULL, 3928124806469601813ULL, 3155990846772900ULL,
	339991010498708ULL, 10743689387941597493ULL, 5103845475ULL, 105070898ULL,
	3928064910068824213ULL, 156265010ULL, 1305138421793636ULL, 27185ULL,
	195459938ULL, 567044449971ULL, 382447549283ULL, 2175279159592324ULL,
	443529919251ULL, 195059004769796ULL, 2165424908404116ULL, 1554158691063110021ULL,
	504228368803ULL, 1436350466655236ULL, 27584723588724ULL, 1900945754488837749ULL,
	122971970ULL, 443829749251ULL, 302601798803ULL, 108558722ULL,
	724700725875ULL, 43570095105972ULL, 2295263717447940ULL, 2860446751369014181ULL,
	2165106202149444ULL, 69275726195ULL, 2860543885641537797ULL, 2165106320445780ULL,
	2280890014640004ULL, 11820349930268368933ULL, 8721082628082003989ULL, 127050770ULL,
	503707084675ULL, 122834978ULL, 2538193642857604ULL, 10129ULL,
	801441490467ULL, 2923200302876740ULL, 1443359556281892ULL, 2901063790822564949ULL,
	2728339631923524ULL, 7103874718248233397ULL, 12775311047932294245ULL, 95520290ULL,
	2623783208098404ULL, 1900908618382410757ULL, 137742672547ULL, 2323440239468964ULL,
	362478212387ULL, 727199575803140ULL, 73425410ULL, 34337ULL,
	163101314ULL, 668566030659ULL, 801204361987ULL, 73030562ULL,
	591509145619ULL, 162574594ULL, 100608342969108ULL, 5553ULL,
	724147968595ULL, 1436604830452292ULL, 176259090ULL, 42001ULL,
	143955266ULL, 2385ULL, 18433ULL, 0ULL,
};


__host__ __device__ int offset_3d(const Vec3i &p, const Vec3i &size)
{
	return (p.z * size.y + p.y) * size.x + p.x;
}

__host__ __device__ int offset_3d(const int x, const int y, const int z, const int sizex, const int sizey, const int sizez)
{
//	return (p.z * size.y + p.y) * size.x + p.x;
    return (z*sizey + y)*sizex + x;
}




__host__ __device__ bool valid(int x, int y, int z, int dimx, int dimy, int dimz)
{
    if ( x >= 0 && x < dimx)
        if ( y >= 0 && y < dimy)
            if ( z >= 0 && z < dimz)
                return true;

   return false; 
}

__host__ __device__ void triangle(Vertex &va, Vertex &vb, Vertex &vc)
{
	const Vec3f ab = va.position - vb.position;
	const Vec3f cb = vc.position - vb.position;
	const Vec3f n = cross(cb, ab);
	va.normal += n;
	vb.normal += n;
	vc.normal += n;
}

__host__ __device__ void do_edge (Vertex* edge_indices, int n_edge, float va, float vb, int axis, const Vec3f &base) {
    if ((va < 0.0) == (vb < 0.0))
        return;

    Vec3f v = base;
    v[axis] += va / (va - vb);
    edge_indices[n_edge] = {v,Vec3f(0)};
    //edge_indices[n_edge] = {v};
};


__device__ void do_edge (float* edge_indices, int n_edge, float va, float vb, int axis, int x, int y, int z) {
    if ((va < 0.0) == (vb < 0.0))
        return;

    float v[3];
    v[0] = x*1.0;
    v[1] = y*1.0;
    v[2] = z*1.0;

    v[axis] += va / (va - vb);
    edge_indices[6*n_edge+0] = v[0];
    edge_indices[6*n_edge+1] = v[1];
    edge_indices[6*n_edge+2] = v[2];
    edge_indices[6*n_edge+3] = 0.0f;
    edge_indices[6*n_edge+4] = 0.0f;
    edge_indices[6*n_edge+5] = 0.0f;

};

//__device__ unsigned int num_vert[1] = {0};

//generate_geometry_kernel<<<grid,block>>>(d_voxels,d_vertices,marching_cube_tris_gpu,num_vert);
//__global__ void generate_geometry_kernel(float* voxels, Vertex* vertices)

extern "C"{

__global__ void mc_center(Vertex* vertices,unsigned char* voxels, int bidx, int boff, int dimx, int dimy, int dimz, unsigned int* num_vert,int* isolist, int start, int end,int N)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    //printf("Call kernel [%d,%d,%d] \n",x,y,z);
    if ( x >= 0 && y >= 0 && z >= 0 &&
         x < dimx && y < dimy && z < dimz){
            
        z += 1;

        float vs[8];

        for (int i = 0 ; i < 8 ; i ++)
            vs[i] = 0;

        for (int j = start ; j < end ; j++)
        { 
            int isovalue = isolist[j];
 
            if (valid(x  ,y  ,z  ,dimx,dimy,dimz+2))
                vs[0] = 1.0*voxels[offset_3d({x,   y,   z},   Vec3i((dimx)))];
            if (valid(x+1,y  ,z  ,dimx,dimy,dimz+2))
                vs[1] = 1.0*voxels[offset_3d({x+1, y,   z},   Vec3i((dimx)))];
            if (valid(x  ,y+1,z  ,dimx,dimy,dimz+2))
                vs[2] = 1.0*voxels[offset_3d({x,   y+1, z},   Vec3i((dimx)))];
            if (valid(x+1,y+1,z  ,dimx,dimy,dimz+2))
                vs[3] = 1.0*voxels[offset_3d({x+1, y+1, z},   Vec3i((dimx)))];
            if (valid(x  ,y  ,z+1,dimx,dimy,dimz+2))
                vs[4] = 1.0*voxels[offset_3d({x,   y,   z+1}, Vec3i((dimx)))];
            if (valid(x+1,y  ,z+1,dimx,dimy,dimz+2))
                vs[5] = 1.0*voxels[offset_3d({x+1, y,   z+1}, Vec3i((dimx)))];
            if (valid(x  ,y+1,z+1,dimx,dimy,dimz+2))
                vs[6] = 1.0*voxels[offset_3d({x,   y+1, z+1}, Vec3i((dimx)))];
            if (valid(x+1,y+1,z+1,dimx,dimy,dimz+2))
                vs[7] = 1.0*voxels[offset_3d({x+1, y+1, z+1}, Vec3i((dimx)))];

            for (int i = 0 ; i < 8 ; i++){
                vs[i] = vs[i] - 0.5;
            }

            const int config_n =
                ((vs[0] < 0.0f) << 0) | // *1
                ((vs[1] < 0.0f) << 1) | // *2
                ((vs[2] < 0.0f) << 2) | // *4
                ((vs[3] < 0.0f) << 3) | // *8
                ((vs[4] < 0.0f) << 4) | // *16
                ((vs[5] < 0.0f) << 5) | // *32
                ((vs[6] < 0.0f) << 6) | // *64
                ((vs[7] < 0.0f) << 7);  // *128

            if (config_n == 0 || config_n == 255)
                continue; 
        //int index_base1 = atomicAdd(num_vert,1);

        //                vector<Vertex> vert;
        //              int edge_indices[12];
            //float edge_indices[12*6];
           
            z -= 1;
            z += bidx*boff;

            x -= N/2;
            y -= N/2;
            z -= N/2;
         
            Vertex edge_indices[12];

            do_edge(edge_indices, 0,  vs[0], vs[1], 0, Vec3f(x, y,   z));
            do_edge(edge_indices, 1,  vs[2], vs[3], 0, Vec3f(x, y+1, z));
            do_edge(edge_indices, 2,  vs[4], vs[5], 0, Vec3f(x, y,   z+1));
            do_edge(edge_indices, 3,  vs[6], vs[7], 0, Vec3f(x, y+1, z+1));

            do_edge(edge_indices, 4,  vs[0], vs[2], 1, Vec3f(x,   y, z));
            do_edge(edge_indices, 5,  vs[1], vs[3], 1, Vec3f(x+1, y, z));
            do_edge(edge_indices, 6,  vs[4], vs[6], 1, Vec3f(x,   y, z+1));
            do_edge(edge_indices, 7,  vs[5], vs[7], 1, Vec3f(x+1, y, z+1));

            do_edge(edge_indices, 8,  vs[0], vs[4], 2, Vec3f(x,   y,   z));
            do_edge(edge_indices, 9,  vs[1], vs[5], 2, Vec3f(x+1, y,   z));
            do_edge(edge_indices, 10, vs[2], vs[6], 2, Vec3f(x,   y+1, z));
            do_edge(edge_indices, 11, vs[3], vs[7], 2, Vec3f(x+1, y+1, z));

            const uint64_t config = marching_cube_tris[config_n];
            const int n_triangles = config & 0xF; // Maximum 15
            const int n_indices = n_triangles * 3; // Maximu 45

            int offset = 4;
            unsigned int index_base = atomicAdd(num_vert,n_indices);
            //num_vert+= n_indices;

            if (true){
                unsigned int index = index_base;
                for (int i = 0; i < n_indices; i++) {
                    const int edge = (config >> offset) & 0xF;
                    //vertices.push_back(edge_indices[edge]);
                    vertices[index++] = edge_indices[edge];
                    offset += 4;
                }

                for (int i = 0; i < n_triangles; i++) {
                    triangle(vertices[index_base+i*3+2],
                            vertices[index_base+i*3+1],
                            vertices[index_base+i*3+0]);

                }

                for (int i = 0; i < n_indices; i++) {
                    vertices[index_base+i].normal = normalize(vertices[index_base+i].normal);
                    vertices[index_base+i].position.data[0] /= N;
                    vertices[index_base+i].position.data[1] /= N;
                    vertices[index_base+i].position.data[2] /= N;

/*
                    int color_num = j%table_num;
        
                    vertices[index_base+i].color.data[0] = color_table[3*color_num+0];
                    vertices[index_base+i].color.data[1] = color_table[3*color_num+1];
                    vertices[index_base+i].color.data[2] = color_table[3*color_num+2];

                    if ( j % 4 == 0){dd
                        vertices[index_base+i].color.data[0] = 0.2;
                        vertices[index_base+i].color.data[1] = 1.0;
                        vertices[index_base+i].color.data[2] = 0.2;
                    }
                    else if ( j % 4 == 1){
                        vertices[index_base+i].color.data[0] = 1.0;
                        vertices[index_base+i].color.data[1] = 0.2;
                        vertices[index_base+i].color.data[2] = 0.2;
                    }
                    else if ( j % 4 == 2){
                        vertices[index_base+i].color.data[0] = 0.2;
                        vertices[index_base+i].color.data[1] = 0.2;
                        vertices[index_base+i].color.data[2] = 1.0;
                    }else {
                        vertices[index_base+i].color.data[0] = 0.2;
                        vertices[index_base+i].color.data[1] = 1.0;
                        vertices[index_base+i].color.data[2] = 1.0;
                    }
*/
                }


            }

        }
    }
}
__global__ void mc_char(Vertex* vertices,unsigned char* voxels, int bidx, int boff, int dimx, int dimy, int dimz, unsigned int* num_vert,int* isolist, int start, int end,int N)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    //printf("Call kernel [%d,%d,%d] \n",x,y,z);
    if ( x >= 0 && y >= 0 && z >= 0 &&
         x < dimx && y < dimy && z < dimz){
            
        z += 1;

        float vs[8];

        for (int i = 0 ; i < 8 ; i ++)
            vs[i] = 0;

        for (int j = start ; j < end ; j++)
        { 
            int isovalue = isolist[j];
 
            if (valid(x  ,y  ,z  ,dimx,dimy,dimz+2))
                vs[0] = 1.0*voxels[offset_3d({x,   y,   z},   Vec3i((dimx)))];
            if (valid(x+1,y  ,z  ,dimx,dimy,dimz+2))
                vs[1] = 1.0*voxels[offset_3d({x+1, y,   z},   Vec3i((dimx)))];
            if (valid(x  ,y+1,z  ,dimx,dimy,dimz+2))
                vs[2] = 1.0*voxels[offset_3d({x,   y+1, z},   Vec3i((dimx)))];
            if (valid(x+1,y+1,z  ,dimx,dimy,dimz+2))
                vs[3] = 1.0*voxels[offset_3d({x+1, y+1, z},   Vec3i((dimx)))];
            if (valid(x  ,y  ,z+1,dimx,dimy,dimz+2))
                vs[4] = 1.0*voxels[offset_3d({x,   y,   z+1}, Vec3i((dimx)))];
            if (valid(x+1,y  ,z+1,dimx,dimy,dimz+2))
                vs[5] = 1.0*voxels[offset_3d({x+1, y,   z+1}, Vec3i((dimx)))];
            if (valid(x  ,y+1,z+1,dimx,dimy,dimz+2))
                vs[6] = 1.0*voxels[offset_3d({x,   y+1, z+1}, Vec3i((dimx)))];
            if (valid(x+1,y+1,z+1,dimx,dimy,dimz+2))
                vs[7] = 1.0*voxels[offset_3d({x+1, y+1, z+1}, Vec3i((dimx)))];

            for (int i = 0 ; i < 8 ; i++){
                vs[i] = vs[i] - 0.5;
            }

            const int config_n =
                ((vs[0] < 0.0f) << 0) | // *1
                ((vs[1] < 0.0f) << 1) | // *2
                ((vs[2] < 0.0f) << 2) | // *4
                ((vs[3] < 0.0f) << 3) | // *8
                ((vs[4] < 0.0f) << 4) | // *16
                ((vs[5] < 0.0f) << 5) | // *32
                ((vs[6] < 0.0f) << 6) | // *64
                ((vs[7] < 0.0f) << 7);  // *128

            if (config_n == 0 || config_n == 255)
                continue; 
        //int index_base1 = atomicAdd(num_vert,1);

        //                vector<Vertex> vert;
        //              int edge_indices[12];
            //float edge_indices[12*6];
           
            z -= 1;
            z += bidx*boff;

            Vertex edge_indices[12];

            do_edge(edge_indices, 0,  vs[0], vs[1], 0, Vec3f(x, y,   z));
            do_edge(edge_indices, 1,  vs[2], vs[3], 0, Vec3f(x, y+1, z));
            do_edge(edge_indices, 2,  vs[4], vs[5], 0, Vec3f(x, y,   z+1));
            do_edge(edge_indices, 3,  vs[6], vs[7], 0, Vec3f(x, y+1, z+1));

            do_edge(edge_indices, 4,  vs[0], vs[2], 1, Vec3f(x,   y, z));
            do_edge(edge_indices, 5,  vs[1], vs[3], 1, Vec3f(x+1, y, z));
            do_edge(edge_indices, 6,  vs[4], vs[6], 1, Vec3f(x,   y, z+1));
            do_edge(edge_indices, 7,  vs[5], vs[7], 1, Vec3f(x+1, y, z+1));

            do_edge(edge_indices, 8,  vs[0], vs[4], 2, Vec3f(x,   y,   z));
            do_edge(edge_indices, 9,  vs[1], vs[5], 2, Vec3f(x+1, y,   z));
            do_edge(edge_indices, 10, vs[2], vs[6], 2, Vec3f(x,   y+1, z));
            do_edge(edge_indices, 11, vs[3], vs[7], 2, Vec3f(x+1, y+1, z));

            const uint64_t config = marching_cube_tris[config_n];
            const int n_triangles = config & 0xF; // Maximum 15
            const int n_indices = n_triangles * 3; // Maximu 45

            int offset = 4;
            unsigned int index_base = atomicAdd(num_vert,n_indices);
            //num_vert+= n_indices;

            if (true){
                unsigned int index = index_base;
                for (int i = 0; i < n_indices; i++) {
                    const int edge = (config >> offset) & 0xF;
                    //vertices.push_back(edge_indices[edge]);
                    vertices[index++] = edge_indices[edge];
                    offset += 4;
                }

                for (int i = 0; i < n_triangles; i++) {
                    triangle(vertices[index_base+i*3+2],
                            vertices[index_base+i*3+1],
                            vertices[index_base+i*3+0]);

                }

                for (int i = 0; i < n_indices; i++) {
                    vertices[index_base+i].normal = normalize(vertices[index_base+i].normal);
                    vertices[index_base+i].position.data[0] /= N;
                    vertices[index_base+i].position.data[1] /= N;
                    vertices[index_base+i].position.data[2] /= N;

/*
                    int color_num = j%table_num;
        
                    vertices[index_base+i].color.data[0] = color_table[3*color_num+0];
                    vertices[index_base+i].color.data[1] = color_table[3*color_num+1];
                    vertices[index_base+i].color.data[2] = color_table[3*color_num+2];

                    if ( j % 4 == 0){dd
                        vertices[index_base+i].color.data[0] = 0.2;
                        vertices[index_base+i].color.data[1] = 1.0;
                        vertices[index_base+i].color.data[2] = 0.2;
                    }
                    else if ( j % 4 == 1){
                        vertices[index_base+i].color.data[0] = 1.0;
                        vertices[index_base+i].color.data[1] = 0.2;
                        vertices[index_base+i].color.data[2] = 0.2;
                    }
                    else if ( j % 4 == 2){
                        vertices[index_base+i].color.data[0] = 0.2;
                        vertices[index_base+i].color.data[1] = 0.2;
                        vertices[index_base+i].color.data[2] = 1.0;
                    }else {
                        vertices[index_base+i].color.data[0] = 0.2;
                        vertices[index_base+i].color.data[1] = 1.0;
                        vertices[index_base+i].color.data[2] = 1.0;
                    }
*/
                }


            }

        }
    }
}


/*

//__global__ void mc_kernel(Vertex* vertices,float* voxels, int bidx, int boff, int dimx, int dimy, int dimz, unsigned int* num_vert,int* isolist, int start, int end, int N)
__global__ void composite(char* output, char* input, int num_img, int dim_x, int dim_y, int dummy1, int dummy2, unsigned int* num_vert) {
    int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_y = threadIdx.y + blockDim.y * blockIdx.y;

    if(dim_x <= idx_x || dim_y <= idx_y) return;
	int index   = (idx_y * dim_x + idx_x);
    
    char* img1 = input;
    char* img2 = input+dim_x*dim_y*7;
    char* img  = img1;

    float *depth_table1 = (float*) (img1 + dim_x*dim_y*3);
    float *depth_table2 = (float*) (img2 + dim_x*dim_y*3);

    float val1 = depth_table1[index];
    float val2 = depth_table2[index];

//    unsigned char* image = (unsigned char*) input;

    if (val1 > val2) 	
        img = img2;

    //RGB Copy
    output[3*index+0] = img[3*index+0];
    output[3*index+1] = img[3*index+1];
    output[3*index+2] = img[3*index+2];

    //Depth Copy 
    output[3*dim_x*dim_y + 4*index+0] = img[3*dim_x*dim_y + 4*index+0]; 
    output[3*dim_x*dim_y + 4*index+1] = img[3*dim_x*dim_y + 4*index+1]; 
    output[3*dim_x*dim_y + 4*index+2] = img[3*dim_x*dim_y + 4*index+2]; 
    output[3*dim_x*dim_y + 4*index+3] = img[3*dim_x*dim_y + 4*index+3]; 
    return ;
}
*/
__global__ void composite(char* output, char* input, int num_img, int dim_x, int dim_y, int dummy1, int dummy2, unsigned int* num_vert) {
    int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_y = threadIdx.y + blockDim.y * blockIdx.y;

    if(dim_x <= idx_x || dim_y <= idx_y) return;
	int index   = (idx_y * dim_x + idx_x);
   

    char* img1 = input;
    float *depth_table1 = (float*) (img1 + dim_x*dim_y*3);
    float val1 = depth_table1[index];

    char* img  = img1;
    //RGB Copy
    output[3*index+0] = img[3*index+0];
    output[3*index+1] = img[3*index+1];
    output[3*index+2] = img[3*index+2];

    for (int i = 1 ; i< num_img ;i++)
    { 
        char* img2 = input+(dim_x*dim_y*7)*i;

        float *depth_table2 = (float*) (img2 + dim_x*dim_y*3);

        float val2 = depth_table2[index];

//    unsigned char* image = (unsigned char*) input;

        if (val1 > val2){	
            val1 = val2;

            //RGB Copy
            output[3*index+0] = img2[3*index+0];
            output[3*index+1] = img2[3*index+1];
            output[3*index+2] = img2[3*index+2];
        }
    } 
    return ;
}
}
