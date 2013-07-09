#include <TH.h>
#include <luaT.h>
#include "simplexnoise.h"

const char* torch_FloatTensor_id = NULL;

int
simplex_generate2d(lua_State* L)
{
	if(lua_gettop(L) != 2) {
    	THError("invalid number of arguments. Expects: FloatTensor2D float");
	}	

	THFloatTensor* img = luaT_checkudata(L, 1, torch_FloatTensor_id);
	float scale = luaL_checknumber(L, 2);

	THArgCheck(THFloatTensor_nDimension(img) == 2, 1, "image tensor not 2D");
	THArgCheck(THFloatTensor_isContiguous(img), 1, "image tensor not contiguous");

	const long height = THFloatTensor_size(img, 0);
	const long width = THFloatTensor_size(img, 1);

	snoise_permtable permtable; 
	snoise_setup_perm(&permtable, mt_genrand_int32());

	float* data = THFloatTensor_data(img);
	
	size_t y;
	#pragma omp parallel for private(y)
	for(y=0; y < height; y++) {
		float* row = data + y*width;
		size_t x;
		for(x=0; x < width; x++) {
			*row++ = snoise2(&permtable, x*scale, y*scale);
		}
	}

	return 0;
}

static const struct luaL_Reg simplex_init [] = {
	{"generate2d", simplex_generate2d},
	{NULL, NULL}
};


int
luaopen_libsimplex(lua_State *L)
{
	torch_FloatTensor_id = luaT_typenameid(L, "torch.FloatTensor");
	luaL_register(L, "libsimplex", simplex_init);
	return 1;
}
