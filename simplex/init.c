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
	
	//! \todo do this in OMP parallel
	for(size_t y=0; y < height; y++) {
		for(size_t x=0; x < width; x++) {
			//! \todo replace this 'set' with faster methods
			THFloatTensor_set2d(img, x, y, snoise2(&permtable, x*scale, y*scale));
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
