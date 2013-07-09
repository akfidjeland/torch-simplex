SET(src init.c simplexnoise.c)
SET(luasrc init.lua)

FIND_PACKAGE(Torch REQUIRED)
ADD_TORCH_PACKAGE(simplex "${src}" "${luasrc}" "Simplex noise generation")
TARGET_LINK_LIBRARIES(simplex luaT TH)
