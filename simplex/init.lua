require 'torch' 
require 'libsimplex'
require 'image'


simplex = {}


--[[! Generate basic simplex noise

@param dim (number) dimension of square output
@param scale (optional number) spatial scale of the noise (default = 0.1)

@return function ∷ Tensor2D. The memory is owned by the generator.

--]]
function simplex.noise2d(dim, scale)

    scale = scale or 0.1
    local noise = torch.FloatTensor(dim, dim):zero()

    return function()
        libsimplex.generate(noise, scale)
        return noise
    end
end



--[[! Generate fractal noise

@return function ∷ Tensor2D. The memory is owned by the generator.

--]]
local function fractal(_dim, scale, abs)

    local dim = _dim
    local out = torch.FloatTensor(dim, dim)
    local buf = torch.FloatTensor(dim, dim)
    nlevels = math.log(dim, 2)

    local img = {}
    for i = 1,nlevels do
        img[i] = torch.FloatTensor(dim, dim)
        dim = dim/2  
    end

    local dim = _dim
    local mul = 2 / (2^nlevels)

    return function()
        out:zero()
        for i = 1,nlevels do
            libsimplex.generate2d(img[i], scale)
            if abs then
                img[i]:abs()
            end
            image.scale(buf, img[i])
            out:add(mul, buf)
            dim = dim * 0.5 
            mul = mul * 2
        end
        return out
    end
end


--[[! Generate fractal noise

This is a composition of multiple levels of basic simplex noise.

@param dim (number) dimension of square 2D output
@param scale (number) spatial scale of noise

@return function ∷ Tensor2D. The memory is owned by the generator.

--]]
function simplex.fractal(dim, scale)
    return fractal(dim, scale or 0.1, false)
end


--[[! Generate turbulence-like noise

This is a composition of multiple levels of basic simplex noise.

@param dim (number) dimension of square 2D output
@param scale (number) spatial scale of noise

@return function ∷ Tensor2D. The memory is owned by the generator.
--]]
function simplex.turbulence(dim, scale)
    return fractal(dim, scale or 0.1, true)
end


--[[ Generate turbulence-like noise with dominant vertical stripes

@param dim (number) dimension of square 2D output
@param nstripes (number) number of vertical stripes in image
@param scale (number) spatial scale of noise

@return function ∷ Tensor2D. The memory is owned by the generator.

--]]
function simplex.turbulenceVertical(dim, nstripes, scale)

    nstripes = nstripes or 4
    scale = scale or 0.1
    local stripes = torch.linspace(0, math.pi*nstripes, dim):float():resize(1, dim):expand(dim, dim)
    local f = fractal(dim, scale, true)

    return function()
        return f():add(stripes):sin()
    end
end


--[[ Generate turbulence-like noise with dominant horizontal stripes

@param dim (number) dimension of square 2D output
@param nstripes (number) number of vertical stripes in image
@param scale (number) spatial scale of noise

@return function ∷ Tensor2D. The memory is owned by the generator.

--]]
function simplex.turbulenceHorizontal(dim, nstripes, scale)
    nstripes = nstripes or 4
    scale = scale or 0.1
    local stripes = torch.linspace(0, math.pi*nstripes, dim):float():resize(dim, 1):expand(dim, dim)
    local f = fractal(dim, scale, true)

    return function()
        return f():add(stripes):sin()
    end
end


return simplex
