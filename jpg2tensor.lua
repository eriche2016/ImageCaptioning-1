require 'paths'
require 'image'

local cmd = torch.CmdLine()
cmd:option('-from', 'data/train2014')
cmd:option('-to', 'data/train2014_jpg')

local opt = cmd:parse(arg or {})

local loadSize   = {3, 256, 256}
local sampleSize = {3, 224, 224}

function loadImage(path)
   local input = image.load(path, 3, 'float')
   -- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)
   if input:size(3) < input:size(2) then
      input = image.scale(input, loadSize[2], loadSize[3] * input:size(2) / input:size(3))
   else
      input = image.scale(input, loadSize[2] * input:size(3) / input:size(2), loadSize[3])
   end
   return input
end

-- VGG preprocessing
local bgr_means = {103.939,116.779,123.68}

function vggPreprocess(img)
  local im2 = img:clone()
  im2[{1,{},{}}] = img[{3,{},{}}]
  im2[{3,{},{}}] = img[{1,{},{}}]

  im2:mul(255)
  for i=1,3 do
    im2[i]:add(-bgr_means[i])
  end
  return im2
end

function centerCrop(input)
   local oH = sampleSize[2]
   local oW = sampleSize[3]
   local iW = input:size(3)
   local iH = input:size(2)
   local w1 = math.ceil((iW-oW)/2)
   local h1 = math.ceil((iH-oH)/2)
   local out = image.crop(input, w1, h1, w1+oW, h1+oW) -- center patch
   return out
end

-- function to load the image
local extractHook = function(path)
   collectgarbage()
   local input = loadImage(path)
   local vggPreprocessed = vggPreprocess(input)
   local out = centerCrop(vggPreprocessed)
   return out
end

cnt = 0
for file in paths.files(opt.from) do
    if file:find('.jpg') then
        local im = extractHook(paths.concat(opt.from, file))
        local name = string.sub(file, 1, file:len() - 4)
        torch.save(opt.to .. name .. '.dat', im)

        cnt = cnt + 1
        if cnt % 10000 == 0 then
            print('done', cnt)
        end
    end
end
