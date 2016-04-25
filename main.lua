require 'nn'
require 'nngraph'
require 'torch'
require 'cutorch'
local DataLoader = require 'dataloader'
local M = require 'soft_att_lstm'

torch.setdefaulttensortype('torch.FloatTensor')
local opts = require 'opts'
local opt = opts.parse(arg)
print(opt)
cutorch.setDevice(opt.nGPU)
torch.manualSeed(opt.seed)

-- Initialize dataloader
local dataloader = DataLoader(opt)

-- Create model
local model = M.create_model(opt) 
print(model.emb)
print(model.soft_att_lstm)
print(model.softmax)
-- print(model.softmax.modules[1].bias)
-- os.exit()
print(model.criterion)
-- os.exit()

-- Train
optim_state = {learningRate = opt.LR}
for epoch = 1, opt.nEpochs do
    local batches = dataloader:gen_batch(dataloader.train_len2captions, opt.batch_size)
    M.train(model, epoch, opt, batches, optim_state, dataloader)
end

    