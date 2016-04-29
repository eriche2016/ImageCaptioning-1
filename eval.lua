require 'nn'
require 'nngraph'
require 'torch'
require 'cutorch'
local DataLoader = require 'dataloader'
local M = require 'soft_att_lstm'
local model_utils = require 'utils.model_utils'
local eval_utils = require 'eval.neuraltalk2.misc.utils'

torch.setdefaulttensortype('torch.FloatTensor')
local opts = require 'opts'
local opt = opts.parse(arg)
print(opt)
cutorch.setDevice(opt.nGPU)
torch.manualSeed(opt.seed)

-- Initialize dataloader
local dataloader = DataLoader(opt)

-- Load model
local model = torch.load(opt.model)

---------------- Beam Search ---------------------
function beam_search(model, dataloader, opt)
    local max_t = opt.truncate > 0 and math.min(opt.max_seq_len, opt.truncate) or opt.max_seq_len
    print('actual clone times ' .. max_t)
    local clones = {}
    local anno_utils = dataloader.anno_utils
    
    for name, proto in pairs(model) do
        print('cloning '.. name)
        if name ~= 'linear' then 
            clones[name] = model_utils.clone_many_times(proto, max_t)
        end
    end
    
    local captions = {}
    local j1 = 1
    while j1 <= #dataloader.val_set do
        collectgarbage()
        
        -- Load a batch of data
        local j2 = math.min(#dataloader.val_set, j1 + opt.val_batch_size)
        local att_seq, fc7_images = dataloader:gen_test_data(j1, j2)
        local image_map
        if opt.lstm_size ~= opt.fc7_size then
            image_map = model.linear:forward(fc7_images)
        else
            image_map = fc7_images
        end
        
        -- print(#image_map)
        -- print(#att_seq)
        -- os.exit()
        -- Traverse each image in the batch
        for k = 1, att_seq:size()[1] do
            
            
            
            
        end
    end
    
end

beam_search(model, dataloader, opt)
os.exit()

---------------- Greedy Search -------------------
function greedy_search(model, dataloader, opt) 
    local max_t = opt.truncate > 0 and math.min(opt.max_seq_len, opt.truncate) or opt.max_seq_len
    print('actual clone times ' .. max_t)
    local clones = {}
    local anno_utils = dataloader.anno_utils
    
    for name, proto in pairs(model) do
        print('cloning '.. name)
        if name ~= 'linear' then 
            clones[name] = model_utils.clone_many_times(proto, max_t)
        end
    end
    
    local captions = {}
    local j1 = 1
    while j1 <= #dataloader.val_set do
        local j2 = math.min(#dataloader.val_set, j1 + opt.val_batch_size)
        att_seq, fc7_images = dataloader:gen_test_data(j1, j2)

        local image_map
        if opt.lstm_size ~= opt.fc7_size then
            image_map = model.linear:forward(fc7_images)
        else
            image_map = fc7_images
        end

        local initstate_c = image_map:clone()
        local initstate_h = image_map
        local init_input = torch.CudaTensor(att_seq:size()[1]):fill(anno_utils.START_NUM)
                    
        ------------------- Forward pass -------------------
        local embeddings = {}              -- input text embeddings
        local lstm_c = {[0]=initstate_c}   -- internal cell states of LSTM
        local lstm_h = {[0]=initstate_h}   -- output values of LSTM
        local predictions = {}             -- softmax outputs
        local max_pred = {[1] = init_input}          -- max outputs 
        local seq_len = max_t     -- sequence length 
                    
        for t = 1, seq_len do
            embeddings[t] = clones.emb[t]:forward(max_pred[t])    -- emb forward
            if opt.use_attention then
                lstm_c[t], lstm_h[t] = unpack(clones.soft_att_lstm[t]:            -- lstm forward
                    forward{embeddings[t], att_seq, lstm_c[t-1], lstm_h[t-1]})
            else
                lstm_c[t], lstm_h[t] = unpack(clones.soft_att_lstm[t]:
                    forward{embeddings[t], lstm_c[t - 1], lstm_h[t - 1]})
            end    
                predictions[t] = clones.softmax[t]:forward(lstm_h[t])             -- softmax forward
                _, max_pred[t + 1] = torch.max(predictions[t], 2)
                max_pred[t + 1] = max_pred[t + 1]:view(-1)
        end

        ------------------- Get captions -------------------
        index2word = dataloader.index2word
        for k = 1, att_seq:size()[1] do
            local caption = ''
            for t = 2, seq_len do
                local word_index = max_pred[t][k]
                if word_index == anno_utils.STOP_NUM then break end
                if caption ~= '' then
                    caption = caption .. ' ' .. index2word[word_index]
                else
                    caption = index2word[word_index]
                end
            end
            if j1 + k - 1 <= 10 then
                print(dataloader.val_set[j1 + k - 1], caption)
            end
            table.insert(captions, {image_id = dataloader.val_set[j1 + k - 1], caption = caption})
        end
        j1 = j2 + 1
    end
    
    -- Evaluate it
    local eval_struct = M.language_eval(captions, 'greedy_concat.1024.128.model')
end

greedy_search(model, dataloader, opt)























