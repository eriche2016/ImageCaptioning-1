require 'paths'
local anno_utils = require 'utils.anno_utils_filter'
local tablex = require('pl.tablex')

local M = {}
local DataLoader = torch.class('lstm.DataLoader', M)

function DataLoader:__init(opt)
    print('Initialize dataloader...')
    self.feat_dirs = {}
    table.insert(self.feat_dirs, paths.concat(opt.data, opt.train_feat))
    table.insert(self.feat_dirs, paths.concat(opt.data, opt.val_feat))
    self.anno_dirs = {}
    table.insert(self.anno_dirs, paths.concat(opt.data, opt.train_anno))
    table.insert(self.anno_dirs, paths.concat(opt.data, opt.val_anno))
    self.fc7_dirs = {}
    table.insert(self.fc7_dirs, paths.concat(opt.data, opt.train_fc7))
    table.insert(self.fc7_dirs, paths.concat(opt.data, opt.val_fc7))

    self.att_size = opt.att_size
    self.feat_size = opt.feat_size
    self.fc7_size = opt.fc7_size

    self.anno_utils = anno_utils

    -- Prepare captions
    self.id2file, self.train_ids, self.val_ids = anno_utils.read_dataset(self.feat_dirs, '.dat')
    self.id2fc7_file, _, _ = anno_utils.read_dataset(self.fc7_dirs, '.dat')
    self.id2captions, self.word2index, self.index2word, self.word_cnt = anno_utils.read_captions(self.anno_dirs, nil)

    print('Dataset summary:')
    print('id2file size: ' .. tablex.size(self.id2file))
    print('id2captions size: ' .. tablex.size(self.id2captions))
    print('word2index size: '.. tablex.size(self.word2index))
    print('word_cnt: ' .. self.word_cnt)
    
    -- This is for model
    opt.word_cnt = self.word_cnt
    
    -- Split dataset into train, val and test
    self.train_set, self.val_set, self.test_set = anno_utils.split_dataset(self.val_ids, opt.val_size, opt.test_size)
    print('validation set size: ' .. tablex.size(self.val_set))
    print('test set size: ' .. tablex.size(self.test_set))
    
    local cur_n = #self.train_set
    for i = 1, #self.train_ids do
        self.train_set[cur_n + i] = self.train_ids[i]
    end
    
    print('train set size: ' .. tablex.size(self.train_set))

    -- Generate len2captions
    self.train_len2captions = anno_utils.gen_len2captions(self.train_set, self.id2captions)
    self.val_len2captions = anno_utils.gen_len2captions(self.val_set, self.id2captions)
    print('size of train_len2captions: ' .. tablex.size(self.train_len2captions))
    print('size of val_len2captions: ' .. tablex.size(self.val_len2captions))
    
    local max_seq_len = 1  
    for k,v in pairs(self.train_len2captions) do
        -- print(k)
        -- if k > opt.max_seq_len then opt.max_seq_len = k end
        max_seq_len = math.max(max_seq_len, k)
    end               
    opt.max_seq_len = max_seq_len + 2
    print('Max sequence length is: ' .. opt.max_seq_len)
    -- os.exit()
end

--------------------------------------------------------
-- genrate training data
-- INPUT
-- batch: a table of captions
-- OUTPUT
-- image: a tensor, 196*512
--------------------------------------------------------
function DataLoader:gen_train_data(batch)
    -- local images = torch.CudaTensor(#batch, 512, 14, 14)
    local images = torch.CudaTensor(#batch, self.att_size, self.feat_size)
    local input_text = torch.CudaTensor(#batch, #batch[1][2] + 1)
    local output_text = torch.CudaTensor(#batch, #batch[1][2] + 1)
    local fc7_images = torch.CudaTensor(#batch, self.fc7_size)
    for i = 1, #batch do
        -- caption = {id, caption}
        local id, caption = batch[i][1], batch[i][2]
        -- local file = files[id2index[id]]
        local file = self.id2file[id]
        local fc7_file = self.id2fc7_file[id]
        -- from 512*14*14
        images[i]:copy(torch.load(file):reshape(self.feat_size, self.att_size):transpose(1, 2))
        fc7_images[i]:copy(torch.load(fc7_file))
        for j = 1, #caption do
            input_text[i][j + 1] = caption[j]
            output_text[i][j] = caption[j]
        end
        input_text[i][1] = anno_utils.START_NUM
        output_text[i][#caption + 1] = anno_utils.STOP_NUM
    end
    return images, fc7_images, input_text, output_text
end

                
-----------------------------------------------------
-- generate batch with same length for training
-----------------------------------------------------
function DataLoader:gen_batch(len2captions, batch_size)
    local batches = {}
    for len, captions in pairs(len2captions) do
        local i = 1; local j = batch_size
        while i <= #captions do
            j = j <= #captions and j or #captions
            local batch = {}
            for k = i, j do table.insert(batch, captions[k]) end
            table.insert(batches, batch)
            i = j + 1; j = i + batch_size - 1
        end
    end
    return batches
end

return M.DataLoader
                
                
