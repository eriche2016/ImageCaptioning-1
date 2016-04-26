require 'nn'
require 'cunn'
require 'nngraph'
require 'optim'
local model_utils = require 'utils.model_utils'
local tablex = require 'pl.tablex'

local M = {}
-- local ATT_NEXT_H = false

-- cmd:option('-emb_size', 100, 'Word embedding size')
-- cmd:option('-lstm_size', 4096, 'LSTM size')
-- cmd:option('-word_cnt', 9520, 'Vocabulary size')
-- cmd:option('-att_size', 196, 'Attention size')
-- cmd:option('-feat_size', 512, 'Feature size for each attention')
-- cmd:option('-batch_size', 32, 'Batch size in SGD')
function M.soft_att_lstm(opt)
    -- Model parameters
    local feat_size = opt.feat_size
    local att_size = opt.att_size
    local batch_size = opt.batch_size
    local rnn_size = opt.lstm_size
    local input_size = opt.emb_size

    local x = nn.Identity()()         -- batch * input_size -- embedded caption at a specific step
    local att_seq = nn.Identity()()   -- batch * att_size * feat_size -- the image patches
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    ------------ Attention part --------------------
    local att = nn.View(-1, feat_size)(att_seq)
    att = nn.Linear(feat_size, rnn_size)(att)
    att = nn.View(-1, att_size, rnn_size)(att)                 -- batch * att_size * rnn_size <- batch * att_size * feat_size

    local dot = nn.MixtureTable(3){prev_h, att}                     -- batch * att_size <- (batch * rnn_size, batch * att_size * rnn_size)
    local weight = nn.SoftMax()(dot)                           -- batch * att_size
    local att_t = nn.Transpose({2, 3})(att)                    -- batch * rnn_size * att_size
    att = nn.MixtureTable(3){weight, att_t}                         -- batch * rnn_size <- (batch * att_size, batch * rnn_size * att_size)
    
    --- Input to LSTM
    local att_add = nn.Linear(rnn_size, 4 * rnn_size)(att) -- batch * (4*rnn_size) <- batch * rnn_size

    ------------- LSTM main part --------------------
    local i2h = nn.Linear(input_size, 4 * rnn_size)(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
    
    -- test
    -- local prev_all_input_sums = nn.CAddTable()({i2h, h2h})
    -- local all_input_sums = nn.CAddTable()({prev_all_input_sums, att_add})

    local all_input_sums = nn.CAddTable()({i2h, h2h, att_add})

    local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)

    local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)

    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) -- batch * rnn_size
      
    return nn.gModule({x, att_seq, prev_c, prev_h}, {next_c, next_h})
    
end


-------------------------------------
-- Train the model for an epoch
-- INPUT
-- batches: {{id, caption}, ..., ...}
-------------------------------------
function M.train(model, epoch, opt, batches, val_batches, optim_state, dataloader)
    local params, grad_params = model_utils.combine_all_parameters(model.emb, model.soft_att_lstm, model.softmax)
    local clones = {}
    anno_utils = dataloader.anno_utils
    
    -- Clone models
    local max_t = opt.truncate > 0 and math.min(opt.max_seq_len, opt.truncate) or opt.max_seq_len
    print('actual clone times ' .. max_t)
    for name, proto in pairs(model) do
        print('cloning '.. name)
        clones[name] = model_utils.clone_many_times(proto, max_t)
    end

    local att_seq, fc7_images, input_text, output_text

    local function feval(params_, update)
        if update == nil then update = true end
        if params_ ~= params then
            params:copy(params_)
        end
        grad_params:zero()

        local initstate_c = fc7_images:clone()
        local initstate_h = fc7_images
        local dfinalstate_c = torch.zeros(input_text:size()[1], opt.lstm_size):cuda()
        
        -- print('Start forward')
        ------------------- forward pass -------------------
        local embeddings = {}              -- input text embeddings
        local lstm_c = {[0]=initstate_c}   -- internal cell states of LSTM
        local lstm_h = {[0]=initstate_h}   -- output values of LSTM
        local predictions = {}             -- softmax outputs
        local loss = 0
        local seq_len = input_text:size()[2]     -- sequence length 
        seq_len = math.min(seq_len, max_t) -- get truncated
        
        for t = 1, seq_len do
            -- print('Time step ' .. t)
            embeddings[t] = clones.emb[t]:forward(input_text:select(2, t))    -- emb forward
            lstm_c[t], lstm_h[t] = unpack(clones.soft_att_lstm[t]:            -- lstm forward
                forward{embeddings[t], att_seq, lstm_c[t-1], lstm_h[t-1]})    
            
            predictions[t] = clones.softmax[t]:forward(lstm_h[t])             -- softmax forward
            loss = loss + clones.criterion[t]:forward(predictions[t], output_text:select(2, t))    -- criterion forward
        end
                    
        ------------------- backward pass -------------------
        if update then
            local dembeddings = {}                                    -- d loss / d input embeddings
            local dlstm_c = {[seq_len]=dfinalstate_c}                 -- internal cell states of LSTM
            local dlstm_h = {}                                        -- output values of LSTM
            
            for t = seq_len, 1, -1 do
                -- print('Time step ' .. t)
                local doutput_t = clones.criterion[t]:backward(predictions[t], output_text:select(2, t))  -- criterion backward
                if t == seq_len then
                    assert(dlstm_h[t] == nil)
                    dlstm_h[t] = clones.softmax[t]:backward(lstm_h[t], doutput_t)
                else
                    dlstm_h[t]:add(clones.softmax[t]:backward(lstm_h[t], doutput_t))     -- softmax backward
                end
                
                -- backprop through LSTM timestep
                dembeddings[t], _, dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.soft_att_lstm[t]:
                    backward({embeddings[t], att_seq, lstm_c[t-1], lstm_h[t-1]},
                    {dlstm_c[t], dlstm_h[t]}))                                           -- lstm backward

                -- backprop through embeddings
                clones.emb[t]:backward(input_text:select(2, t), dembeddings[t])          -- emb backward
            end
        end
        
        grad_params:clamp(-5, 5)
        return loss, grad_params
    end 
    --- end of feval

    local function comp_error(batches)
        local loss = 0; local inst_cnt = 0
        for j = 1, #batches do
            att_seq, fc7_images, input_text, output_text = dataloader:gen_train_data(batches[j])
            local t_loss, _ = feval(params, false)
            loss = loss + t_loss
            inst_cnt = inst_cnt + input_text:size()[1]
            if inst_cnt >= opt.max_eval_inst then break end
        end
        return loss
    end
    
    local max_bleu_4 = 0
    local index = torch.randperm(#batches)
    for i = 1, #batches do
        att_seq, fc7_images, input_text, output_text = dataloader:gen_train_data(batches[index[i]])
        optim.adagrad(feval, params, optim_state)
        
        ----------------- Evaluate the model in validation set ----------------
        if i == 1 or i % opt.loss_period == 0 then
            train_loss = comp_error(batches)
            val_loss = comp_error(val_batches)
            print(epoch, i, 'train', train_loss, 'val', val_loss)
            collectgarbage()
        end

        if i == 1 or i % opt.eval_period == 0 then
            local captions = {}
            for j = 1, #val_batches do
                att_seq, fc7_images, input_text, output_text = dataloader:gen_train_data(val_batches[j])

                local initstate_c = fc7_images:clone()
                local initstate_h = fc7_images
                local init_input = torch.CudaTensor(input_text:size()[1]):fill(anno_utils.START_NUM)
                
                ------------------- forward pass -------------------
                local embeddings = {}              -- input text embeddings
                local lstm_c = {[0]=initstate_c}   -- internal cell states of LSTM
                local lstm_h = {[0]=initstate_h}   -- output values of LSTM
                local predictions = {}             -- softmax outputs
                local max_pred = {[1] = init_input}                -- max outputs 
                local seq_len = input_text:size()[2]     -- sequence length 
                seq_len = math.min(seq_len, max_t) -- get truncated
                
                for t = 1, seq_len do
                    print(max_pred[t]:size())
                    embeddings[t] = clones.emb[t]:forward(max_pred[t])    -- emb forward
                    lstm_c[t], lstm_h[t] = unpack(clones.soft_att_lstm[t]:            -- lstm forward
                        forward{embeddings[t], att_seq, lstm_c[t-1], lstm_h[t-1]})    
                    predictions[t] = clones.softmax[t]:forward(lstm_h[t])             -- softmax forward
                    print(torch.max(predictions[t], 2)[2]:size())
                    max_pred[t + 1] = torch.max(predictions[t], 2)[2]:view(-1)
                    clones.criterion[t]:forward(predictions[t], output_text:select(2, t))    -- criterion forward
                end

                index2word = dataloader.index2word
                for k = 1, input_text:size()[1] do
                    local caption = ''
                    for t = 1, seq_len do
                        local word_index = max_pred[t][l]
                        if word_index == anno_utils.STOP_NUM then break end
                        if caption ~= '' then
                            caption = caption .. ' ' .. index2word[word_index]
                        else
                            caption = index2word[word_index]
                        end
                    end
                    if j == 1 and k <= 10 then
                        print(val_batches[j][k][1], caption)
                    end
                    table.insert(captions, {image_id = val_batches[j][k][1], caption})
                end
            end

            local eval_struct = utils.language_eval(captions, 'attention')
            local bleu_4 = eval_struct.Bleu_4

            if bleu_4 > max_bleu_4 then
                max_bleu_4 = bleu_4
                if opt.save_file then
                    torch.save('models/' .. opt.save_file_name, model)
                end
            end
            print(epoch, i, 'max_bleu', max_bleu_4, 'bleu', bleu_4)
        end
        
        ::continue::
    end
    -- end of for loop    
end


-------------------------
-- create the final model
-------------------------
function M.create_model(opt)
    local model = {}
    model.emb = nn.LookupTable(opt.word_cnt, opt.emb_size)
    model.soft_att_lstm = M.soft_att_lstm(opt) 
    model.softmax = nn.Sequential():add(nn.Linear(opt.lstm_size, opt.word_cnt)):add(nn.LogSoftMax())
    model.criterion = nn.ClassNLLCriterion()
    
    if opt.nGPU > 0 then
        model.emb:cuda()
        model.soft_att_lstm:cuda()
        model.softmax:cuda()
        model.criterion:cuda()
    end
    return model
end

-------------------------
-- Eval the model
-------------------------
function M.language_eval(predictions, id)
    local out_struct = {val_predictions = predictions}
    eval_utils.write_json('eval/neuraltalk2/coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
    os.execute('./eval/neuraltalk2/misc/call_python_caption_eval.sh val' .. id .. '.json')
    local result_struct = eval_utils.read_json('eval/neuraltalk2/coco-caption/val' .. id .. '.json_out.json')
    return result_struct
end

return M




