require 'nn'
require 'cunn'
require 'nngraph'
require 'optim'
require 'cudnn'
local model_utils = require 'utils.model_utils'
local eval_utils = require 'eval.neuraltalk2.misc.utils'
local tablex = require 'pl.tablex'

local M = {}

-------------------------------------
-- Train the model for an epoch
-- INPUT
-- batches: {{id, caption}, ..., ...}
-------------------------------------
function M.train(model, opt, batches, val_batches, optim_state, dataloader)
    local optim_epsilon = 1e-8
    local optim_state, cnn_optim_state = {}, {}

    local vgg16_input_fc7_model
    if opt.load_vgg_file then
        vgg16_input_fc7_model = torch.load('models/' .. opt.load_file_name .. '.vgg16_input_fc7_model')
    else
        vgg16_input_fc7_model = torch.load('models/vgg_vd16_input_fc7_cudnn.t7')
    end
    if opt.cnn_dropout then vgg16_input_fc7_model.modules[35].p = 0.5 end
    model_utils.unsanitize_gradients(vgg16_input_fc7_model)
    cnn_params, cnn_grad_params = vgg16_input_fc7_model:getParameters()

    local params, grad_params
    local model_list
    if opt.lstm_size ~= opt.fc7_size then
        if opt.use_noun then
            -- params, grad_params = model_utils.combine_all_parameters(model.emb, model.soft_att_lstm, model.lstm, model.softmax, model.linear, model.reason_softmax, model.pooling)
            model_list = {model.emb, model.lstm, model.softmax, model.linear, model.reason_softmax, model.pooling}
        else    
            model_list = {model.emb, model.lstm, model.softmax, model.linear}
        end
    else
        if opt.use_noun then
            model_list = {model.emb, model.lstm, model.softmax, model.reason_softmax, model.pooling}
        else
            model_list = {model.emb, model.lstm, model.softmax}
        end
    end
    for t = 1, opt.reason_step do
        table.insert(model_list, model.soft_att_lstm[t])
    end
    params, grad_params = model_utils.combine_all_parameters(unpack(model_list))
    local clones = {}
    anno_utils = dataloader.anno_utils
    
    -- Clone models
    local max_t = opt.truncate > 0 and math.min(opt.max_seq_len, opt.truncate) or opt.max_seq_len
    print('actual clone times ' .. max_t)
    for name, proto in pairs(model) do
        print('cloning '.. name)
        if name ~= 'soft_att_lstm' and name ~= 'reason_softmax' and name ~= 'reason_criterion'
            and name ~= 'pooling' and name ~= 'linear' and name ~= 'google_linear' then 
            clones[name] = model_utils.clone_many_times(proto, max_t)
        end
    end
    print('cloning reasoning lstm')
    -- clones.soft_att_lstm = model_utils.clone_many_times(model.soft_att_lstm, opt.reason_step)
    clones.soft_att_lstm = model.soft_att_lstm
    if opt.use_noun then
        clones.reason_softmax = model_utils.clone_many_times(model.reason_softmax, opt.reason_step)
        -- clones.reason_criterion = model_utils.clone_many_times(model.reason_criterion, opt.reason_step)
    end 

    local att_seq, input_text, output_text, noun_list, fc7_google_images, jpg

    local function evaluate()
        for t = 1, opt.reason_step do clones.soft_att_lstm[t]:evaluate() end
        for t = 1, max_t do clones.lstm[t]:evaluate() end
        model.linear:evaluate()
        vgg16_input_fc7_model:evaluate()
    end

    local function training()
        for t = 1, opt.reason_step do clones.soft_att_lstm[t]:training() end
        for t = 1, max_t do clones.lstm[t]:training() end
        model.linear:training()
        vgg16_input_fc7_model:training()
    end

    local function feval(update)
        if update == nil then update = true end
        if update then training() else evaluate() end
        grad_params:zero()
        cnn_grad_params:zero()

        local fc7_images = vgg16_input_fc7_model:forward(jpg)

        local image_map
        if opt.use_google then
            image_map = model.linear:forward{fc7_images, fc7_google_images}
        elseif opt.fc7_size ~= opt.lstm_size then
            image_map = model.linear:forward(fc7_images)
        else
            image_map = fc7_images
        end

        local zero_tensor = torch.zeros(input_text:size()[1], opt.lstm_size):cuda()
        local reason_c = {[0] = image_map}
        local reason_h = {[0] = image_map}
        local reason_h_att = torch.CudaTensor(input_text:size()[1], opt.reason_step, opt.lstm_size)
        local embeddings, lstm_c, lstm_h, predictions, reason_preds = {}, {}, {}, {}, {}
        local reason_pred_mat = torch.CudaTensor(input_text:size()[1], opt.reason_step, opt.word_cnt)
        local loss = 0
        local seq_len = math.min(input_text:size()[2], max_t)
        local reason_len = opt.reason_step
        
        for t = 1, reason_len do
            reason_c[t], reason_h[t] = unpack(clones.soft_att_lstm[t]:
                forward{att_seq, reason_c[t - 1], reason_h[t - 1]})
            reason_h_att:select(2, t):copy(reason_h[t])
            if opt.use_noun then
                reason_preds[t] = clones.reason_softmax[t]:forward(reason_h[t])
                reason_pred_mat:select(2, t):copy(reason_preds[t])
            end
        end

        local reason_pool
        local loss_2 = 0
        if opt.use_noun then
            reason_pool = model.pooling:forward(reason_pred_mat):float()
            local t_loss = model.reason_criterion:forward(reason_pool, noun_list) * opt.reason_weight
            -- if update then loss = loss + t_loss else loss_2 = loss_2 + t_loss end
            loss_2 = loss_2 + t_loss
        end

        lstm_c[0] = reason_c[reason_len]
        lstm_h[0] = reason_h[reason_len]

        for t = 1, seq_len do
            embeddings[t] = clones.emb[t]:forward(input_text:select(2, t))
            lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:
                forward{embeddings[t], reason_h_att, lstm_c[t - 1], lstm_h[t - 1]})
            predictions[t] = clones.softmax[t]:forward(lstm_h[t])
            loss = loss + clones.criterion[t]:forward(predictions[t], output_text:select(2, t))
        end

        if update then
            local dreason_pred
            if opt.use_noun then
                dreason_pred = model.reason_criterion:backward(reason_pool, noun_list):cuda() * opt.reason_weight
                dreason_pred = model.pooling:backward(reason_pred_mat, dreason_pred)
            end

            local dembeddings, dlstm_c, dlstm_h, dreason_c, dreason_h = {}, {}, {}, {}, {}
            local dreason_h_att = torch.CudaTensor(input_text:size()[1], opt.reason_step, opt.lstm_size):zero()
            dlstm_c[seq_len] = zero_tensor:clone()
            dlstm_h[seq_len] = zero_tensor:clone()

            for t = seq_len, 1, -1 do
                local doutput_t = clones.criterion[t]:backward(predictions[t], output_text:select(2, t))
                dlstm_h[t]:add(clones.softmax[t]:backward(lstm_h[t], doutput_t))
                dembeddings[t], doutput_t, dlstm_c[t - 1], dlstm_h[t - 1] = unpack(clones.lstm[t]:
                    backward({embeddings[t], reason_h_att, lstm_c[t - 1], lstm_h[t - 1]},
                    {dlstm_c[t], dlstm_h[t]}))
                dreason_h_att:add(doutput_t)
                clones.emb[t]:backward(input_text:select(2, t), dembeddings[t])
            end
            dreason_c[reason_len] = dlstm_c[0]
            dreason_h[reason_len] = dlstm_h[0]
            for t = reason_len, 1, -1 do
                if opt.use_noun then
                    local doutput_t = clones.reason_softmax[t]:backward(reason_h[t], dreason_pred:select(2, t))
                    dreason_h[t]:add(doutput_t)
                end
                dreason_h[t]:add(dreason_h_att:select(2, t))
                _, dreason_c[t - 1], dreason_h[t - 1] = unpack(clones.soft_att_lstm[t]:
                    backward({att_seq, reason_c[t - 1], reason_h[t - 1]},
                    {dreason_c[t], dreason_h[t]}))
            end
            local d_fc7_images
            if opt.use_google then
                dreason_c[0]:add(dreason_h[0])
                d_fc7_images, _ = unpack(model.linear:backward({fc7_images, fc7_google_images}, dreason_c[0]))
            elseif opt.fc7_size ~= opt.lstm_size then
                dreason_c[0]:add(dreason_h[0])
                d_fc7_images = model.linear:backward(fc7_images, dreason_c[0])
            end
            vgg16_input_fc7_model:backward(jpg, d_fc7_images)

            grad_params:clamp(-5, 5)
            cnn_grad_params:clamp(-5, 5)
        end

        -- return loss, update and grad_params or loss_2
        return loss, loss_2
    end 
    --- end of feval

    local function comp_error(batches)
        local loss = 0
        local loss_2 = 0
        for j = 1, opt.max_eval_batch do
            if j > #batches then break end
            att_seq, _, input_text, output_text, noun_list, fc7_google_images, jpg = dataloader:gen_train_data(batches[j])
            local t_loss, t_loss_2 = feval(false)
            loss = loss + t_loss
            loss_2 = loss_2 + t_loss_2
        end
        return loss, loss_2
    end
    
    local max_bleu_4 = 0
    for epoch = 1, opt.nEpochs do
        local index = torch.randperm(#batches)
        for i = 1, #batches do
            att_seq, _, input_text, output_text, noun_list, fc7_google_images, jpg = dataloader:gen_train_data(batches[index[i]])
            feval(true)
            -- model_utils.adagrad(params, grad_params, opt.LR, optim_epsilon, optim_state)
            -- model_utils.adagrad(cnn_params, cnn_grad_params, opt.cnn_LR, optim_epsilon, cnn_optim_state)
            model_utils.adam(params, grad_params, opt.LR, 0.8, 0.999, optim_epsilon, optim_state)
            model_utils.adam(cnn_params, cnn_grad_params, opt.cnn_LR, 0.8, 0.999, optim_epsilon, cnn_optim_state)
            
            ----------------- Evaluate the model in validation set ----------------
            if i == 1 or i % opt.loss_period == 0 then
                evaluate()
                train_loss, train_loss_2 = comp_error(batches)
                val_loss, val_loss_2 = comp_error(val_batches)
                print(epoch, i, 'train', train_loss, train_loss_2, 'val', val_loss, val_loss_2)
                collectgarbage()
            end

            if i == 1 or i % opt.eval_period == 0 then
                evaluate()
                local captions = {}
                local j1 = 1
                while j1 <= #dataloader.val_set do
                    local j2 = math.min(#dataloader.val_set, j1 + opt.val_batch_size)
                    att_seq, _, fc7_google_images, jpg = dataloader:gen_test_data(j1, j2)

                    local fc7_images = vgg16_input_fc7_model:forward(jpg)

                    local image_map
                    if opt.use_google then
                        image_map = model.linear:forward{fc7_images, fc7_google_images}
                    elseif opt.fc7_size ~= opt.lstm_size then
                        image_map = model.linear:forward(fc7_images)
                    else
                        image_map = fc7_images
                    end

                    local reason_c = {[0] = image_map}
                    local reason_h = {[0] = image_map}
                    local reason_h_att = torch.CudaTensor(att_seq:size()[1], opt.reason_step, opt.lstm_size)
                    local embeddings, lstm_c, lstm_h, predictions, max_pred = {}, {}, {}, {}, {}
                    local reason_len = opt.reason_step
                    local seq_len = max_t
                    
                    for t = 1, reason_len do
                        reason_c[t], reason_h[t] = unpack(clones.soft_att_lstm[t]:
                            forward{att_seq, reason_c[t - 1], reason_h[t - 1]})
                        reason_h_att:select(2, t):copy(reason_h[t])
                    end

                    lstm_c[0] = reason_c[reason_len]
                    lstm_h[0] = reason_h[reason_len]
                    max_pred[1] = torch.CudaTensor(att_seq:size()[1]):fill(anno_utils.START_NUM)

                    for t = 1, seq_len do
                        embeddings[t] = clones.emb[t]:forward(max_pred[t])
                        lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:
                            forward{embeddings[t], reason_h_att, lstm_c[t - 1], lstm_h[t - 1]})
                        predictions[t] = clones.softmax[t]:forward(lstm_h[t])
                        _, max_pred[t + 1] = torch.max(predictions[t], 2)
                        max_pred[t + 1] = max_pred[t + 1]:view(-1)
                    end

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

                local eval_struct = M.language_eval(captions, 'attention')
                local bleu_4
                if opt.early_stop == 'cider' then
                    bleu_4 = eval_struct.CIDEr
                else
                    bleu_4 = eval_struct.Bleu_4
                end

                if bleu_4 > max_bleu_4 then
                    max_bleu_4 = bleu_4
                    if opt.save_file then
                        vgg16_input_fc7_model:clearState()
                        torch.save('models/' .. opt.save_file_name .. '.vgg16_input_fc7_model', vgg16_input_fc7_model)
                        torch.save('models/' .. opt.save_file_name, model)
                    end
                end
                if opt.early_stop == 'cider' then
                    print(epoch, i, 'max_cider', max_bleu_4, 'cider', bleu_4)
                else
                    print(epoch, i, 'max_bleu', max_bleu_4, 'bleu', bleu_4)
                end
            end
            
            ::continue::
        end
        -- end of for i
    end
    -- end of for epoch
end

-------------------------
-- Eval the model
-------------------------
function M.language_eval(predictions, id)
    print('using reasoning att')
    local out_struct = {val_predictions = predictions}
    eval_utils.write_json('eval/neuraltalk2/coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
    os.execute('./eval/neuraltalk2/misc/call_python_caption_eval.sh val' .. id .. '.json')
    local result_struct = eval_utils.read_json('eval/neuraltalk2/coco-caption/val' .. id .. '.json_out.json')
    return result_struct
end

return M




